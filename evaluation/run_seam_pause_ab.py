"""Create blind, source-preserving pause comparisons at selected unit seams.

This is an evaluation tool. It reads a completed schema-5 run and writes only
short comparison WAVs and a manifest under a separate output directory.
"""

import argparse
from array import array
import hashlib
import json
from pathlib import Path
import sys
import wave

from src.audiobook.cosyvoice import file_sha256
from src.audiobook.unit_execution import _load, _read_qc_evidence


MINIMUM_PAUSE_MS = (150, 200, 250, 300)
RMS_THRESHOLD_DBFS = -55
PEAK_THRESHOLD_DBFS = -45
WINDOW_MS = 10
CONTEXT_SECONDS = 4


def _quiet_window(samples, rms_limit, peak_limit):
    return (max(abs(value) for value in samples) <= peak_limit
            and sum(value * value for value in samples) <=
            len(samples) * rms_limit * rms_limit)


def edge_quiet_frames(pcm, rate, side):
    """Count contiguous quiet 10 ms windows from the true WAV edge."""
    if side not in {"leading", "trailing"} or rate <= 0 or len(pcm) % 2:
        raise ValueError("Expected an edge direction and complete PCM16 frames.")
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    width = round(rate * WINDOW_MS / 1000)
    if width <= 0:
        raise ValueError("Sample rate cannot resolve a 10 ms window.")
    rms_limit = 32768 * 10 ** (RMS_THRESHOLD_DBFS / 20)
    peak_limit = 32768 * 10 ** (PEAK_THRESHOLD_DBFS / 20)
    quiet = 0
    if side == "leading":
        positions = range(0, len(samples) - width + 1, width)
    else:
        positions = range(len(samples) - width, -1, -width)
    for start in positions:
        if not _quiet_window(samples[start:start + width], rms_limit, peak_limit):
            break
        quiet += width
    return quiet


def _read_selected_wav(run_directory, unit, attempt, rate):
    path = (run_directory / attempt["output_path"]).resolve()
    path.relative_to(run_directory)
    if file_sha256(path) != attempt["wav_sha256"]:
        raise ValueError(f"{unit['id']} selected WAV hash differs from the manifest.")
    with wave.open(str(path), "rb") as audio:
        if (audio.getcomptype(), audio.getnchannels(), audio.getsampwidth(),
                audio.getframerate()) != ("NONE", 1, 2, rate):
            raise ValueError(f"{unit['id']} is not the expected mono PCM16 WAV.")
        frames = audio.getnframes()
        pcm = audio.readframes(frames)
    if (len(pcm) != frames * 2 or frames != attempt["audio"]["frames"]):
        raise ValueError(f"{unit['id']} selected WAV payload is incomplete.")
    return pcm


def _source_context(left, right):
    left_text = left["source_text"].rstrip()
    right_text = right["source_text"].lstrip()
    trailing = left["source_text"][len(left_text):].replace("\r", "")
    if trailing.count("\n") >= 2:
        structure = "paragraph_break"
    elif "\n" in trailing:
        structure = "line_break"
    else:
        structure = "inline"
    return {
        "structure": structure,
        "left_final_character": left_text[-1:] or None,
        "left_final_punctuation": next(
            (character for character in reversed(left_text)
             if character in "。！？!?；;，,：:"), None),
        "left_tail": left_text[-28:],
        "right_head": right_text[:28],
        "trailing_source_newlines": trailing.count("\n"),
    }


def _write_clip(path, rate, pcm):
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        output.writeframes(pcm)
    with wave.open(str(path), "rb") as check:
        if check.readframes(check.getnframes()) != pcm:
            raise ValueError(f"Comparison clip does not preserve its PCM: {path}")


def inserted_frames(minimum_ms, existing_quiet_frames, rate):
    """Return only the silence missing from the requested seam minimum."""
    if minimum_ms < 0 or existing_quiet_frames < 0 or rate <= 0:
        raise ValueError("Pause, existing silence, and sample rate must be valid.")
    return max(0, round(rate * minimum_ms / 1000) - existing_quiet_frames)


def build_experiment(run_directory, output_directory, after_units):
    run_directory, _, manifest = _load(run_directory)
    output_directory = Path(output_directory).expanduser().resolve()
    if output_directory == run_directory or run_directory in output_directory.parents:
        raise ValueError("Experiment output must be outside the accepted run.")
    if output_directory.exists():
        raise FileExistsError(f"Experiment output already exists: {output_directory}")
    assembly = manifest.get("assembly", {})
    if manifest["status"] != "generated" or assembly.get("status") != "assembled":
        raise ValueError("Experiment requires a completed, assembled unit run.")
    if assembly.get("extra_silence_ms_between_units") != 0:
        raise ValueError("Expected the accepted zero-added-silence assembly.")
    final_path = (run_directory / assembly["output_path"]).resolve()
    final_path.relative_to(run_directory)
    if file_sha256(final_path) != assembly["wav_sha256"]:
        raise ValueError("Accepted assembled chapter hash differs.")

    units = [unit for scene in manifest["scenes"]
             for unit in scene["synthesis_units"]]
    clips = assembly["units"]
    rate = assembly["audio"]["sample_rate_hz"]
    if rate != 24000 or len(units) != len(clips):
        raise ValueError("Expected one 24 kHz assembly record per unit.")
    if not after_units or len(set(after_units)) != len(after_units):
        raise ValueError("Select distinct unit boundary numbers.")
    if any(not 1 <= value < len(units) for value in after_units):
        raise ValueError(f"Boundaries must be after units 1 through {len(units)-1}.")
    if any(clip["unit_id"] != unit["id"] or
           clip["selected_attempt_id"] != unit["generation"]["selected_attempt_id"]
           for clip, unit in zip(clips, units)):
        raise ValueError("Assembly unit order or selection differs from the frozen plan.")

    selected = {}
    for number in sorted(set(after_units) | {value + 1 for value in after_units}):
        unit = units[number - 1]
        attempt_id = unit["generation"]["selected_attempt_id"]
        attempt = next((item for item in unit["generation"]["attempts"]
                        if item["id"] == attempt_id), None)
        if attempt is None or attempt.get("content_qc", {}).get("status") != "passed":
            raise ValueError(f"{unit['id']} has no selected QC-passed attempt.")
        clip = clips[number - 1]
        if (clip["artifact_path"] != attempt["output_path"] or
                clip["artifact_wav_sha256"] != attempt["wav_sha256"]):
            raise ValueError(f"{unit['id']} assembly artifact differs from selection.")
        _read_qc_evidence(run_directory, manifest, unit, attempt)
        selected[number] = (attempt, _read_selected_wav(
            run_directory, unit, attempt, rate))

    output_directory.mkdir(parents=True)
    records = []
    for after in after_units:
        left, right = units[after - 1], units[after]
        left_attempt, left_pcm = selected[after]
        right_attempt, right_pcm = selected[after + 1]
        left_quiet = edge_quiet_frames(left_pcm, rate, "trailing")
        right_quiet = edge_quiet_frames(right_pcm, rate, "leading")
        existing = left_quiet + right_quiet
        context_frames = CONTEXT_SECONDS * rate
        left_context = left_pcm[-context_frames * 2:]
        right_context = right_pcm[:context_frames * 2]
        folder = output_directory / f"after_unit_{after:04d}"
        folder.mkdir()
        choices = ("original",) + MINIMUM_PAUSE_MS
        # Opaque deterministic label order permits listening before reading the key.
        shuffled = sorted(choices, key=lambda choice: hashlib.sha256(
            f"{assembly['wav_sha256']}:{after}:{choice}".encode("ascii")
        ).digest())
        variants = []
        for label, choice in zip("ABCDE", shuffled):
            minimum = 0 if choice == "original" else choice
            inserted = (0 if choice == "original" else
                        inserted_frames(minimum, existing, rate))
            pcm = left_context + b"\x00" * (inserted * 2) + right_context
            relative = f"after_unit_{after:04d}/{label}.wav"
            output = output_directory / relative
            _write_clip(output, rate, pcm)
            variants.append({
                "blind_label": label,
                "variant": choice,
                "requested_minimum_ms": minimum,
                "inserted_frames": inserted,
                "actual_inserted_ms": inserted * 1000 / rate,
                "output_filename": relative,
                "output_sha256": file_sha256(output),
            })
        records.append({
            "after_unit_number": after,
            "left_unit_id": left["id"],
            "right_unit_id": right["id"],
            "left_selected_attempt_id": left_attempt["id"],
            "right_selected_attempt_id": right_attempt["id"],
            "left_wav_sha256": left_attempt["wav_sha256"],
            "right_wav_sha256": right_attempt["wav_sha256"],
            "chapter_boundary_seconds": clips[after - 1]["end_frame_exclusive"] / rate,
            "source_context": _source_context(left, right),
            "measured_left_trailing_silence_ms": left_quiet * 1000 / rate,
            "measured_right_leading_silence_ms": right_quiet * 1000 / rate,
            "measured_existing_total_seam_silence_ms": existing * 1000 / rate,
            "variants": variants,
        })

    evidence = {
        "schema_version": 1,
        "purpose": "listening_only_minimum_unit_seam_pause_ab",
        "source_run": str(run_directory),
        "source_chapter_wav_sha256": assembly["wav_sha256"],
        "ordered_unit_plan_sha256": manifest["synthesis_unit_plan"][
            "ordered_unit_plan_sha256"],
        "sample_rate_hz": rate,
        "channel_count": 1,
        "pcm_sample_width_bytes": 2,
        "context_seconds_each_side": CONTEXT_SECONDS,
        "quiet_detection": {
            "window_ms": WINDOW_MS,
            "rms_threshold_dbfs": RMS_THRESHOLD_DBFS,
            "peak_threshold_dbfs": PEAK_THRESHOLD_DBFS,
            "requires_contiguous_quiet_windows_from_each_wav_edge": True,
            "partial_window_behavior": "not_counted",
        },
        "minimum_pause_ms": list(MINIMUM_PAUSE_MS),
        "boundaries": records,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--after", nargs="+", type=int, required=True,
                        help="Unit numbers whose following seam should be compared.")
    args = parser.parse_args(argv)
    result = build_experiment(
        args.run_directory, args.output_directory, args.after,
    )
    print(f"Created {len(result['boundaries'])} seams and "
          f"{sum(len(item['variants']) for item in result['boundaries'])} clips.")
    print(f"Listening clips: {args.output_directory.expanduser().resolve()}")
    print("Keep manifest.json closed until after blind listening.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
