"""Build blind, copy-only edge-cut and pause comparisons for selected unit seams.

Example: python -B -m evaluation.run_edge_cleanup_ab RUN OUTPUT --minimum-ms 200 \
    --seam 5:left:190:230 --seam 58:left:160:180 --seam 69:right:400:440

The pause value is an experiment parameter, never a production default.
"""

import argparse
import hashlib
import json
from pathlib import Path

from evaluation.run_seam_pause_ab import (
    CONTEXT_SECONDS, _read_selected_wav, _write_clip, edge_quiet_frames,
    inserted_frames,
)
from src.audiobook.cosyvoice import file_sha256
from src.audiobook.unit_execution import _load, _read_qc_evidence


def parse_seam(value):
    try:
        number_text, side, conservative_text, stronger_text = value.split(":")
        number = int(number_text)
        conservative = int(conservative_text)
        stronger = int(stronger_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "Expected UNIT:left|right:CONSERVATIVE_MS:STRONGER_MS"
        ) from error
    if number < 1 or side not in {"left", "right"} or not 0 < conservative < stronger:
        raise argparse.ArgumentTypeError("Use a valid unit, side, and increasing positive cuts.")
    return number, side, conservative, stronger


def render_variant(left_pcm, right_pcm, rate, side, cut_ms,
                   minimum_ms, context_seconds=CONTEXT_SECONDS):
    """Slice source PCM copies, remeasure both edges, and optionally fill pause."""
    if side not in {"left", "right"} or cut_ms < 0 or rate <= 0:
        raise ValueError("Invalid side, cut, or sample rate.")
    if minimum_ms is not None and minimum_ms < 0:
        raise ValueError("Minimum pause must be nonnegative.")
    cut_frames = round(rate * cut_ms / 1000)
    cut_bytes = cut_frames * 2
    if cut_bytes >= len(left_pcm if side == "left" else right_pcm):
        raise ValueError("Edge cut would remove a complete selected WAV.")
    clean_left = left_pcm[:-cut_bytes] if side == "left" and cut_bytes else left_pcm
    clean_right = right_pcm[cut_bytes:] if side == "right" and cut_bytes else right_pcm
    left_quiet = edge_quiet_frames(clean_left, rate, "trailing")
    right_quiet = edge_quiet_frames(clean_right, rate, "leading")
    remaining_quiet = left_quiet + right_quiet
    added = (0 if minimum_ms is None else
             inserted_frames(minimum_ms, remaining_quiet, rate))
    context_bytes = context_seconds * rate * 2
    payload = (clean_left[-context_bytes:] + b"\x00\x00" * added +
               clean_right[:context_bytes])
    return payload, {
        "left_cut_frames": cut_frames if side == "left" else 0,
        "right_cut_frames": cut_frames if side == "right" else 0,
        "left_cut_ms": cut_frames * 1000 / rate if side == "left" else 0,
        "right_cut_ms": cut_frames * 1000 / rate if side == "right" else 0,
        "resulting_left_trailing_silence_ms": left_quiet * 1000 / rate,
        "resulting_right_leading_silence_ms": right_quiet * 1000 / rate,
        "resulting_total_seam_silence_ms": remaining_quiet * 1000 / rate,
        "requested_minimum_pause_ms": minimum_ms,
        "inserted_silence_frames": added,
        "inserted_silence_ms": added * 1000 / rate,
    }


def build_experiment(run_directory, output_directory, seams, minimum_ms):
    run_directory, _, manifest = _load(run_directory)
    output_directory = Path(output_directory).expanduser().resolve()
    if (output_directory.exists() or output_directory == run_directory or
            run_directory in output_directory.parents):
        raise ValueError("Use a new output directory outside the accepted run.")
    assembly = manifest.get("assembly", {})
    if (manifest.get("schema_version") != 5 or manifest.get("status") != "generated"
            or assembly.get("status") != "assembled"
            or assembly.get("extra_silence_ms_between_units") != 0):
        raise ValueError("Expected completed schema-5 assembly with no injected silence.")
    chapter_path = (run_directory / assembly["output_path"]).resolve()
    chapter_path.relative_to(run_directory)
    original_chapter_hash = file_sha256(chapter_path)
    if original_chapter_hash != assembly["wav_sha256"]:
        raise ValueError("Accepted chapter hash differs from assembly evidence.")
    units = [unit for scene in manifest["scenes"] for unit in scene["synthesis_units"]]
    clips = assembly["units"]
    rate = assembly["audio"]["sample_rate_hz"]
    if rate != 24000 or len(units) != len(clips) or not seams:
        raise ValueError("Expected a 24 kHz unit assembly and at least one seam.")
    if (len({item[0] for item in seams}) != len(seams) or
            any(not 1 <= item[0] < len(units) for item in seams)):
        raise ValueError("Use distinct valid unit seams.")
    if minimum_ms < 0:
        raise ValueError("Minimum pause must be nonnegative.")

    selected = {}
    for number in sorted({index for after, *_ in seams for index in (after, after + 1)}):
        unit = units[number - 1]
        clip = clips[number - 1]
        attempt_id = unit["generation"]["selected_attempt_id"]
        attempt = next((item for item in unit["generation"]["attempts"]
                        if item["id"] == attempt_id), None)
        if (attempt is None or attempt.get("content_qc", {}).get("status") != "passed"
                or clip["unit_id"] != unit["id"]
                or clip["selected_attempt_id"] != attempt_id
                or clip["artifact_path"] != attempt["output_path"]
                or clip["artifact_wav_sha256"] != attempt["wav_sha256"]):
            raise ValueError(f"Selected assembly/QC mismatch at unit {number}.")
        _read_qc_evidence(run_directory, manifest, unit, attempt)
        selected[number] = (unit, attempt, _read_selected_wav(
            run_directory, unit, attempt, rate))

    output_directory.mkdir(parents=True)
    records = []
    for after, side, conservative_ms, stronger_ms in seams:
        left_unit, left_attempt, left_pcm = selected[after]
        right_unit, right_attempt, right_pcm = selected[after + 1]
        cases = (
            ("original", 0, None),
            ("conservative_cut", conservative_ms, None),
            ("stronger_cut", stronger_ms, None),
            ("original_with_pause", 0, minimum_ms),
            ("conservative_cut_with_pause", conservative_ms, minimum_ms),
        )
        # Labels are stable but opaque until the listener opens the manifest.
        blinded = sorted(cases, key=lambda case: hashlib.sha256(
            f"{original_chapter_hash}:{after}:{case[0]}".encode("ascii")
        ).digest())
        folder = output_directory / f"after_unit_{after:04d}"
        folder.mkdir()
        variants = []
        for label, (name, cut_ms, pause_ms) in zip("ABCDE", blinded):
            payload, details = render_variant(
                left_pcm, right_pcm, rate, side, cut_ms, pause_ms)
            path = folder / f"{label}.wav"
            _write_clip(path, rate, payload)
            variants.append({
                "blind_label": label,
                "treatment": name,
                **details,
                "output_filename": path.relative_to(output_directory).as_posix(),
                "output_sha256": file_sha256(path),
            })
        records.append({
            "after_unit_number": after,
            "left_unit_id": left_unit["id"],
            "right_unit_id": right_unit["id"],
            "left_selected_attempt_id": left_attempt["id"],
            "right_selected_attempt_id": right_attempt["id"],
            "left_source_wav_sha256": left_attempt["wav_sha256"],
            "right_source_wav_sha256": right_attempt["wav_sha256"],
            "experimental_cut_side": side,
            "variants": variants,
        })

    for number, (_, attempt, _) in selected.items():
        path = (run_directory / attempt["output_path"]).resolve()
        if file_sha256(path) != attempt["wav_sha256"]:
            raise ValueError(f"Selected source WAV changed during experiment: {number}.")
    if file_sha256(chapter_path) != original_chapter_hash:
        raise ValueError("Accepted chapter changed during experiment.")
    evidence = {
        "schema_version": 1,
        "purpose": "blind_copy_only_edge_cleanup_and_pause_listening_experiment",
        "source_run": str(run_directory),
        "source_chapter_sha256": original_chapter_hash,
        "reference_minimum_pause_ms": minimum_ms,
        "reference_pause_is_not_a_production_default": True,
        "context_seconds_each_side": CONTEXT_SECONDS,
        "sample_rate_hz": rate,
        "silence_measurement": "Contiguous edge 10 ms windows, RMS <= -55 dBFS and peak <= -45 dBFS; rerun after each cut.",
        "boundaries": records,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--minimum-ms", type=int, required=True,
                        help="Experimental reference only; not a production default.")
    parser.add_argument("--seam", type=parse_seam, action="append", required=True,
                        help="UNIT:left|right:CONSERVATIVE_MS:STRONGER_MS")
    args = parser.parse_args(argv)
    result = build_experiment(
        args.run_directory, args.output_directory, args.seam, args.minimum_ms)
    print(f"Created {len(result['boundaries'])} seam folders and "
          f"{sum(len(item['variants']) for item in result['boundaries'])} blind clips.")
    print(f"Output: {args.output_directory.expanduser().resolve()}")
    print("Listen before opening manifest.json to preserve the blind labels.")


if __name__ == "__main__":
    main()
