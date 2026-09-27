"""Create a blind copy-only comparison of one manually annotated unit seam.

The event times are explicit experiment inputs. This tool does not detect
artifacts or change the accepted run, selected WAVs, or production assembly.
"""

import argparse
import hashlib
import json
from pathlib import Path

from evaluation.run_seam_pause_ab import (
    CONTEXT_SECONDS, _read_selected_wav, _write_clip, edge_quiet_frames,
)
from src.audiobook.cosyvoice import file_sha256
from src.audiobook.unit_execution import _load, _read_qc_evidence


def _frames(milliseconds, rate):
    return round(milliseconds * rate / 1000)


def _selected(run, manifest, unit, assembly_unit, rate):
    attempt_id = unit["generation"]["selected_attempt_id"]
    attempt = next((item for item in unit["generation"]["attempts"]
                    if item["id"] == attempt_id), None)
    if (attempt is None or attempt.get("content_qc", {}).get("status") != "passed"
            or assembly_unit["unit_id"] != unit["id"]
            or assembly_unit["selected_attempt_id"] != attempt_id
            or assembly_unit["artifact_path"] != attempt["output_path"]
            or assembly_unit["artifact_wav_sha256"] != attempt["wav_sha256"]):
        raise ValueError(f"Selection or assembly differs for {unit['id']}.")
    _read_qc_evidence(run, manifest, unit, attempt)
    pcm = _read_selected_wav(run, unit, attempt, rate)
    if len(pcm) // 2 != assembly_unit["frame_count"]:
        raise ValueError(f"Assembly frame count differs for {unit['id']}.")
    return attempt, pcm


def build_experiment(run_directory, output_directory, after, left_cut_ms,
                     right_cut_ms, event_ms):
    run, _, manifest = _load(run_directory)
    output = Path(output_directory).expanduser().resolve()
    if output.exists() or output == run or run in output.parents:
        raise ValueError("Choose a new output directory outside the accepted run.")
    assembly = manifest.get("assembly", {})
    if (manifest.get("schema_version") != 5 or manifest.get("status") != "generated"
            or assembly.get("status") != "assembled"
            or assembly.get("extra_silence_ms_between_units") != 0):
        raise ValueError("Expected completed schema-5 zero-added-silence assembly.")
    units = [unit for scene in manifest["scenes"] for unit in scene["synthesis_units"]]
    assembled = assembly["units"]
    rate = assembly["audio"]["sample_rate_hz"]
    if rate != 24000 or not 1 <= after < len(units):
        raise ValueError("Expected a valid 24 kHz synthesis-unit boundary.")
    chapter = (run / assembly["output_path"]).resolve()
    chapter.relative_to(run)
    chapter_hash = file_sha256(chapter)
    if chapter_hash != assembly["wav_sha256"]:
        raise ValueError("Accepted chapter hash differs from the manifest.")
    left_attempt, left_pcm = _selected(run, manifest, units[after - 1],
                                       assembled[after - 1], rate)
    right_attempt, right_pcm = _selected(run, manifest, units[after],
                                         assembled[after], rate)
    left_frames, right_frames = len(left_pcm) // 2, len(right_pcm) // 2
    left_cut, right_cut = _frames(left_cut_ms, rate), _frames(right_cut_ms, rate)
    marks = {name: _frames(value, rate) for name, value in event_ms.items()}
    required = {"left_speech_end", "left_partial_breath_start",
                "right_artifact_start", "right_artifact_end",
                "right_natural_breath_start", "right_natural_breath_end",
                "right_speech_onset"}
    if set(marks) != required:
        raise ValueError("Specify all seven manually inspected event boundaries.")
    if not (0 <= marks["left_speech_end"] < marks["left_partial_breath_start"]
            < left_frames and 0 < left_cut < left_frames - marks["left_speech_end"]):
        raise ValueError("Left event marks or proposed cut are incompatible.")
    if not (0 <= marks["right_artifact_start"] < marks["right_artifact_end"]
            <= right_cut < marks["right_natural_breath_start"]
            < marks["right_natural_breath_end"] < marks["right_speech_onset"]
            < right_frames):
        raise ValueError("Right cut must end after the artifact and before the natural breath.")
    if left_frames - left_cut < marks["left_speech_end"]:
        raise ValueError("Left cut would remove intended speech.")

    cases = (("original", 0, 0),
             ("right_head_only", 0, right_cut),
             ("combined", left_cut, right_cut))
    blinded = sorted(cases, key=lambda case: hashlib.sha256(
        f"{chapter_hash}:{after}:{case[0]}".encode("ascii")
    ).digest())
    output.mkdir(parents=True)
    variants = []
    for label, (treatment, lcut, rcut) in zip("ABC", blinded):
        clean_left = left_pcm[:(left_frames - lcut) * 2]
        clean_right = right_pcm[rcut * 2:]
        left_quiet = edge_quiet_frames(clean_left, rate, "trailing")
        right_quiet = edge_quiet_frames(clean_right, rate, "leading")
        context_bytes = CONTEXT_SECONDS * rate * 2
        payload = clean_left[-context_bytes:] + clean_right[:context_bytes]
        path = output / f"{label}.wav"
        _write_clip(path, rate, payload)

        # Qpre is defined only when the pre-breath region is actually quiet.
        # For contaminated controls, record the elapsed interval and unwanted
        # events separately rather than mislabeling them as silence.
        unwanted = []
        if left_frames - lcut > marks["left_partial_breath_start"]:
            unwanted.append("left_partial_inhale")
        if rcut < marks["right_artifact_end"]:
            unwanted.append("right_head_vocal_artifact")
        pre_breath = ((left_frames - lcut - marks["left_speech_end"])
                      + marks["right_natural_breath_start"] - rcut)
        breath = (marks["right_natural_breath_end"]
                  - marks["right_natural_breath_start"])
        post_breath = (marks["right_speech_onset"]
                       - marks["right_natural_breath_end"])
        transition = pre_breath + breath + post_breath
        qpre = pre_breath if not unwanted else None
        if qpre is not None and qpre + breath + post_breath != transition:
            raise ValueError("Clean event timeline is inconsistent.")
        variants.append({
            "blind_label": label,
            "treatment": treatment,
            "left_trim_frames": lcut,
            "left_trim_ms": lcut * 1000 / rate,
            "right_trim_frames": rcut,
            "right_trim_ms": rcut * 1000 / rate,
            "end_of_intended_left_speech_ms_in_original_left_wav":
                marks["left_speech_end"] * 1000 / rate,
            "end_of_intended_left_speech_ms_in_output_clip":
                ((len(clean_left[-context_bytes:]) // 2)
                 - (left_frames - lcut - marks["left_speech_end"])) * 1000 / rate,
            "Qpre_quiet_before_retained_breath_ms":
                qpre * 1000 / rate if qpre is not None else None,
            "pre_breath_elapsed_ms_including_unwanted_events":
                pre_breath * 1000 / rate if unwanted else None,
            "B_retained_natural_breath_ms": breath * 1000 / rate,
            "Qpost_quiet_after_breath_before_speech_ms": post_breath * 1000 / rate,
            "D_speech_to_speech_transition_ms": transition * 1000 / rate,
            "D_equals_Qpre_plus_B_plus_Qpost": not unwanted,
            "unwanted_events_retained": unwanted,
            "measured_left_edge_trailing_quiet_ms": left_quiet * 1000 / rate,
            "measured_right_edge_leading_quiet_ms": right_quiet * 1000 / rate,
            "inserted_silence_frames": 0,
            "output_filename": path.name,
            "output_sha256": file_sha256(path),
        })

    for attempt in (left_attempt, right_attempt):
        selected_path = (run / attempt["output_path"]).resolve()
        if file_sha256(selected_path) != attempt["wav_sha256"]:
            raise ValueError("Selected source WAV changed during experiment.")
    if file_sha256(chapter) != chapter_hash:
        raise ValueError("Accepted chapter changed during experiment.")
    evidence = {
        "schema_version": 1,
        "purpose": "blind_manual_bidirectional_seam_cleanup_without_pause",
        "source_run": str(run),
        "source_chapter_sha256": chapter_hash,
        "after_unit_number": after,
        "left_unit_id": units[after - 1]["id"],
        "right_unit_id": units[after]["id"],
        "left_selected_attempt_id": left_attempt["id"],
        "right_selected_attempt_id": right_attempt["id"],
        "left_source_wav_sha256": left_attempt["wav_sha256"],
        "right_source_wav_sha256": right_attempt["wav_sha256"],
        "sample_rate_hz": rate,
        "context_seconds_each_side": CONTEXT_SECONDS,
        "event_marks_ms_in_original_wavs": event_ms,
        "event_annotation_provenance": (
            "User listening identified both unwanted edge events and the later natural inhale; "
            "time bounds are approximate manual readings of 10 ms energy and spectrograms. "
            "The recorded ASR first expected-token spike is corroboration, not a phoneme boundary."),
        "event_model_note": (
            "Qpre is null when a retained partial inhale or head artifact interrupts the "
            "pre-breath interval. D remains the full speech-to-speech elapsed time; "
            "D = Qpre + B + Qpost applies to the clean combined variant only."),
        "variants": variants,
    }
    (output / "manifest.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--after", type=int, required=True)
    parser.add_argument("--left-cut-ms", type=int, required=True)
    parser.add_argument("--right-cut-ms", type=int, required=True)
    for name in ("left-speech-end", "left-partial-breath-start",
                 "right-artifact-start", "right-artifact-end",
                 "right-natural-breath-start", "right-natural-breath-end",
                 "right-speech-onset"):
        parser.add_argument(f"--{name}-ms", type=int, required=True)
    args = parser.parse_args(argv)
    event_ms = {
        name: getattr(args, name + "_ms")
        for name in ("left_speech_end", "left_partial_breath_start",
                     "right_artifact_start", "right_artifact_end",
                     "right_natural_breath_start", "right_natural_breath_end",
                     "right_speech_onset")
    }
    result = build_experiment(
        args.run_directory, args.output_directory, args.after,
        args.left_cut_ms, args.right_cut_ms, event_ms)
    print(f"Created {len(result['variants'])} blind variants in "
          f"{args.output_directory.expanduser().resolve()}")
    print("Listen before opening manifest.json to preserve the blind labels.")


if __name__ == "__main__":
    main()
