"""Build blind pause probes from fixed, manually approved cleanup on copies.

Run as ``python -B -m evaluation.run_event_pause_ab RUN PLAN OUTPUT``.
This evaluation tool reads selected WAVs and never writes inside RUN.
"""

import argparse
import hashlib
import json
from pathlib import Path

from evaluation.run_combined_seam_cleanup_ab import _selected
from evaluation.run_seam_pause_ab import _source_context, _write_clip, edge_quiet_frames
from src.audiobook.cosyvoice import file_sha256
from src.audiobook.unit_execution import _load


def _frames(milliseconds, rate):
    return round(milliseconds * rate / 1000)


def _millis(frames, rate):
    return frames * 1000 / rate


def build_experiment(run_directory, plan_path, output_directory):
    run, _, manifest = _load(run_directory)
    plan_path = Path(plan_path).expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    output = Path(output_directory).expanduser().resolve()
    if output.exists() or output == run or run in output.parents:
        raise ValueError("Output must be a new directory outside the accepted run.")
    assembly = manifest.get("assembly", {})
    if (manifest.get("schema_version") != 5 or manifest.get("status") != "generated"
            or assembly.get("status") != "assembled"
            or assembly.get("extra_silence_ms_between_units") != 0):
        raise ValueError("Expected completed schema-5 zero-silence assembly.")
    chapter = (run / assembly["output_path"]).resolve()
    chapter.relative_to(run)
    chapter_hash = file_sha256(chapter)
    if chapter_hash != assembly["wav_sha256"] or chapter_hash != plan["source_chapter_sha256"]:
        raise ValueError("Chapter hash differs from accepted assembly or experiment plan.")
    if plan.get("schema_version") != 1 or not plan.get("seams"):
        raise ValueError("Expected a nonempty schema-1 experiment plan.")
    rate = assembly["audio"]["sample_rate_hz"]
    if rate != 24000:
        raise ValueError("Expected 24 kHz selected WAVs.")
    units = [unit for scene in manifest["scenes"] for unit in scene["synthesis_units"]]
    assembled = assembly["units"]
    specs = plan["seams"]
    if (len({spec["after_unit_number"] for spec in specs}) != len(specs)
            or any(not 1 <= spec["after_unit_number"] < len(units) for spec in specs)):
        raise ValueError("Experiment plan has an invalid or duplicate seam.")
    context_frames = plan["context_seconds_each_side"] * rate
    if not isinstance(context_frames, int) or context_frames <= 0:
        raise ValueError("Listening context must be a positive whole frame count.")

    selected = {}
    for number in sorted({index for spec in specs for index in
                          (spec["after_unit_number"], spec["after_unit_number"] + 1)}):
        selected[number] = _selected(run, manifest, units[number - 1],
                                     assembled[number - 1], rate)
    output.mkdir(parents=True)
    records = []
    for spec in specs:
        after = spec["after_unit_number"]
        left, right = units[after - 1], units[after]
        left_attempt, left_pcm = selected[after]
        right_attempt, right_pcm = selected[after + 1]
        source_context = _source_context(left, right)
        if source_context["structure"] != spec["expected_structure"]:
            raise ValueError(f"Source boundary structure changed after unit {after}.")
        left_cut = _frames(spec["left_trim_ms"], rate)
        right_cut = _frames(spec["right_trim_ms"], rate)
        if (left_cut < 0 or right_cut < 0 or left_cut * 2 >= len(left_pcm)
                or right_cut * 2 >= len(right_pcm)):
            raise ValueError(f"Invalid fixed cleanup after unit {after}.")
        clean_left = left_pcm[:len(left_pcm) - left_cut * 2] if left_cut else left_pcm
        clean_right = right_pcm[right_cut * 2:] if right_cut else right_pcm
        left_quiet = edge_quiet_frames(clean_left, rate, "trailing")
        right_quiet = edge_quiet_frames(clean_right, rate, "leading")
        breath = spec["retained_natural_breath"]
        if breath:
            marks = {key: _frames(value, rate) for key, value in
                     spec["event_marks_ms_in_original_wavs"].items()}
            if not (set(marks) == {"left_speech_end", "right_natural_breath_start",
                                   "right_natural_breath_end", "right_speech_onset"}
                    and 0 <= marks["left_speech_end"] <= len(clean_left) // 2
                    and right_cut < marks["right_natural_breath_start"]
                    < marks["right_natural_breath_end"]
                    < marks["right_speech_onset"] <= len(right_pcm) // 2):
                raise ValueError(f"Fixed cleanup threatens breath or speech after unit {after}.")
            q_base = ((len(clean_left) // 2 - marks["left_speech_end"])
                      + (marks["right_natural_breath_start"] - right_cut))
            breath_frames = (marks["right_natural_breath_end"]
                             - marks["right_natural_breath_start"])
            qpost = marks["right_speech_onset"] - marks["right_natural_breath_end"]
            if (_millis(q_base, rate) != spec["expected_unchanged_Qpre_ms"]
                    or _millis(breath_frames, rate) != spec["expected_B_ms"]
                    or _millis(qpost, rate) != spec["expected_Qpost_ms"]):
                raise ValueError(f"Manual event marks changed after unit {after}.")
        else:
            q_base = left_quiet + right_quiet
            breath_frames = qpost = None
            if _millis(q_base, rate) != spec["expected_unchanged_Q_ms"]:
                raise ValueError(f"Measured cleaned quiet changed after unit {after}.")

        targets = spec["targets_ms"]
        if len(targets) != 4 or targets[0] is not None or any(
                not isinstance(value, int) or value < 0 for value in targets[1:]):
            raise ValueError("Expected unchanged plus three nonnegative timing targets.")
        cases = list(enumerate(targets))
        blinded = sorted(cases, key=lambda case: hashlib.sha256(
            f"{chapter_hash}:{after}:{case[0]}".encode("ascii")
        ).digest())
        folder = output / f"after_unit_{after:04d}"
        folder.mkdir()
        variants = []
        prior_hashes = {}
        for label, (probe_index, target_ms) in zip("ABCD", blinded):
            target_frames = _frames(target_ms, rate) if target_ms is not None else None
            added = (0 if target_frames is None else max(0, target_frames - q_base))
            left_context = clean_left[-context_frames * 2:]
            right_context = clean_right[:context_frames * 2]
            payload = left_context + b"\x00\x00" * added + right_context
            path = folder / f"{label}.wav"
            _write_clip(path, rate, payload)
            if breath:
                # The complete retained inhale, following quiet, and speech
                # onset must be exact original unit-70 PCM for every variant.
                protected_start = marks["right_natural_breath_start"] - right_cut
                protected_end = marks["right_speech_onset"] - right_cut
                output_start = len(left_context) + added * 2 + protected_start * 2
                protected = right_pcm[
                    marks["right_natural_breath_start"] * 2:
                    marks["right_speech_onset"] * 2]
                if payload[output_start:output_start + len(protected)] != protected:
                    raise ValueError("Natural breath or lead-in differs from selected WAV.")
            digest = file_sha256(path)
            variants.append({
                "blind_label": label,
                "probe_index": probe_index,
                "requested_target_ms": target_ms,
                "target_applies_to": "Qpre" if breath else "Q",
                "Q_quiet_ms": None if breath else _millis(q_base + added, rate),
                "Qpre_quiet_before_breath_ms": _millis(q_base + added, rate) if breath else None,
                "B_retained_breath_ms": _millis(breath_frames, rate) if breath else None,
                "Qpost_quiet_after_breath_ms": _millis(qpost, rate) if breath else None,
                "D_speech_to_speech_ms": _millis(
                    q_base + added + (breath_frames or 0) + (qpost or 0), rate),
                "actual_inserted_frames": added,
                "actual_inserted_ms": _millis(added, rate),
                "insertion_location": (
                    "cleaned_unit_join_before_retained_breath" if breath else
                    "between_cleaned_selected_unit_wavs"),
                "output_filename": path.relative_to(output).as_posix(),
                "output_sha256": digest,
                "exact_audio_duplicate_of_blind_label": prior_hashes.get(digest),
            })
            prior_hashes.setdefault(digest, label)
        records.append({
            "seam_id": f"{left['id']}__to__{right['id']}",
            "after_unit_number": after,
            "structural_boundary_type": source_context["structure"],
            "source_context": source_context,
            "acoustic_screening": spec.get("acoustic_screening"),
            "cleanup_applied": {
                "left_tail_trim_frames": left_cut,
                "left_tail_trim_ms": _millis(left_cut, rate),
                "right_head_trim_frames": right_cut,
                "right_head_trim_ms": _millis(right_cut, rate),
            },
            "retained_natural_breath": breath,
            "unchanged_Q_or_Qpre_ms": _millis(q_base, rate),
            "unchanged_B_ms": _millis(breath_frames, rate) if breath else None,
            "unchanged_Qpost_ms": _millis(qpost, rate) if breath else None,
            "measured_clean_left_trailing_silence_ms": _millis(left_quiet, rate),
            "measured_clean_right_leading_silence_ms": _millis(right_quiet, rate),
            "left_selected_wav_sha256": left_attempt["wav_sha256"],
            "right_selected_wav_sha256": right_attempt["wav_sha256"],
            "variants": variants,
        })

    for number, (attempt, _) in selected.items():
        path = (run / attempt["output_path"]).resolve()
        if file_sha256(path) != attempt["wav_sha256"]:
            raise ValueError(f"Selected WAV changed during experiment: {number}.")
    if file_sha256(chapter) != chapter_hash:
        raise ValueError("Accepted chapter changed during experiment.")
    evidence = {
        "schema_version": 1,
        "purpose": "blind_fixed_cleanup_event_aware_pause_timing_experiment",
        "source_run": str(run),
        "source_chapter_sha256": chapter_hash,
        "experiment_plan_path": str(plan_path),
        "experiment_plan_sha256": file_sha256(plan_path),
        "sample_rate_hz": rate,
        "context_seconds_each_side": plan["context_seconds_each_side"],
        "silence_measurement": (
            "Consecutive edge 10 ms windows with RMS <= -55 dBFS and peak <= -45 dBFS. "
            "Natural breath duration uses manually inspected event marks, never silence counts."),
        "all_pause_targets_are_experimental": True,
        "seams": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("plan_path", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args(argv)
    evidence = build_experiment(
        args.run_directory, args.plan_path, args.output_directory)
    print(f"Created {len(evidence['seams'])} seam folders and "
          f"{sum(len(seam['variants']) for seam in evidence['seams'])} blind clips.")
    print(f"Output: {args.output_directory.expanduser().resolve()}")
    print("Listen before opening manifest.json to preserve blind labels.")


if __name__ == "__main__":
    main()
