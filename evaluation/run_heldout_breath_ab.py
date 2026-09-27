"""Copy-only held-out paragraph breath A/B with one sentence reference.

Manual seam selection and timings are experiment-specific. Nothing here is
imported by production generation, QC, retry, or assembly.
"""

import argparse
import hashlib
import json
from pathlib import Path

from evaluation.run_combined_seam_cleanup_ab import _selected
from evaluation.run_seam_pause_ab import _source_context, _write_clip, edge_quiet_frames
from src.audiobook.cosyvoice import file_sha256
from src.audiobook.unit_execution import _load


RATE = 24000
CONTEXT_FRAMES = 4 * RATE
DONOR_START, DONOR_END = 11520, 17280
DONOR_FRAMES = DONOR_END - DONOR_START
CHAPTER_HASH = "a3dce3fbb9d6cdf333af32dcc260b9b9e02ba3458f865f137b2f747437ebba1b"
DONOR_WAV_HASH = "8c8d3b46f7de1a475f36345a980292acc710437f6db23bdc0bd3f538f8f0a1eb"
DONOR_PCM_HASH = "2214bf52feb20e9ff0df1c4de6ed537defc2b09c8b50c721199426b3542f2540"

# after-unit ID, expected left/right quiet ms, source transition, editing rule
PAIRS = (
    (4, 160, 120, "narration_to_dialogue_introducing_paragraph", "insert"),
    (17, 170, 350, "narration_to_narration_paragraph_long_pause", "replace_right_quiet"),
    (24, 190, 130, "narration_to_narration_paragraph", "insert"),
    (45, 120, 80, "narration_to_narration_paragraph", "insert"),
    (48, 60, 90, "dialogue_to_narration_paragraph", "insert"),
)
REFERENCE = (33, 180, 40, "dialogue_sentence_to_narration_inline")


def _samples(pcm, frame):
    return int.from_bytes(pcm[frame * 2:frame * 2 + 2], "little", signed=True)


def _splice_steps(pcm, boundaries):
    checks = []
    for name, frame in boundaries:
        if not 0 < frame < len(pcm) // 2:
            raise ValueError("Splice boundary outside clip.")
        step = abs(_samples(pcm, frame - 1) - _samples(pcm, frame))
        checks.append({"name": name, "output_frame": frame, "pcm16_step": step})
        if step > 200:
            raise ValueError(f"Potential click at {name}: {step} PCM16 counts.")
    return checks


def render_pair(left, right, donor, rule, breath):
    """Return raw PCM, protected-context spans, and edit boundaries."""
    if len(donor) != DONOR_FRAMES * 2:
        raise ValueError("Approved donor must be exactly 5760 PCM16 frames.")
    lctx, rctx = left[-CONTEXT_FRAMES * 2:], right[:CONTEXT_FRAMES * 2]
    join = len(lctx) // 2
    if rule == "insert":
        middle = donor if breath else b"\x00\x00" * DONOR_FRAMES
        pcm = lctx + middle + rctx
        boundaries = (("left_to_block", join), ("block_to_right", join + DONOR_FRAMES))
        preserved = ((0, len(lctx), lctx),
                     (len(lctx) + len(middle), len(pcm), rctx))
        location = {"kind": "insert_at_selected_unit_join", "output_start_frame": join,
                    "frame_count": DONOR_FRAMES}
    elif rule == "replace_right_quiet":
        if edge_quiet_frames(right, RATE, "leading") < DONOR_FRAMES:
            raise ValueError("Right unit lacks 240 ms of measured leading quiet.")
        middle = donor if breath else right[:DONOR_FRAMES * 2]
        pcm = lctx + middle + rctx[DONOR_FRAMES * 2:]
        boundaries = (("unit_join", join), ("block_to_preserved_right", join + DONOR_FRAMES))
        preserved = ((0, len(lctx), lctx),
                     (len(lctx) + len(middle), len(pcm), rctx[DONOR_FRAMES * 2:]))
        location = {"kind": ("replace_first_240ms_of_measured_right_leading_quiet"
                             if breath else "preserve_original_right_leading_quiet"),
                    "source_right_start_frame": 0, "output_start_frame": join,
                    "frame_count": DONOR_FRAMES}
    else:
        raise ValueError(f"Unsupported held-out rule: {rule}.")
    for start, end, source in preserved:
        if pcm[start:end] != source:
            raise ValueError("Protected source PCM was changed.")
    if breath and pcm[join * 2:(join + DONOR_FRAMES) * 2] != donor:
        raise ValueError("Donor PCM changed during rendering.")
    return pcm, location, _splice_steps(pcm, boundaries)


def build_experiment(run_directory, output_directory):
    run, _, manifest = _load(run_directory)
    output = Path(output_directory).expanduser().resolve()
    if output.exists() or output == run or run in output.parents:
        raise ValueError("Output must be new and outside the accepted run.")
    assembly = manifest.get("assembly", {})
    if (manifest.get("schema_version") != 5 or manifest.get("status") != "generated"
            or assembly.get("status") != "assembled"
            or assembly.get("extra_silence_ms_between_units") != 0
            or assembly["audio"]["sample_rate_hz"] != RATE):
        raise ValueError("Expected completed schema-5 24-kHz zero-silence run.")
    chapter = (run / assembly["output_path"]).resolve()
    chapter.relative_to(run)
    if assembly["wav_sha256"] != CHAPTER_HASH or file_sha256(chapter) != CHAPTER_HASH:
        raise ValueError("Accepted chapter hash differs.")
    units = [unit for scene in manifest["scenes"] for unit in scene["synthesis_units"]]
    assembled = assembly["units"]
    needed = {70, REFERENCE[0], REFERENCE[0] + 1}
    for after, *_ in PAIRS:
        needed.update((after, after + 1))
    selected = {number: _selected(run, manifest, units[number - 1],
                                   assembled[number - 1], RATE)
                for number in sorted(needed)}
    donor_attempt, donor_source = selected[70]
    donor = donor_source[DONOR_START * 2:DONOR_END * 2]
    if (donor_attempt["wav_sha256"] != DONOR_WAV_HASH
            or hashlib.sha256(donor).hexdigest() != DONOR_PCM_HASH):
        raise ValueError("Approved donor source or raw PCM hash differs.")

    output.mkdir(parents=True)
    records = []
    for after, expected_left, expected_right, transition, rule in PAIRS:
        left_attempt, left = selected[after]
        right_attempt, right = selected[after + 1]
        source = _source_context(units[after - 1], units[after])
        if source["structure"] != "paragraph_break":
            raise ValueError(f"Held-out seam {after} lost its paragraph break.")
        left_quiet = edge_quiet_frames(left, RATE, "trailing")
        right_quiet = edge_quiet_frames(right, RATE, "leading")
        if (left_quiet, right_quiet) != (expected_left * RATE // 1000,
                                         expected_right * RATE // 1000):
            raise ValueError(f"Held-out seam {after} quiet measurement changed.")
        cases = sorted(("quiet", "donor"), key=lambda treatment: hashlib.sha256(
            f"{CHAPTER_HASH}:heldout:{after}:{treatment}".encode("ascii")
        ).digest())
        folder = output / f"after_unit_{after:04d}"
        folder.mkdir()
        variants = []
        rendered = {}
        for label, treatment in zip("AB", cases):
            breath = treatment == "donor"
            pcm, location, checks = render_pair(left, right, donor, rule, breath)
            rendered[treatment] = pcm
            path = folder / f"{label}.wav"
            _write_clip(path, RATE, pcm)
            d = expected_left + expected_right + (240 if rule == "insert" else 0)
            variants.append({
                "blind_label": label,
                "treatment": treatment,
                "original_left_trailing_quiet_ms": expected_left,
                "original_right_leading_quiet_ms": expected_right,
                "Qpre_quiet_before_donor_block_ms": expected_left,
                "donor_block_duration_ms": 240 if breath else 0,
                "central_quiet_block_duration_ms": 0 if breath else 240,
                "central_quiet_block_is_exact_zero_pcm": (
                    not breath and (rule == "insert" or
                                    right[:DONOR_FRAMES * 2] == b"\x00\x00" * DONOR_FRAMES)),
                "audible_breath_duration_ms_estimate": 220 if breath else 0,
                "Qpost_quiet_after_donor_block_ms": (expected_right if rule == "insert"
                                                       else expected_right - 240),
                "D_speech_to_speech_ms": d,
                "inserted_zero_frames": DONOR_FRAMES if rule == "insert" and not breath else 0,
                "inserted_donor_frames": DONOR_FRAMES if breath and rule == "insert" else 0,
                "replaced_right_quiet_frames": DONOR_FRAMES if breath and rule == "replace_right_quiet" else 0,
                "location": location,
                "donor_raw_pcm_sha256": DONOR_PCM_HASH if breath else None,
                "splice_checks": checks,
                "output_filename": path.relative_to(output).as_posix(),
                "output_sha256": file_sha256(path),
            })
        quiet_pcm, breath_pcm = rendered["quiet"], rendered["donor"]
        join = min(len(left) // 2, CONTEXT_FRAMES)
        start, end = join * 2, (join + DONOR_FRAMES) * 2
        if (len(quiet_pcm) != len(breath_pcm)
                or quiet_pcm[:start] != breath_pcm[:start]
                or quiet_pcm[end:] != breath_pcm[end:]
                or breath_pcm[start:end] != donor):
            raise ValueError(f"Held-out pair {after} differs outside the 240-ms donor block.")
        if rule == "insert" and quiet_pcm[start:end] != b"\x00\x00" * DONOR_FRAMES:
            raise ValueError("Breathless insert must be exact zero PCM.")
        records.append({
            "after_unit_number": after,
            "source_transition_type": transition,
            "source_context": source,
            "original_left_trailing_quiet_ms": expected_left,
            "original_right_leading_quiet_ms": expected_right,
            "left_source_wav_sha256": left_attempt["wav_sha256"],
            "right_source_wav_sha256": right_attempt["wav_sha256"],
            "matched_D_ms": d,
            "variants": variants,
        })

    after, expected_left, expected_right, transition = REFERENCE
    left_attempt, left = selected[after]
    right_attempt, right = selected[after + 1]
    source = _source_context(units[after - 1], units[after])
    if (source["structure"] != "inline"
            or edge_quiet_frames(left, RATE, "trailing") != expected_left * RATE // 1000
            or edge_quiet_frames(right, RATE, "leading") != expected_right * RATE // 1000):
        raise ValueError("Unchanged sentence-reference structure or timing differs.")
    folder = output / f"after_unit_{after:04d}"
    folder.mkdir()
    lctx, rctx = left[-CONTEXT_FRAMES * 2:], right[:CONTEXT_FRAMES * 2]
    pcm = lctx + rctx
    reference_path = folder / "reference.wav"
    checks = _splice_steps(pcm, (("unchanged_unit_join", len(lctx) // 2),))
    _write_clip(reference_path, RATE, pcm)
    records.append({
        "after_unit_number": after,
        "source_transition_type": transition,
        "source_context": source,
        "original_left_trailing_quiet_ms": expected_left,
        "original_right_leading_quiet_ms": expected_right,
        "left_source_wav_sha256": left_attempt["wav_sha256"],
        "right_source_wav_sha256": right_attempt["wav_sha256"],
        "unchanged_breath_free_reference": True,
        "D_speech_to_speech_ms": expected_left + expected_right,
        "splice_checks": checks,
        "output_filename": reference_path.relative_to(output).as_posix(),
        "output_sha256": file_sha256(reference_path),
    })

    for number, (attempt, _) in selected.items():
        if file_sha256(run / attempt["output_path"]) != attempt["wav_sha256"]:
            raise ValueError(f"Selected unit {number} changed during experiment.")
    if file_sha256(chapter) != CHAPTER_HASH:
        raise ValueError("Accepted chapter changed during experiment.")
    evidence = {
        "schema_version": 1,
        "purpose": "blind_heldout_narrator_inhale_generalization_experiment",
        "experimental_only": True,
        "source_run": str(run),
        "accepted_chapter_sha256": CHAPTER_HASH,
        "sample_rate_hz": RATE,
        "context_seconds_each_side": 4,
        "donor": {
            "source_unit_number": 70,
            "source_wav_path": donor_attempt["output_path"],
            "source_wav_sha256": DONOR_WAV_HASH,
            "start_frame": DONOR_START,
            "end_frame_exclusive": DONOR_END,
            "duration_frames": DONOR_FRAMES,
            "duration_ms": 240,
            "raw_pcm_sha256": DONOR_PCM_HASH,
            "audible_breath_duration_ms_estimate": 220,
            "quiet_shoulders_ms_estimate": {"before": 10, "after": 10},
        },
        "no_gain_fade_or_other_processing": True,
        "excluded_suspect_seam_after_unit": 61,
        "seams": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    result = build_experiment(args.run_directory, args.output_directory)
    total = sum(len(record.get("variants", [record])) for record in result["seams"])
    print(f"Created {total} blind/reference clips in {args.output_directory.resolve()}")
    print("Listen before opening manifest.json to preserve the blind mappings.")


if __name__ == "__main__":
    main()
