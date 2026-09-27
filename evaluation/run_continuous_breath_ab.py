"""Build a blind, copy-only 57-second sparse-breath comparison.

Uses selected units 20–23 from the accepted chapter. This is an evaluation
script with manually screened boundaries, not a production placement policy.
"""

import argparse
import hashlib
import json
from pathlib import Path

from evaluation.run_combined_seam_cleanup_ab import _selected
from evaluation.run_heldout_breath_ab import (
    CHAPTER_HASH, DONOR_END, DONOR_FRAMES, DONOR_PCM_HASH, DONOR_START,
    DONOR_WAV_HASH, RATE, _splice_steps,
)
from evaluation.run_seam_pause_ab import _source_context, _write_clip, edge_quiet_frames
from src.audiobook.cosyvoice import file_sha256
from src.audiobook.unit_execution import _load


UNIT_NUMBERS = (20, 21, 22, 23)
EXPECTED_QUIET_MS = {20: (160, 390), 21: (230, 180), 22: (170, 120)}
EDITED_AFTER_UNITS = (20, 22)


def render_passage(pcms, donor, breath):
    """Preserve speech and unit order; edit only two fixed 240-ms quiet blocks."""
    if len(pcms) != 4 or len(donor) != DONOR_FRAMES * 2:
        raise ValueError("Expected four selected PCM units and the exact 240-ms donor.")
    first, second, third, fourth = pcms
    if edge_quiet_frames(second, RATE, "leading") < DONOR_FRAMES:
        raise ValueError("Unit 21 no longer has enough measured leading quiet.")
    block = donor if breath else b"\x00\x00" * DONOR_FRAMES
    # 20→21: replace the same first 240 ms of unit 21's measured quiet in both files.
    # 21→22: exact original join. 22→23: insert the same 240-ms block in both files.
    pieces = (first, block, second[DONOR_FRAMES * 2:], third, block, fourth)
    positions = []
    cursor = 0
    for piece in pieces:
        positions.append(cursor)
        cursor += len(piece) // 2
    payload = b"".join(pieces)
    edits = {
        20: {"kind": "replace_first_240ms_of_unit21_measured_leading_quiet",
             "output_start_frame": positions[1], "source_unit": 21,
             "source_start_frame": 0, "frame_count": DONOR_FRAMES},
        22: {"kind": "insert_at_selected_unit_join",
             "output_start_frame": positions[4], "frame_count": DONOR_FRAMES},
    }
    joins = {
        20: positions[1],
        21: positions[3],
        22: positions[4],
    }
    boundaries = (
        ("20_to_first_block", positions[1]),
        ("first_block_to_unit21", positions[2]),
        ("unchanged_21_to_22", positions[3]),
        ("22_to_second_block", positions[4]),
        ("second_block_to_unit23", positions[5]),
    )
    checks = _splice_steps(payload, boundaries)
    if (payload[:len(first)] != first
            or payload[positions[2] * 2:positions[3] * 2] != second[DONOR_FRAMES * 2:]
            or payload[positions[3] * 2:positions[4] * 2] != third
            or payload[positions[5] * 2:] != fourth):
        raise ValueError("Protected source PCM differs from selected WAVs.")
    for after in EDITED_AFTER_UNITS:
        start = edits[after]["output_start_frame"]
        if payload[start * 2:(start + DONOR_FRAMES) * 2] != block:
            raise ValueError("Edited block does not match exact donor or zero PCM.")
    return payload, edits, joins, checks


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
        raise ValueError("Expected completed zero-added-silence schema-5 run.")
    chapter = (run / assembly["output_path"]).resolve()
    chapter.relative_to(run)
    if assembly["wav_sha256"] != CHAPTER_HASH or file_sha256(chapter) != CHAPTER_HASH:
        raise ValueError("Accepted chapter hash changed.")
    units = [unit for scene in manifest["scenes"] for unit in scene["synthesis_units"]]
    assembled = assembly["units"]
    selected = {n: _selected(run, manifest, units[n - 1], assembled[n - 1], RATE)
                for n in (*UNIT_NUMBERS, 70)}
    donor_attempt, donor_source = selected[70]
    donor = donor_source[DONOR_START * 2:DONOR_END * 2]
    if (donor_attempt["wav_sha256"] != DONOR_WAV_HASH
            or hashlib.sha256(donor).hexdigest() != DONOR_PCM_HASH):
        raise ValueError("Approved donor WAV or raw PCM differs.")
    pcms = [selected[n][1] for n in UNIT_NUMBERS]
    original_frames = sum(len(pcm) // 2 for pcm in pcms)
    first_assembly = assembled[19]
    last_assembly = assembled[22]
    if (first_assembly["start_frame"] / RATE != 324.16
            or last_assembly["end_frame_exclusive"] / RATE != 381.48
            or original_frames != last_assembly["end_frame_exclusive"]
            - first_assembly["start_frame"]):
        raise ValueError("Accepted continuous-passage source span changed.")
    original_chapter_pcm = b"".join(pcms)
    if len(original_chapter_pcm) != original_frames * 2:
        raise ValueError("Selected PCM frame count differs.")
    boundaries = []
    for after in (20, 21, 22):
        source = _source_context(units[after - 1], units[after])
        if source["structure"] != "paragraph_break":
            raise ValueError(f"Unit boundary {after} is no longer a paragraph break.")
        left_quiet = edge_quiet_frames(selected[after][1], RATE, "trailing")
        right_quiet = edge_quiet_frames(selected[after + 1][1], RATE, "leading")
        expected = EXPECTED_QUIET_MS[after]
        if (left_quiet, right_quiet) != tuple(ms * RATE // 1000 for ms in expected):
            raise ValueError(f"Quiet measurement changed after unit {after}.")
        boundaries.append({
            "after_unit_number": after,
            "structural_type": source["structure"],
            "source_context": source,
            "original_left_trailing_quiet_ms": expected[0],
            "original_right_leading_quiet_ms": expected[1],
            "modified": after in EDITED_AFTER_UNITS,
            "eligibility_screen": (
                "paragraph; no separate unexplained edge event or protected complete inhale"
                if after in EDITED_AFTER_UNITS else
                "eligible paragraph intentionally left untouched for sparse placement"),
        })
    rendered = {treatment: render_passage(pcms, donor, treatment == "sparse_donor")
                for treatment in ("quiet", "sparse_donor")}
    quiet, quiet_edits, quiet_joins, _ = rendered["quiet"]
    breath, breath_edits, breath_joins, _ = rendered["sparse_donor"]
    if len(quiet) != len(breath) or len(quiet) // 2 != original_frames + DONOR_FRAMES:
        raise ValueError("Continuous versions differ in duration or source span.")
    differing = [(edit["output_start_frame"] * 2,
                  (edit["output_start_frame"] + DONOR_FRAMES) * 2)
                 for edit in quiet_edits.values()]
    for start, end in sorted(differing, reverse=True):
        if quiet[start:end] != b"\x00\x00" * DONOR_FRAMES or breath[start:end] != donor:
            raise ValueError("Versions differ from exact zero/donor treatment.")
        quiet = quiet[:start] + breath[start:end] + quiet[end:]
    if quiet != breath:
        raise ValueError("Continuous versions differ outside the two donor intervals.")

    cases = sorted(("quiet", "sparse_donor"), key=lambda name: hashlib.sha256(
        f"{CHAPTER_HASH}:continuous_sparse:{name}".encode("ascii")
    ).digest())
    output.mkdir(parents=True)
    variants = []
    for label, treatment in zip("AB", cases):
        payload, edits, joins, checks = rendered[treatment]
        path = output / f"{label}.wav"
        _write_clip(path, RATE, payload)
        per_boundary = []
        for record in boundaries:
            after = record["after_unit_number"]
            left_ms = record["original_left_trailing_quiet_ms"]
            right_ms = record["original_right_leading_quiet_ms"]
            edited = after in EDITED_AFTER_UNITS
            if after == 20:
                qpost = right_ms - 240
                d = left_ms + right_ms
            elif after == 22:
                qpost = right_ms
                d = left_ms + 240 + right_ms
            else:
                qpost = right_ms
                d = left_ms + right_ms
            per_boundary.append({
                "after_unit_number": after,
                "modified": edited,
                "Qpre_quiet_before_block_ms": left_ms if edited else None,
                "block_kind": ("donor_pcm" if treatment == "sparse_donor" else "zero_pcm")
                    if edited else "none",
                "block_duration_ms": 240 if edited else 0,
                "audible_breath_duration_ms_estimate":
                    220 if edited and treatment == "sparse_donor" else 0,
                "Qpost_quiet_after_block_ms": qpost if edited else None,
                "Q_unmodified_quiet_ms": d if not edited else None,
                "D_speech_to_speech_ms": d,
                "inserted_zero_frames": (DONOR_FRAMES if after == 22 and treatment == "quiet" else 0),
                "inserted_donor_frames": (DONOR_FRAMES if after == 22 and treatment == "sparse_donor" else 0),
                "replaced_right_quiet_frames": DONOR_FRAMES if after == 20 else 0,
                "edit_location": edits.get(after),
                "output_join_frame": joins[after],
            })
        variants.append({
            "blind_label": label,
            "treatment": treatment,
            "donor_insertions": 2 if treatment == "sparse_donor" else 0,
            "source_unit_numbers": list(UNIT_NUMBERS),
            "output_frames": len(payload) // 2,
            "output_duration_seconds": len(payload) / (2 * RATE),
            "per_boundary": per_boundary,
            "splice_checks": checks,
            "output_filename": path.name,
            "output_sha256": file_sha256(path),
        })
    for n, (attempt, _) in selected.items():
        if file_sha256(run / attempt["output_path"]) != attempt["wav_sha256"]:
            raise ValueError(f"Selected unit {n} changed during experiment.")
    if file_sha256(chapter) != CHAPTER_HASH:
        raise ValueError("Accepted chapter changed during experiment.")
    evidence = {
        "schema_version": 1,
        "purpose": "blind_sparse_repeated_inhale_continuous_passage_experiment",
        "experimental_only": True,
        "source_run": str(run),
        "accepted_chapter_sha256": CHAPTER_HASH,
        "source_chapter_interval_seconds": [324.16, 381.48],
        "sample_rate_hz": RATE,
        "selected_source_wav_sha256": {str(n): selected[n][0]["wav_sha256"] for n in UNIT_NUMBERS},
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
        },
        "selection_note": "Two screened paragraph joins separated by an untouched paragraph join; no automatic rule.",
        "no_gain_fade_or_other_processing": True,
        "boundaries": boundaries,
        "variants": variants,
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
    print(f"Created {len(result['variants'])} blind continuous WAVs in "
          f"{args.output_directory.expanduser().resolve()}")
    print("Listen before opening manifest.json to preserve the blind mapping.")


if __name__ == "__main__":
    main()
