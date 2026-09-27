"""Build the frozen Chapter 2 prospective full-chapter breath comparison.

This evaluation-only renderer inserts one exact 240-ms block at each frozen
join. It never trims, regenerates, or writes into either accepted run.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from evaluation.run_combined_seam_cleanup_ab import _selected
from evaluation.run_heldout_breath_ab import (
    DONOR_END, DONOR_FRAMES, DONOR_PCM_HASH, DONOR_START, DONOR_WAV_HASH,
    RATE, _splice_steps,
)
from evaluation.run_seam_pause_ab import _source_context, _write_clip, edge_quiet_frames
from src.audiobook.cosyvoice import file_sha256
from src.audiobook.unit_execution import _load


CHAPTER_HASH = "b945c14800adfcb7f09483d8994ff8ac6b3e31e742261f23af8615443fd504c8"
SELECTED_AFTER_UNITS = (2, 9, 16, 22, 30, 36, 44, 50)
EXPECTED_QUIET_MS = {
    2: (120, 170), 9: (90, 180), 16: (140, 110), 22: (190, 160),
    30: (160, 170), 36: (60, 120), 44: (210, 110), 50: (120, 80),
}

BASELINE_LISTENING_OBSERVATIONS = (
    ("0:03-0:04", "当江绫，章岳", "approximately 当江绫章 → pause/inhale → 岳"),
    ("0:05", "离去后，议事厅内", "approximately 离去后议事 → pause/inhale → 厅内"),
    ("0:10", "还是性子火爆的陆惊雷忍耐不住", "unusual pause/inhale before 忍耐不住"),
    ("2:16", "这让得陆鸣一下子知道", "approximately 这 → pause/inhale → 让得陆鸣一下子知道"),
    ("4:15", "这么突然?不多做一些准备吗?", "first question mark had essentially no perceptible pause"),
    ("4:25", "忐忑不安。陆瑾哑然", "essentially no pause across this transition"),
    ("4:27", "准备?做什么准备", "question mark had essentially no perceptible pause"),
    ("5:54", "这三代人口都", "slight unusual pause between 人 and 口"),
    ("6:25", "有些惊喜:”上品开骨丹?”", "dialogue introduction/colon pause felt too short"),
    ("7:41", "开骨丹药力", "unusual pause between 开骨丹 and 药力"),
    ("8:08", "银芒?中品神通骨吗", "essentially no perceptible pause at question mark"),
    ("9:11", "盯着陆鸣胸腔处", "pause/inhale between 陆鸣 and 胸腔处"),
    ("9:18", "他眼瞳微缩", "incorrect/hallucinated sound perceived approximately as ‘pao kai’"),
    ("9:24", "而陆青炎，陆惊雷也很快", "incorrect/hallucinated sound perceived approximately as ‘zhe’"),
    ("9:30", "结结巴巴起来", "incorrect/hallucinated sound perceived approximately as ‘mian’"),
    ("11:06", "后被“血痣”所吞噬。", "final 噬 truncated; approximately only a ‘sh’ onset"),
    ("12:40", "盯着陆鸣眉心", "unusual pause between 陆鸣 and 眉心"),
)


def render_full_chapter(pcms, block, selected_after_units=SELECTED_AFTER_UNITS):
    """Concatenate source PCM unchanged, inserting one block at fixed joins."""
    selected = set(selected_after_units)
    if len(selected) != len(selected_after_units):
        raise ValueError("Selected boundary numbers must be distinct.")
    if len(block) != DONOR_FRAMES * 2:
        raise ValueError("Inserted block must be exactly 5760 PCM16 frames.")
    if any(not 1 <= after < len(pcms) for after in selected):
        raise ValueError("Selected boundary falls outside the supplied units.")
    pieces, edits, cursor = [], {}, 0
    for number, pcm in enumerate(pcms, 1):
        if len(pcm) % 2:
            raise ValueError("Source unit has an incomplete PCM16 frame.")
        pieces.append(pcm)
        cursor += len(pcm) // 2
        if number in selected:
            edits[number] = cursor
            pieces.append(block)
            cursor += DONOR_FRAMES
    return b"".join(pieces), edits


def build_experiment(run_directory, donor_run_directory, output_directory):
    run, _, manifest = _load(run_directory)
    donor_run, _, donor_manifest = _load(donor_run_directory)
    output = Path(output_directory).expanduser().resolve()
    if output.exists() or output in (run, donor_run) or run in output.parents or donor_run in output.parents:
        raise ValueError("Output must be new and outside both accepted runs.")

    assembly = manifest.get("assembly", {})
    if (manifest.get("schema_version") != 5 or manifest.get("status") != "generated"
            or assembly.get("status") != "assembled"
            or assembly.get("extra_silence_ms_between_units") != 0
            or assembly.get("audio", {}).get("sample_rate_hz") != RATE):
        raise ValueError("Expected the accepted schema-5 24-kHz zero-silence run.")
    chapter = (run / assembly["output_path"]).resolve()
    chapter.relative_to(run)
    if assembly.get("wav_sha256") != CHAPTER_HASH or file_sha256(chapter) != CHAPTER_HASH:
        raise ValueError("Accepted Chapter 2 hash changed.")
    units = [unit for scene in manifest["scenes"] for unit in scene["synthesis_units"]]
    assembled = assembly["units"]
    if len(units) != 52 or len(assembled) != 52:
        raise ValueError("Expected the accepted 52-unit Chapter 2 run.")
    selected = [_selected(run, manifest, unit, item, RATE)
                for unit, item in zip(units, assembled)]
    pcms = [item[1] for item in selected]

    donor_units = [unit for scene in donor_manifest["scenes"] for unit in scene["synthesis_units"]]
    donor_assembled = donor_manifest["assembly"]["units"]
    donor_attempt, donor_source = _selected(
        donor_run, donor_manifest, donor_units[69], donor_assembled[69], RATE)
    donor = donor_source[DONOR_START * 2:DONOR_END * 2]
    if (donor_attempt["wav_sha256"] != DONOR_WAV_HASH
            or file_sha256(donor_run / donor_attempt["output_path"]) != DONOR_WAV_HASH
            or hashlib.sha256(donor).hexdigest() != DONOR_PCM_HASH):
        raise ValueError("Approved donor WAV or exact raw PCM changed.")

    original_pcm = b"".join(pcms)
    if len(original_pcm) // 2 != assembly["audio"]["frames"]:
        raise ValueError("Selected source PCM frame total differs from assembly.")
    boundary_records = []
    for after in SELECTED_AFTER_UNITS:
        context = _source_context(units[after - 1], units[after])
        if context["structure"] != "paragraph_break":
            raise ValueError(f"Frozen boundary {after} is no longer a paragraph break.")
        left = edge_quiet_frames(pcms[after - 1], RATE, "trailing")
        right = edge_quiet_frames(pcms[after], RATE, "leading")
        expected = tuple(value * RATE // 1000 for value in EXPECTED_QUIET_MS[after])
        if (left, right) != expected:
            raise ValueError(f"Frozen quiet measurement changed after unit {after}.")
        boundary_records.append({
            "after_unit_number": after,
            "source_context": context,
            "original_left_trailing_quiet_ms": EXPECTED_QUIET_MS[after][0],
            "original_right_leading_quiet_ms": EXPECTED_QUIET_MS[after][1],
            "Qpre_ms": EXPECTED_QUIET_MS[after][0],
            "block_ms": 240,
            "Qpost_ms": EXPECTED_QUIET_MS[after][1],
            "matched_D_ms": sum(EXPECTED_QUIET_MS[after]) + 240,
            "insertion_location": "physical selected-unit join",
        })

    zero = b"\x00\x00" * DONOR_FRAMES
    rendered = {
        "matched_quiet": render_full_chapter(pcms, zero),
        "sparse_donor": render_full_chapter(pcms, donor),
    }
    quiet_pcm, quiet_edits = rendered["matched_quiet"]
    donor_pcm, donor_edits = rendered["sparse_donor"]
    expected_frames = assembly["audio"]["frames"] + len(SELECTED_AFTER_UNITS) * DONOR_FRAMES
    if (len(quiet_pcm) != len(donor_pcm) or len(quiet_pcm) // 2 != expected_frames
            or quiet_edits != donor_edits):
        raise ValueError("Experimental versions are not duration/location matched.")
    reconstructed = bytearray(quiet_pcm)
    for start in quiet_edits.values():
        begin, end = start * 2, (start + DONOR_FRAMES) * 2
        if quiet_pcm[begin:end] != zero or donor_pcm[begin:end] != donor:
            raise ValueError("An inserted block differs from exact zero/donor PCM.")
        reconstructed[begin:end] = donor_pcm[begin:end]
    if bytes(reconstructed) != donor_pcm:
        raise ValueError("Experimental versions differ outside the eight matched blocks.")

    output.mkdir(parents=True)
    reference = output / "reference.wav"
    shutil.copyfile(chapter, reference)
    if file_sha256(reference) != CHAPTER_HASH:
        raise ValueError("Untouched reference copy differs from accepted chapter.")
    order = sorted(rendered, key=lambda treatment: hashlib.sha256(
        f"{CHAPTER_HASH}:prospective_ch02:{treatment}".encode("ascii")).digest())
    variants = []
    for label, treatment in zip("AB", order):
        pcm, edits = rendered[treatment]
        path = output / f"{label}.wav"
        _write_clip(path, RATE, pcm)
        checks = []
        for after, start in edits.items():
            checks.extend(_splice_steps(pcm, (
                (f"after_unit_{after}_left_to_block", start),
                (f"after_unit_{after}_block_to_right", start + DONOR_FRAMES),
            )))
        variants.append({
            "blind_label": label,
            "treatment": treatment,
            "output_filename": path.name,
            "output_sha256": file_sha256(path),
            "output_frames": len(pcm) // 2,
            "output_duration_seconds": len(pcm) / (2 * RATE),
            "placement_count": len(edits),
            "inserted_zero_frames_per_placement": DONOR_FRAMES if treatment == "matched_quiet" else 0,
            "inserted_donor_frames_per_placement": DONOR_FRAMES if treatment == "sparse_donor" else 0,
            "output_start_frames": {str(key): value for key, value in edits.items()},
            "splice_checks": checks,
        })

    source_hashes = []
    for number, (attempt, _) in enumerate(selected, 1):
        path = run / attempt["output_path"]
        if file_sha256(path) != attempt["wav_sha256"]:
            raise ValueError(f"Selected source WAV {number} changed.")
        source_hashes.append({"unit_number": number, "wav_sha256": attempt["wav_sha256"]})
    if file_sha256(chapter) != CHAPTER_HASH:
        raise ValueError("Accepted Chapter 2 changed during experiment.")

    evidence = {
        "schema_version": 1,
        "purpose": "prospective_same_narrator_sparse_breath_full_chapter_comparison",
        "experimental_only": True,
        "source_run": str(run),
        "accepted_chapter_path": str(chapter),
        "accepted_chapter_sha256": CHAPTER_HASH,
        "reference": {"output_filename": reference.name, "output_sha256": file_sha256(reference)},
        "frozen_policy": {
            "selected_after_units": list(SELECTED_AFTER_UNITS),
            "already_long_threshold_ms": 500,
            "minimum_sparse_spacing_seconds": 90,
            "selection_frozen_before_baseline_listening": True,
            "listening_observations_changed_selection_or_timing": False,
            "no_trimming_or_source_speech_modification": True,
        },
        "donor": {
            "source_run": str(donor_run), "source_unit_number": 70,
            "source_wav_path": donor_attempt["output_path"],
            "source_wav_sha256": DONOR_WAV_HASH,
            "start_frame": DONOR_START, "end_frame_exclusive": DONOR_END,
            "duration_frames": DONOR_FRAMES, "duration_ms": 240,
            "raw_pcm_sha256": DONOR_PCM_HASH,
        },
        "boundaries": boundary_records,
        "variants": variants,
        "selected_source_wav_hashes": source_hashes,
        "baseline_listening_observations": {
            "recorded_after_sparse_plan_was_frozen": True,
            "used_to_change_selection_timing_or_audio": False,
            "overall": "Reasonably good for 13:48; other small pauses sounded natural/acceptable; listed items were unusually noticeable.",
            "items": [{"timestamp": time, "text": text, "observation": observation}
                      for time, text, observation in BASELINE_LISTENING_OBSERVATIONS],
        },
        "processing": {
            "source_speech_preserved_exactly": True,
            "gain_normalization_fade_denoise_pitch_or_time_processing": False,
            "synthesis_or_regeneration_performed": False,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("donor_run_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    result = build_experiment(args.run_directory, args.donor_run_directory, args.output_directory)
    print(f"Created reference plus {len(result['variants'])} blind WAVs in {args.output_directory.resolve()}")
    print("Listen before opening manifest.json to preserve the blind mapping.")


if __name__ == "__main__":
    main()
