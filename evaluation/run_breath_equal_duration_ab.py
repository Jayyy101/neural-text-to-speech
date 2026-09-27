"""Build the approved, copy-only seven-clip breath-versus-quiet experiment.

All coordinates below are manual annotations for this listening experiment only.
This module is not imported by production assembly or generation.
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
DONOR_START = 480 * RATE // 1000
DONOR_END = 720 * RATE // 1000
CHAPTER_SHA256 = "a3dce3fbb9d6cdf333af32dcc260b9b9e02ba3458f865f137b2f747437ebba1b"
EXPECTED_DONOR_WAV_SHA256 = "8c8d3b46f7de1a475f36345a980292acc710437f6db23bdc0bd3f538f8f0a1eb"


def _ms(frames):
    return frames * 1000 / RATE


def _zeros(frames):
    return b"\x00\x00" * frames


def _splice_peak(left, right):
    """Absolute PCM16 step across an edit boundary, in sample counts."""
    return abs(int.from_bytes(left[-2:], "little", signed=True)
               - int.from_bytes(right[:2], "little", signed=True))


def render_case(after, treatment, left, right, donor):
    """Return exact PCM and its edit regions; caller supplies already cleaned units."""
    if len(donor) != (DONOR_END - DONOR_START) * 2:
        raise ValueError("Donor span has the wrong number of frames.")
    lctx, rctx = left[-CONTEXT_FRAMES * 2:], right[:CONTEXT_FRAMES * 2]
    regions = []
    if after == 69:
        if treatment == "native":
            payload = lctx + rctx
        elif treatment == "quiet_equal":
            start = DONOR_START - 400 * RATE // 1000
            end = DONOR_END - 400 * RATE // 1000
            if rctx[start * 2:end * 2] != donor:
                raise ValueError("Native donor interval differs from selected unit 70.")
            payload = lctx + rctx[:start * 2] + _zeros(end - start) + rctx[end * 2:]
            regions.append(("replace_native_inhale_with_zero_pcm", len(lctx) // 2 + start,
                            end - start))
        else:
            raise ValueError(treatment)
    elif after == 5:
        if treatment == "baseline":
            middle = b""
        elif treatment == "quiet_equal":
            middle = _zeros(240 * RATE // 1000)
            regions.append(("insert_zero_pcm_at_cleaned_join", len(lctx) // 2,
                            len(middle) // 2))
        elif treatment == "donor_equal":
            middle = donor
            regions.append(("insert_donor_at_cleaned_join", len(lctx) // 2,
                            len(donor) // 2))
        else:
            raise ValueError(treatment)
        payload = lctx + middle + rctx
    elif after == 58:
        if treatment == "quiet_equal":
            middle = _zeros(360 * RATE // 1000)
            regions.append(("insert_zero_pcm_at_cleaned_join", len(lctx) // 2,
                            len(middle) // 2))
        elif treatment == "donor_equal":
            prefix = _zeros(120 * RATE // 1000)
            middle = prefix + donor
            regions.extend((("insert_zero_pcm_before_donor", len(lctx) // 2,
                             len(prefix) // 2),
                            ("insert_donor", len(lctx) // 2 + len(prefix) // 2,
                             len(donor) // 2)))
        else:
            raise ValueError(treatment)
        payload = lctx + middle + rctx
    else:
        raise ValueError(after)
    return payload, regions, lctx, rctx


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
        raise ValueError("Expected accepted, zero-added-silence schema-5 run at 24 kHz.")
    chapter = (run / assembly["output_path"]).resolve()
    chapter.relative_to(run)
    if file_sha256(chapter) != CHAPTER_SHA256 or assembly["wav_sha256"] != CHAPTER_SHA256:
        raise ValueError("Accepted chapter hash changed.")
    units = [unit for scene in manifest["scenes"] for unit in scene["synthesis_units"]]
    assembled = assembly["units"]
    selected = {number: _selected(run, manifest, units[number - 1],
                                   assembled[number - 1], RATE)
                for number in (5, 6, 58, 59, 69, 70)}
    donor_attempt, donor_source = selected[70]
    if donor_attempt["wav_sha256"] != EXPECTED_DONOR_WAV_SHA256:
        raise ValueError("Approved donor source hash changed.")
    donor = donor_source[DONOR_START * 2:DONOR_END * 2]
    donor_hash = hashlib.sha256(donor).hexdigest()
    donor_wav = run / donor_attempt["output_path"]

    specifications = (
        (69, 120, 400, ("native", "quiet_equal"), 360),
        (5, 190, 0, ("baseline", "quiet_equal", "donor_equal"), None),
        (58, 160, 0, ("quiet_equal", "donor_equal"), 450),
    )
    output.mkdir(parents=True)
    seams = []
    for after, lcut_ms, rcut_ms, treatments, common_d in specifications:
        left_attempt, left_pcm = selected[after]
        right_attempt, right_pcm = selected[after + 1]
        left = left_pcm[:-(lcut_ms * RATE // 1000) * 2] if lcut_ms else left_pcm
        right = right_pcm[(rcut_ms * RATE // 1000) * 2:]
        source_context = _source_context(units[after - 1], units[after])
        if source_context["structure"] != "paragraph_break":
            raise ValueError(f"Source structure changed after unit {after}.")
        left_quiet = edge_quiet_frames(left, RATE, "trailing")
        right_quiet = edge_quiet_frames(right, RATE, "leading")
        expected = {5: (110, 100), 58: (60, 30)}
        if after in expected and (left_quiet, right_quiet) != tuple(
                value * RATE // 1000 for value in expected[after]):
            raise ValueError(f"Fixed cleanup quiet measurement changed after unit {after}.")
        if after == 69:
            # Manually verified left intended-speech end and right speech onset.
            left_speech_end = 19820 * RATE // 1000
            right_speech_onset = 740 * RATE // 1000
            transition_frames = ((len(left) // 2 - left_speech_end)
                                 + right_speech_onset - rcut_ms * RATE // 1000)
            if transition_frames != 360 * RATE // 1000:
                raise ValueError("Native speech-to-speech transition changed.")
        folder = output / f"after_unit_{after:04d}"
        folder.mkdir()
        randomized = sorted(treatments, key=lambda treatment: hashlib.sha256(
            f"{CHAPTER_SHA256}:breath_equal:{after}:{treatment}".encode("ascii")
        ).digest())
        variants = []
        for label, treatment in zip("ABC", randomized):
            payload, regions, lctx, rctx = render_case(after, treatment, left, right, donor)
            if after == 69:
                native = lctx + rctx
                if (payload[:len(lctx)] != lctx
                        or payload[:len(lctx) + (DONOR_START - rcut_ms * RATE // 1000) * 2]
                        != native[:len(lctx) + (DONOR_START - rcut_ms * RATE // 1000) * 2]
                        or payload[len(lctx) + (DONOR_END - rcut_ms * RATE // 1000) * 2:]
                        != native[len(lctx) + (DONOR_END - rcut_ms * RATE // 1000) * 2:]):
                    raise ValueError("Native context changed outside donor replacement.")
                qpre, b, qpost = (110, 220, 30) if treatment == "native" else (None, 0, None)
                q_total = 360 if treatment == "quiet_equal" else None
                d = 360
            elif after == 5:
                if treatment == "baseline":
                    q_total, qpre, b, qpost, d = 210, None, 0, None, 210
                elif treatment == "quiet_equal":
                    q_total, qpre, b, qpost, d = 450, None, 0, None, 450
                else:
                    q_total, qpre, b, qpost, d = None, 120, 220, 110, 450
            else:
                if treatment == "quiet_equal":
                    q_total, qpre, b, qpost, d = 450, None, 0, None, 450
                else:
                    q_total, qpre, b, qpost, d = None, 190, 220, 40, 450
            if (common_d is not None and d != common_d) or (
                    q_total is not None and q_total != d) or (
                    b and qpre + b + qpost != d):
                raise ValueError("Equal-duration event timing is inconsistent.")
            if after != 69 and (not payload.startswith(lctx)
                                 or not payload.endswith(rctx)):
                raise ValueError("Spoken context differs from selected source PCM.")
            # No fade is permitted. Check every edit boundary for an abrupt PCM step.
            splice_checks = []
            join_step = _splice_peak(payload[len(lctx) - 2:len(lctx)],
                                     payload[len(lctx):len(lctx) + 2])
            splice_checks.append({"output_frame": len(lctx) // 2,
                                  "pcm16_step": join_step, "kind": "cleaned_unit_join"})
            if join_step > 200:
                raise ValueError(f"Potential unit-join click after unit {after}: {join_step} PCM counts.")
            for kind, start, length in regions:
                for boundary in (start, start + length):
                    if boundary <= 0 or boundary >= len(payload) // 2:
                        continue
                    step = _splice_peak(payload[(boundary - 1) * 2:boundary * 2],
                                        payload[boundary * 2:(boundary + 1) * 2])
                    splice_checks.append({"output_frame": boundary, "pcm16_step": step})
                    if step > 200:
                        raise ValueError(f"Potential splice click after unit {after}: {step} PCM counts.")
            path = folder / f"{label}.wav"
            _write_clip(path, RATE, payload)
            variants.append({
                "blind_label": label,
                "treatment": treatment,
                "fixed_left_tail_trim_ms": lcut_ms,
                "fixed_right_head_trim_ms": rcut_ms,
                "Q_quiet_ms": q_total,
                "Qpre_quiet_ms": qpre,
                "B_audible_breath_ms_estimate": b,
                "Qpost_quiet_ms": qpost,
                "D_speech_to_speech_ms": d,
                "donor_pcm_span_ms_including_quiet_shoulders": 240 if b else 0,
                "inserted_zero_frames": sum(n for kind, _, n in regions
                                            if kind.startswith("insert_zero_pcm")),
                "zero_pcm_replacement_frames": sum(n for kind, _, n in regions
                                                    if kind.startswith("replace_")),
                "inserted_donor_frames": sum(n for kind, _, n in regions if "insert_donor" in kind),
                "replaced_native_donor_frames": (DONOR_END - DONOR_START)
                    if treatment == "quiet_equal" and after == 69 else 0,
                "edit_regions_output_frames": [
                    {"kind": kind, "start_frame": start, "frame_count": length}
                    for kind, start, length in regions],
                "splice_boundary_checks": splice_checks,
                "output_filename": path.relative_to(output).as_posix(),
                "output_sha256": file_sha256(path),
            })
        seams.append({
            "after_unit_number": after,
            "seam_id": f"{units[after-1]['id']}__to__{units[after]['id']}",
            "structural_boundary": source_context["structure"],
            "source_context": source_context,
            "left_source_wav_sha256": left_attempt["wav_sha256"],
            "right_source_wav_sha256": right_attempt["wav_sha256"],
            "fixed_cleanup": {"left_tail_trim_ms": lcut_ms, "right_head_trim_ms": rcut_ms},
            "measured_left_trailing_quiet_ms": _ms(left_quiet),
            "measured_right_leading_quiet_ms": _ms(right_quiet),
            "variants": variants,
        })
    for number, (attempt, _) in selected.items():
        if file_sha256(run / attempt["output_path"]) != attempt["wav_sha256"]:
            raise ValueError(f"Selected source WAV changed: unit {number}.")
    if file_sha256(chapter) != CHAPTER_SHA256:
        raise ValueError("Accepted chapter changed during experiment.")
    evidence = {
        "schema_version": 1,
        "purpose": "blind_equal_duration_natural_inhale_vs_quiet_experiment",
        "experimental_only": True,
        "source_run": str(run),
        "source_chapter_sha256": CHAPTER_SHA256,
        "sample_rate_hz": RATE,
        "context_seconds_each_side": CONTEXT_FRAMES // RATE,
        "donor": {
            "source_wav": str(donor_wav),
            "source_wav_sha256": donor_attempt["wav_sha256"],
            "start_frame_in_source": DONOR_START,
            "end_frame_exclusive_in_source": DONOR_END,
            "duration_frames": DONOR_END - DONOR_START,
            "duration_ms": _ms(DONOR_END - DONOR_START),
            "raw_pcm_sha256": donor_hash,
            "listener_review": "complete clean inhale; no hang or attached speech",
            "audible_breath_start_ms_estimate": 490,
            "audible_breath_end_ms_estimate": 710,
            "quiet_shoulders_ms_estimate": {"before": 10, "after": 10},
        },
        "edits": "PCM16 copies only; zero PCM and exact donor PCM; no gain, fade, or other processing",
        "seams": seams,
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
    print(f"Created {sum(len(s['variants']) for s in result['seams'])} blind clips in "
          f"{args.output_directory.expanduser().resolve()}")
    print("Listen before opening manifest.json to preserve blind labels.")


if __name__ == "__main__":
    main()
