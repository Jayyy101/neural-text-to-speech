"""Apply a manual JSON pause plan to one PCM16 WAV.

All timestamps refer to the ORIGINAL input WAV. Labels are reporting metadata
only. Uses the shared earliest-valley-within-3-dB selector; no text alignment or
boundary detection. Resolved points within 10 ms are rejected as ambiguous.
"""

import argparse
import json
from pathlib import Path
import wave

if __package__:
    from . import repair_pause as pause_audio
else:
    import repair_pause as pause_audio


def read_pause_plan(path):
    """Read a UTF-8 JSON object containing a nonempty pauses list."""
    try:
        plan = json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid UTF-8 JSON pause plan: {error}") from error
    if not isinstance(plan, dict) or not isinstance(plan.get("pauses"), list) or not plan["pauses"]:
        raise ValueError("Pause plan must contain a non-empty 'pauses' list.")
    for index, entry in enumerate(plan["pauses"], 1):
        if not isinstance(entry, dict):
            raise ValueError(f"Pause plan index {index} must be an object.")
        for key in ("around", "add_ms"):
            pause_audio.validate_positive(f"Pause plan index {index}: {key}", entry.get(key))
        if "label" in entry and not isinstance(entry["label"], str):
            raise ValueError(f"Pause plan index {index}: label must be text.")
    return plan["pauses"]


def apply_pause_plan(input_path, plan_path, output_path, search_ms=250.0):
    """Resolve against one original payload and write one final WAV, with no intermediates."""
    pause_audio.validate_positive("search-ms", search_ms)
    entries = read_pause_plan(plan_path)
    input_path, output_path = pause_audio.check_output_paths(input_path, output_path)
    data, rate, channels = pause_audio.read_pcm16(input_path)
    repairs = []
    for index, entry in enumerate(entries, 1):
        try:
            repair = pause_audio.plan_pause(data, rate, channels, entry["around"], entry["add_ms"], search_ms)
        except ValueError as error:
            raise ValueError(f"Pause plan index {index}: {error}") from error
        repair.update(plan_index=index, label=entry.get("label"))
        repairs.append(repair)

    # Only output coordinates accumulate silence; selection always used `data`.
    added_frames = 0
    for repair in sorted(repairs, key=lambda item: item["insertion_frame"]):
        repair["final_insertion_seconds"] = (repair["insertion_frame"] + added_frames) / rate
        added_frames += repair["added_frames"]
    pause_audio.write_pause_insertions(output_path, data, rate, channels, repairs)
    original_frames = len(data) // (channels * 2)
    return {
        "repairs": repairs,  # Original plan order; plan_index is one-based.
        "input_duration_seconds": original_frames / rate,
        "repair_count": len(repairs),
        "total_requested_add_ms": sum(entry["add_ms"] for entry in entries),
        "total_added_frames": added_frames,
        "total_actual_add_ms": added_frames / rate * 1000,
        "output_duration_seconds": (original_frames + added_frames) / rate,
        "output_path": str(output_path),
        "sample_rate_hz": rate,
        "channels": channels,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original PCM16 WAV.")
    parser.add_argument("--plan", type=Path, required=True, help="UTF-8 JSON pause plan.")
    parser.add_argument("--output", type=Path, required=True, help="New PCM16 WAV; must not exist.")
    parser.add_argument("--search-ms", type=float, default=250.0,
                        help="Search radius per original timestamp in ms (default: 250).")
    args = parser.parse_args(argv)
    try:
        result = apply_pause_plan(args.input, args.plan, args.output, args.search_ms)
    except (OSError, ValueError, wave.Error) as error:
        parser.error(str(error))

    for repair in result["repairs"]:
        label = json.dumps(repair["label"], ensure_ascii=True) if repair["label"] is not None else "(no label)"
        print(f"Plan index {repair['plan_index']} {label}: requested ORIGINAL {repair['around_seconds']:.6f} s; "
              f"search {repair['search_start_seconds']:.6f}..{repair['search_end_seconds']:.6f} s; "
              f"selected ORIGINAL {repair['insertion_seconds']:.6f} s (frame {repair['insertion_frame']}); "
              f"RMS {repair['quiet_region_rms_pcm16']:.3f} PCM16 units; "
              f"add {repair['requested_add_ms']:.6f} ms requested / {repair['actual_add_ms']:.6f} ms actual; "
              f"FINAL insertion start {repair['final_insertion_seconds']:.6f} s")
    print(f"Original duration: {result['input_duration_seconds']:.6f} s; repairs: {result['repair_count']}; "
          f"silence: {result['total_requested_add_ms']:.6f} ms requested / "
          f"{result['total_actual_add_ms']:.6f} ms actual ({result['total_added_frames']} frames); "
          f"final duration: {result['output_duration_seconds']:.6f} s")
    print(f"Output: {result['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
