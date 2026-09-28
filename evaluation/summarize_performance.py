"""Summarize an opt-in audiobook profiling directory without loading models.

Usage: python -B -m evaluation.summarize_performance PATH_TO_PROFILE
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median


def read_events(directory):
    directory = Path(directory)
    paths = sorted(directory.glob("events-*.jsonl"))
    if not paths:
        raise ValueError(f"No profiling trace files in {directory}")
    events = []
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid trace at {path}:{number}: {error}") from error
                if event.get("schema") != 1 or not all(
                        key in event for key in ("name", "span_id", "parent_id", "pid",
                                              "thread_id", "start_ns", "end_ns",
                                              "duration_ns", "outcome", "metadata")):
                    raise ValueError(f"Unsupported trace event at {path}:{number}")
                events.append(event)
    return events


def _seconds(event):
    return event["duration_ns"] / 1_000_000_000


def _clock_key(event):
    """Monotonic timestamps are comparable only within one process."""
    return event.get("process_uid") or ("pid", event["pid"])


def _union_ns(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def exclusive_seconds(event, by_parent):
    """Subtract direct child coverage, merging overlaps on the same process."""
    children = by_parent.get(event["span_id"], ())
    intervals = [
        (max(event["start_ns"], child["start_ns"]),
         min(event["end_ns"], child["end_ns"]))
        for child in children if _clock_key(child) == _clock_key(event)
    ]
    return max(0, event["duration_ns"] - _union_ns(
        (start, end) for start, end in intervals if start < end
    )) / 1_000_000_000


def summarize(events):
    by_name = defaultdict(list)
    by_parent = defaultdict(list)
    for event in events:
        by_name[event["name"]].append(event)
        by_parent[event["parent_id"]].append(event)

    def total(name):
        return sum(_seconds(event) for event in by_name[name])

    def detail(name):
        if name == "tts.conditioning" and not by_name[name]:
            return {"inclusive_s": None, "exclusive_s": None, "count": 0}
        return {"inclusive_s": round(total(name), 6),
                "exclusive_s": round(sum(exclusive_seconds(event, by_parent)
                                         for event in by_name[name]), 6),
                "count": len(by_name[name])}

    roots = by_name["ui.generate_to_complete"]
    if len(roots) > 1:
        raise ValueError("Profile contains multiple UI jobs; summarize one job directory.")
    root = roots[0] if roots else None
    export_marks = by_name["ui.export_ready"]
    export_ready = None
    if root and export_marks:
        marks = [event for event in export_marks
                 if _clock_key(event) == _clock_key(root)]
        if marks:
            export_ready = (marks[0]["end_ns"] - root["start_ns"]) / 1_000_000_000

    attempts = by_name["unit.attempt"]
    synth = by_name["tts.inference"]
    by_unit = defaultdict(list)
    for event in attempts:
        unit_id = event["metadata"].get("unit_id")
        if unit_id:
            by_unit[unit_id].append(event)
    per_unit = []
    for unit_id, unit_attempts in sorted(by_unit.items()):
        unit_attempts.sort(key=lambda event: event["start_ns"])
        selected_id = next((event["metadata"].get("selected_attempt_id")
                            for event in by_name["unit.cycle"]
                            if event["metadata"].get("unit_id") == unit_id), None)
        chosen = next((event for event in unit_attempts
                       if event["metadata"].get("attempt_id") == selected_id),
                      unit_attempts[-1])
        chosen_id = chosen["metadata"].get("attempt_id")
        selected_events = [event for event in synth
                           if event["metadata"].get("unit_id") == unit_id
                           and event["metadata"].get("attempt_id") == chosen_id]
        inference = sum(_seconds(event) for event in selected_events) if selected_events else None
        all_inference = sum(_seconds(event) for event in synth
                            if event["metadata"].get("unit_id") == unit_id)
        audio = chosen["metadata"].get("audio_seconds")
        per_unit.append({
            "unit_id": unit_id, "selected_attempt_id": chosen_id,
            "attempts": len(unit_attempts),
            "synthesis_inference_s": round(inference, 6) if inference is not None else None,
            "all_attempt_inference_s": round(all_inference, 6) if all_inference else None,
            "generated_audio_s": audio,
            "rtf": round(inference / audio, 6) if inference is not None
                   and isinstance(audio, (int, float))
                   and audio > 0 else None,
        })

    rejected = sum(event["metadata"].get("decision") == "rejected"
                   for event in by_name["unit.qc"])
    elapsed = _seconds(root) if root else None
    # The Windows process supplies both endpoints of the worker wall-time
    # window. WSL durations describe its interior but cannot be placed on this
    # clock or subtracted from this window to infer bootstrap/exit latency.
    windows_key = _clock_key(root) if root else None
    exits = [event for event in by_name["launcher.worker_exit_observed"]
             if windows_key is not None and _clock_key(event) == windows_key]
    launches = [event for event in by_name["launcher.process_launch"]
                if windows_key is not None and _clock_key(event) == windows_key]
    lifecycle = (root and len(exits) == len(launches) == 1
                 and root["start_ns"] <= launches[0]["start_ns"]
                 <= launches[0]["end_ns"] <= exits[0]["end_ns"]
                 <= root["end_ns"])
    worker_interval = ((launches[0]["end_ns"], exits[0]["end_ns"])
                       if lifecycle else None)
    worker_elapsed = ((exits[0]["end_ns"] - launches[0]["start_ns"]) / 1_000_000_000
                      if lifecycle else None)

    # These UI/launcher spans are on the same Windows clock. Merge intervals
    # so nested command/path spans and the observed worker window count once.
    critical_names = (
        "ui.source_write", "ui.launch", "launcher.command_preparation",
        "launcher.wslpath", "launcher.process_launch",
        "ui.final_validation", "ui.named_file_copy",
    )
    intervals = []
    if root:
        for name in critical_names:
            intervals.extend(
                (max(root["start_ns"], event["start_ns"]),
                 min(root["end_ns"], event["end_ns"]))
                for event in by_name[name]
                if _clock_key(event) == windows_key
                and max(root["start_ns"], event["start_ns"])
                < min(root["end_ns"], event["end_ns"])
            )
    ui_intervals = intervals
    if worker_interval:
        intervals = [*ui_intervals, worker_interval]
    classified = _union_ns(intervals) / 1_000_000_000 if root else None
    remainder = max(0, elapsed - classified) if root else None

    def unmeasured_window(start, end):
        measured = _union_ns(
            (max(start, left), min(end, right)) for left, right in ui_intervals
            if max(start, left) < min(end, right)
        )
        return (end - start - measured) / 1e9

    critical_path = {
        "ui_before_worker_s": ((launches[0]["end_ns"] - root["start_ns"]) / 1e9
                               if lifecycle else None),
        "worker_launch_to_exit_observed_s": ((worker_interval[1] - worker_interval[0]) / 1e9
                                             if worker_interval else None),
        "ui_after_worker_s": ((root["end_ns"] - exits[0]["end_ns"]) / 1e9
                              if lifecycle else None),
    }
    unmeasured_windows = {
        "before_worker_s": (unmeasured_window(root["start_ns"], worker_interval[0])
                            if lifecycle else None),
        "after_worker_s": (unmeasured_window(worker_interval[1], root["end_ns"])
                           if lifecycle else None),
    }

    # A missing CLI completion event is common when the daemon trace writer
    # loses its final queued record. Its children still describe useful WSL
    # work, but they cannot fill a Windows-clock gap.
    backend_markers = by_name["cli.entry"] or by_name["workflow.generation"]
    backend_key = _clock_key(backend_markers[0]) if len(backend_markers) == 1 else None
    backend_events = ([event for event in events if _clock_key(event) == backend_key]
                      if backend_key is not None else [])
    backend_coverage = (_union_ns((event["start_ns"], event["end_ns"])
                                  for event in backend_events) / 1e9
                        if backend_events else None)
    backend_envelope = ((max(event["end_ns"] for event in backend_events)
                         - min(event["start_ns"] for event in backend_events)) / 1e9
                        if backend_events else None)
    command_recorded = any(event["name"] == "cli.command" for event in backend_events)
    assembly_root = by_name["workflow.assembly"] or by_name["assembly.total"]
    assembly_duration = (sum(_seconds(event) for event in assembly_root)
                         if assembly_root else None)
    assembly_subphases = [event for event in backend_events
                          if event["name"].startswith("assembly.")
                          and event["name"] != "assembly.total"]
    assembly_observed = (_union_ns((event["start_ns"], event["end_ns"])
                                   for event in assembly_subphases) / 1e9
                         if assembly_subphases else None)

    totals = {
        "startup_cli_imports_and_preflight_s": total("cli.startup_imports") + total("cli.preflight"),
        "planning_and_frontend_release_s": sum(total(name) for name in (
            "workflow.planning", "workflow.unit_preparation", "workflow.frontend_release")),
        "model_loads_s": sum(total(name) for name in (
            "model.planning_load", "model.synthesis_load", "asr.model_load")),
        "synthesis_backend_calls_s": total("unit.synthesis"),
        "cosyvoice_inference_s": total("tts.inference"),
        "narrator_conditioning_s": (total("tts.conditioning")
                                   if by_name["tts.conditioning"] else None),
        "content_qc_s": total("unit.qc"),
        "manifest_persistence_s": total("manifest.save"),
        "assembly_s": assembly_duration,
        "assembly_observed_subphases_s": assembly_observed,
        "final_validation_s": total("ui.final_validation"),
        "named_file_copy_s": total("ui.named_file_copy"),
        "progress_inspection_s": total("ui.progress_inspection"),
    }
    measured_warm = [item["synthesis_inference_s"] for item in per_unit[1:]
                     if item["synthesis_inference_s"] is not None]

    return {
        "end_to_end_generate_to_complete_s": round(elapsed, 6) if elapsed is not None else None,
        "generate_to_export_ready_s": round(export_ready, 6) if export_ready is not None else None,
        "completion_outcome": root["outcome"] if root else None,
        "worker_launch_to_exit_observed_s": (round(worker_elapsed, 6)
                                             if worker_elapsed is not None else None),
        "launch_bootstrap_exit_poll_gap_s": None,
        "critical_path_segments_s": {name: round(value, 6) if value is not None else None
                                     for name, value in critical_path.items()},
        "unmeasured_windows_s": {name: round(value, 6) if value is not None else None
                                 for name, value in unmeasured_windows.items()},
        "backend_local_observed_coverage_s": (round(backend_coverage, 6)
                                              if backend_coverage is not None else None),
        "backend_local_unmeasured_between_spans_s": (
            round(max(0, backend_envelope - backend_coverage), 6)
            if backend_envelope is not None else None),
        "backend_command_span_recorded": command_recorded,
        "critical_path_accounted_s": round(classified, 6) if root else None,
        "unclassified_remainder_s": round(remainder, 6) if remainder is not None else None,
        "note": ("Chapter critical-path accounting uses Windows-clock intervals: "
                 "measured UI/launcher spans plus the observed worker window. "
                 "The remainder is unmeasured Windows time outside those intervals. "
                 "WSL phase durations use a separate clock and are inclusive details; "
                 "they must not be added to or subtracted from the Windows wall total. "
                 "Bootstrap/exit/poll latency is unavailable without correlated endpoints. "
                 "Progress inspection overlaps the worker window."),
        "phases": {name: detail(name) for name in (
            "ui.source_write", "launcher.command_preparation", "launcher.wslpath",
            "launcher.process_launch", "cli.startup_imports", "cli.preflight",
            "cli.command", "workflow.planning", "workflow.unit_preparation",
            "workflow.frontend_release", "workflow.generation", "workflow.assembly",
            "planning.normalize", "planning.normalize_heading",
            "model.planning_initialize", "model.planning_load",
            "model.synthesis_initialize", "model.synthesis_load", "model.runtime_imports",
            "model.verify_rl_view", "model.frontend_verification", "asr.worker_startup",
            "asr.model_initialization", "asr.model_load", "unit.cycle", "unit.attempt", "unit.synthesis",
            "tts.inference", "tts.conditioning", "tts.cache_prepare", "tts.seed_prepare",
            "tts.wav_write", "unit.qc", "unit.qc_evidence_write", "asr.round_trip",
            "asr.wav_hash", "asr.audio_preparation", "asr.feature_extraction",
            "asr.gpu_inference", "asr.decoding", "asr.content_comparison",
            "manifest.read", "manifest.save", "manifest.serialize", "manifest.write",
            "manifest.replace", "manifest.lock_wait", "assembly.total",
            "assembly.selected_validation", "assembly.read_pcm", "assembly.write_pcm",
            "io.file_hash", "io.wav_validation", "ui.progress_inspection",
            "ui.final_validation", "ui.named_file_copy",
        )},
        "totals": {name: round(value, 6) if value is not None else None
                   for name, value in totals.items()},
        "first_unit_inference_s": per_unit[0]["synthesis_inference_s"] if per_unit else None,
        "warm_unit_inference_s": round(sum(measured_warm), 6) if measured_warm else None,
        "warm_unit_median_s": round(median(measured_warm), 6) if measured_warm else None,
        "warm_unit_count": len(measured_warm),
        "rejected_qc_attempts": rejected,
        "extra_physical_attempts": sum(max(0, len(items) - 1) for items in by_unit.values()),
        "per_unit": per_unit,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_directory", type=Path)
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary.")
    parser.add_argument("--details", action="store_true", help="Also print every measured phase.")
    args = parser.parse_args(argv)
    summary = summarize(read_events(args.profile_directory))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    print(f"Generate -> Complete: {summary['end_to_end_generate_to_complete_s']} s "
          f"({summary['completion_outcome']})")
    print(f"Generate -> export ready: {summary['generate_to_export_ready_s']} s")
    print(f"Worker launch -> exit observed: {summary['worker_launch_to_exit_observed_s']} s; "
          "bootstrap/exit/poll gap: unavailable")
    print("Windows critical-path windows (s): "
          + ", ".join(f"{name}={value}" for name, value in
                      summary["critical_path_segments_s"].items()))
    print(f"Backend-local observed span coverage: "
          f"{summary['backend_local_observed_coverage_s']} s; "
          f"CLI command completion span recorded: {summary['backend_command_span_recorded']}")
    print(f"Backend-local gaps between recorded spans: "
          f"{summary['backend_local_unmeasured_between_spans_s']} s "
          "(excludes unknown time before/after the trace)")
    print("Totals in seconds (inclusive; nested totals overlap):")
    for name, value in summary["totals"].items():
        print(f"  {name:38s} {value:10.3f}" if value is not None
              else f"  {name:38s} unavailable")
    if args.details:
        print("Measured phases: inclusive / exclusive / count")
        for name, phase in summary["phases"].items():
            if phase["count"]:
                print(f"  {name:32s} {phase['inclusive_s']:10.3f} / "
                      f"{phase['exclusive_s']:10.3f} / {phase['count']}")
    print(f"Windows critical-path accounted: {summary['critical_path_accounted_s']} s")
    print(f"Unmeasured Windows time outside worker window: "
          f"{summary['unclassified_remainder_s']} s")
    print("  Before worker: {before_worker_s} s; after worker: {after_worker_s} s".format(
        **summary["unmeasured_windows_s"]))
    print(f"First-unit inference: {summary['first_unit_inference_s']} s; "
          f"warm-unit total/median: {summary['warm_unit_inference_s']} / "
          f"{summary['warm_unit_median_s']} s over {summary['warm_unit_count']} units")
    print(f"QC rejections: {summary['rejected_qc_attempts']}; "
          f"extra physical attempts: {summary['extra_physical_attempts']}")
    print("Per unit: ID | attempts | inference s | audio s | RTF")
    for unit in summary["per_unit"]:
        print(f"  {unit['unit_id']} | {unit['attempts']} | "
              f"{unit['synthesis_inference_s']} | "
              f"{unit['generated_audio_s']} | {unit['rtf']}")
    print(summary["note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
