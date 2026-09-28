"""Evaluate a frozen end-of-unit completion rule on saved independent ASR."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.run_mandarin_asr_unit09_feasibility import (
    file_sha256,
    levenshtein_steps,
    require,
    save_json,
)


SOURCE_REPORT = ROOT / (
    "outputs/evaluation/mandarin_asr_history_audit_2026-09-20_10-10-14_697193/"
    "report.json"
)
SOURCE_REPORT_SHA256 = "480186c29db71da7eb9ea8cb6c1ec4c9fac2055d2f998cff49d30672e11d724e"
TAIL_EXPECTED_HAN_COUNT = 4
MAX_TERMINAL_MARGIN_SECONDS = 0.200

KNOWN_BAD_ID = "history_047"
KNOWN_GOOD_IDS = (
    "history_004", "history_005",
    "history_018", "history_019", "history_020", "history_021",
    "history_022", "history_023", "history_024", "history_025",
    "history_027", "history_028", "history_029",
    "history_030", "history_031", "history_032", "history_033",
    "history_034", "history_035",
)


def contiguous_deleted_suffix(tail_steps):
    count = 0
    for step in reversed(tail_steps):
        if step["operation"] != "deletion":
            break
        count += 1
    return count


def inspect_tail(result):
    expected = result["intended"]["comparison_tokens"]
    recognized_meta = result["comparison_tokens"]
    recognized = [item["comparison_token"] for item in recognized_meta]
    steps, _ = levenshtein_steps(expected, recognized)
    tail_start = max(0, len(expected) - TAIL_EXPECTED_HAN_COUNT)
    tail_steps = [
        step for step in steps
        if step["expected_index"] is not None
        and step["expected_index"] >= tail_start
    ]
    require(len(tail_steps) == len(expected) - tail_start,
            f"Incomplete tail alignment: {result['id']}")
    deleted_suffix = contiguous_deleted_suffix(tail_steps)
    aligned_tail_steps = [
        step for step in tail_steps if step["recognized_index"] is not None
    ]
    last_tail_step = aligned_tail_steps[-1] if aligned_tail_steps else None
    last_recognized_index = len(recognized_meta) - 1
    tail_reaches_last_asr_token = (
        last_tail_step is not None
        and last_tail_step["recognized_index"] == last_recognized_index
    )
    last_token_end = (
        recognized_meta[last_recognized_index]["end_seconds"]
        if recognized_meta else None
    )
    terminal_margin = (
        result["audio"]["duration_seconds"] - last_token_end
        if last_token_end is not None else None
    )
    eligible_tail_coverage = (
        deleted_suffix == 0
        or 1 <= deleted_suffix <= 3
    )
    flagged = bool(
        eligible_tail_coverage
        and tail_reaches_last_asr_token
        and terminal_margin is not None
        and terminal_margin <= MAX_TERMINAL_MARGIN_SECONDS
    )
    return {
        "expected_tail_start": tail_start,
        "expected_tail": "".join(expected[tail_start:]),
        "alignment": [{
            "expected_index": step["expected_index"],
            "expected": step["expected"],
            "operation": step["operation"],
            "recognized": step["recognized"],
            "recognized_index": step["recognized_index"],
            "recognized_time_seconds": (
                [recognized_meta[step["recognized_index"]]["start_seconds"],
                 recognized_meta[step["recognized_index"]]["end_seconds"]]
                if step["recognized_index"] is not None else None
            ),
        } for step in tail_steps],
        "recognized_tail_rendering": "".join(
            step["recognized"] if step["recognized"] is not None else "∅"
            for step in tail_steps
        ),
        "contiguous_deleted_expected_suffix": deleted_suffix,
        "last_asr_token": recognized[-1] if recognized else None,
        "last_asr_token_end_seconds": last_token_end,
        "audio_duration_seconds": result["audio"]["duration_seconds"],
        "terminal_margin_seconds": terminal_margin,
        "tail_reaches_last_asr_token": tail_reaches_last_asr_token,
        "flagged": flagged,
    }


def execute(output_parent):
    require(SOURCE_REPORT.is_file(), "Saved independent-ASR report is missing")
    require(file_sha256(SOURCE_REPORT) == SOURCE_REPORT_SHA256,
            "Saved independent-ASR report hash differs")
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    require(source.get("status") == "completed", "Source ASR audit is incomplete")
    require(source.get("intended_text_used_during_inference") is False,
            "Source ASR did not record independent inference")
    require(source.get("all_audio_inference_completed_before_intended_text_load") is True,
            "Source intended text was not isolated from inference")
    require(source.get("model_revision")
            == "99ccb2737be22b8bb50dcfcc39ad4d567fb90cfd",
            "Source ASR model revision differs")

    by_id = {result["id"]: result for result in source["results"]}
    selected_ids = (*KNOWN_GOOD_IDS, KNOWN_BAD_ID)
    require(len(set(selected_ids)) == 20, "Expected 20 unique endpoint cases")
    require(set(selected_ids) <= set(by_id), "Endpoint corpus is incomplete")
    results = []
    for item_id in selected_ids:
        source_result = by_id[item_id]
        require(file_sha256(source_result["path"]) == source_result["sha256"],
                f"WAV hash differs: {source_result['path']}")
        results.append({
            "id": item_id,
            "label": "known_bad_trailing_truncation"
            if item_id == KNOWN_BAD_ID else "known_good_content_complete",
            "path": source_result["path"],
            "wav_sha256": source_result["sha256"],
            "intended_text_sha256": source_result["intended"][
                "normalized_text_sha256"
            ],
            "tail": inspect_tail(source_result),
        })

    good = [item for item in results
            if item["label"] == "known_good_content_complete"]
    bad = next(item for item in results
               if item["label"] == "known_bad_trailing_truncation")
    false_positives = [item["id"] for item in good if item["tail"]["flagged"]]
    summary = {
        "known_good_total": len(good),
        "known_good_false_positive_count": len(false_positives),
        "known_good_false_positive_ids": false_positives,
        "known_bad_flagged": bad["tail"]["flagged"],
        "feasibility_passed": not false_positives and bad["tail"]["flagged"],
    }
    run = output_parent / (
        "mandarin_asr_endpoint_feasibility_"
        + datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f")
    )
    run.mkdir(parents=True)
    report = {
        "schema_version": 1,
        "experiment_id": "mandarin_asr_endpoint_completion_feasibility",
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": file_sha256(Path(__file__)),
        "source_asr_report": str(SOURCE_REPORT.resolve()),
        "source_asr_report_sha256": SOURCE_REPORT_SHA256,
        "asr_provenance": {
            "model_id": source["model_id"],
            "model_revision": source["model_revision"],
            "decoder": source["inference_contract"],
            "intended_text_used_during_inference": False,
            "unk_preserved": True,
        },
        "global_omission_rule_changed": False,
        "endpoint_rule_frozen_before_additional_good_control_evaluation": True,
        "endpoint_rule": {
            "tail_expected_han_count": TAIL_EXPECTED_HAN_COUNT,
            "maximum_terminal_margin_seconds": MAX_TERMINAL_MARGIN_SECONDS,
            "requirements": [
                "last ASR token aligns within the final expected Han window",
                "tail has full expected coverage or a 1-3 character deleted suffix",
                "last ASR token ends no more than 200 ms before WAV end",
            ],
        },
        "results": results,
        "summary": summary,
    }
    report_path = run / "report.json"
    save_json(report_path, report)
    print(json.dumps(summary, indent=2))
    print(f"Completed: {report_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-parent", type=Path,
        default=ROOT / "outputs/evaluation",
    )
    args = parser.parse_args()
    return execute(args.output_parent.expanduser().resolve())


if __name__ == "__main__":
    raise SystemExit(main())
