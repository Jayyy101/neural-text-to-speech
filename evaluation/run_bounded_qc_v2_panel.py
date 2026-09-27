"""Evaluate the frozen QC panel without generating or modifying source audio."""

import argparse
import hashlib
import json
from pathlib import Path

from evaluation.bounded_qc_research import evaluate_candidates
from src.audiobook.content_qc import compare_recognition


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "evaluation/inputs/bounded_qc_v2_fixed_panel.json"
PANEL_SHA256 = "36f5bb9ab0e08f02bd6bad08883b1314d5790ce4a8aec8452afcc53b11db124e"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(case):
    source = json.loads((ROOT / case["source_record_path"]).read_text(encoding="utf-8"))
    if case["id"].startswith("history_") or case["id"].startswith("clean12_"):
        record = next(item for item in source["results"] if item["id"] == case["id"])
        return record["intended"]["normalized_text"], record
    unit_number = int(case["id"].split("_unit_")[1].split("_attempt_")[0])
    unit_id = f"scene_0001_unit_{unit_number:04d}"
    unit = next(unit for scene in source["scenes"]
                for unit in scene["synthesis_units"] if unit["id"] == unit_id)
    evidence = json.loads((ROOT / case["evidence_path"]).read_text(encoding="utf-8"))
    return unit["normalized_text"], evidence["recognition"]


def run(output):
    if _sha256(PANEL) != PANEL_SHA256:
        raise ValueError("Frozen panel hash differs.")
    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    if len(panel["cases"]) != panel["case_count"] or len({
        case["id"] for case in panel["cases"]
    }) != panel["case_count"]:
        raise ValueError("Frozen panel membership is invalid.")
    results = []
    for case in panel["cases"]:
        for path_key, digest_key in (("wav_path", "wav_sha256"),
                                     ("evidence_path", "evidence_sha256"),
                                     ("source_record_path", "source_record_sha256")):
            if _sha256(ROOT / case[path_key]) != case[digest_key]:
                raise ValueError(f"Frozen panel binding differs: {case['id']} {path_key}")
        normalized, recognition = _source(case)
        old = compare_recognition(normalized, recognition)
        evaluated = evaluate_candidates(normalized, recognition, ROOT / case["wav_path"])
        if old != evaluated["comparison"]:
            raise ValueError(f"Version 1 comparison changed: {case['id']}")
        decision = evaluated["decision"]
        outcome = ("rejected" if decision == "rejected" else
                   "abstained" if evaluated["abstentions"] else "passed")
        results.append({
            "id": case["id"], "role": case["role"],
            "panel_label": case["panel_label"], "existing_label": case["existing_label"],
            "wav_sha256": case["wav_sha256"],
            "v1_decision": old["decision"], "v2_decision": decision,
            "outcome": outcome,
            "rejection_reasons": evaluated["rejection_reasons"],
            "abstentions": evaluated["abstentions"],
            "measurements": evaluated["measurements"],
        })
    false_rejections = [r["id"] for r in results
                        if r["panel_label"] == "clean" and r["v2_decision"] == "rejected"]
    severe_missed = [r["id"] for r in results
                     if r["panel_label"] == "severe" and r["v2_decision"] != "rejected"]
    ambiguous_rejected = [r["id"] for r in results
                          if r["panel_label"] == "ambiguous" and r["v2_decision"] == "rejected"]
    report = {
        "panel_id": panel["panel_id"], "panel_sha256": PANEL_SHA256,
        "candidate_reasons_active": False,
        "summary": {"case_count": len(results),
                    "clean_false_rejections": false_rejections,
                    "severe_missed": severe_missed,
                    "ambiguous_rejected": ambiguous_rejected,
                    "promotion_gate_passed": not (false_rejections or severe_missed or ambiguous_rejected)},
        "results": results,
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/evaluation/bounded_qc_v2_panel_report.json")
    args = parser.parse_args()
    result = run(args.output.expanduser().resolve())
    for case in result["results"]:
        print(f"{case['id']}: {case['outcome']} "
              f"(label={case['panel_label']}, decision={case['v2_decision']})")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
