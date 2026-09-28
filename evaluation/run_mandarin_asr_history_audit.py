"""Audit the fixed independent-ASR omission rule on existing CosyVoice history."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.run_mandarin_asr_unit09_feasibility import (
    DELETION_FLAG_THRESHOLD,
    MODEL_ID,
    MODEL_REVISION,
    align_result,
    file_sha256,
    infer_audio_only,
    intended_han,
    load_asr_once,
    require,
    save_json,
    text_sha256,
)


OUTPUT_ROOT = ROOT / "outputs/evaluation"
MAX_WHOLE_WAV_SECONDS = 30.0
COLLECTION_PATTERNS = (
    "cosyvoice_baseline_*/manifest.json",
    "cosyvoice_clean12_*/manifest.json",
    "cosyvoice_cr_format_ab_*/manifest.json",
    "cosyvoice_multiunit_path_ab_*/manifest.json",
    "cosyvoice_prompt_conditioning_ab_*/manifest.json",
    "cosyvoice_rng_sweep_*/manifest.json",
)


def record_audio_path(manifest_path, record):
    relative = record.get("output_path") or record.get("output_filename")
    return manifest_path.parent / relative if isinstance(relative, str) else None


def record_duration(record):
    audio = record.get("audio")
    if isinstance(audio, dict) and isinstance(audio.get("duration_seconds"), (int, float)):
        return float(audio["duration_seconds"])
    value = record.get("duration_seconds")
    return float(value) if isinstance(value, (int, float)) else None


def intended_locator(manifest, record, record_index):
    if isinstance(record.get("normalized_text"), str):
        return {
            "kind": "record_field",
            "record_index": record_index,
            "text_field": "normalized_text",
            "hash_field": "normalized_text_sha256",
        }
    if isinstance(record.get("text"), str):
        return {
            "kind": "record_field",
            "record_index": record_index,
            "text_field": "text",
            "hash_field": "text_sha256",
        }
    condition = record.get("condition")
    unit_index = record.get("unit_index")
    condition_data = manifest.get("conditions", {}).get(condition, {})
    units = condition_data.get("units")
    if (isinstance(units, list) and isinstance(unit_index, int)
            and 1 <= unit_index <= len(units)
            and isinstance(units[unit_index - 1].get("text"), str)):
        return {
            "kind": "condition_unit",
            "record_index": record_index,
            "condition": condition,
            "unit_index": unit_index,
        }
    return None


def manifest_paths():
    paths = set()
    for pattern in COLLECTION_PATTERNS:
        paths.update(OUTPUT_ROOT.glob(pattern))
    return sorted(paths)


def discover_audio_only_corpus():
    """Discover hashed WAV records without placing intended text in corpus items."""
    require(DELETION_FLAG_THRESHOLD == 4, "The predefined deletion threshold changed")
    mappings = []
    exclusions = Counter()
    collection_counts = Counter()
    for manifest_path in manifest_paths():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest.get("records")
        if not isinstance(records, list):
            exclusions["no_records"] += 1
            continue
        manifest_hash = file_sha256(manifest_path)
        for record_index, record in enumerate(records):
            if not isinstance(record, dict):
                exclusions["invalid_record"] += 1
                continue
            if record.get("status") not in (None, "passed_wav_check"):
                exclusions["unsuccessful_record"] += 1
                continue
            locator = intended_locator(manifest, record, record_index)
            audio_path = record_audio_path(manifest_path, record)
            wav_hash = record.get("wav_sha256")
            duration = record_duration(record)
            if locator is None or audio_path is None or not isinstance(wav_hash, str):
                exclusions["incomplete_mapping"] += 1
                continue
            if duration is None or duration <= 0:
                exclusions["invalid_duration"] += 1
                continue
            if duration > MAX_WHOLE_WAV_SECONDS:
                exclusions["over_30_seconds"] += 1
                continue
            require(audio_path.is_file(), f"Mapped WAV is missing: {audio_path}")
            require(file_sha256(audio_path) == wav_hash,
                    f"Mapped WAV hash differs: {audio_path}")
            collection = manifest_path.parent.name
            collection_counts[collection] += 1
            mappings.append({
                "mapping_id": f"{collection}:record_{record_index + 1:03d}",
                "collection": collection,
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": manifest_hash,
                "locator": locator,
                "path": str(audio_path.resolve()),
                "sha256": wav_hash,
                "duration_seconds": duration,
            })

    by_hash = {}
    for mapping in mappings:
        if mapping["sha256"] not in by_hash:
            by_hash[mapping["sha256"]] = {
                **mapping,
                "id": f"history_{len(by_hash) + 1:03d}",
                "aliases": [],
            }
        else:
            by_hash[mapping["sha256"]]["aliases"].append({
                key: mapping[key]
                for key in ("mapping_id", "collection", "manifest_path",
                            "manifest_sha256", "locator", "path")
            })
    corpus = list(by_hash.values())
    require(len(mappings) == 82, "Expected 82 eligible record mappings")
    require(len(corpus) == 58, "Expected 58 unique eligible WAVs")
    return corpus, {
        "manifest_count": len(manifest_paths()),
        "trustworthy_record_mappings_before_deduplication": len(mappings),
        "unique_wavs_after_sha256_deduplication": len(corpus),
        "collection_record_counts_before_deduplication": dict(collection_counts),
        "exclusions": dict(exclusions),
        "whole_wav_duration_limit_seconds": MAX_WHOLE_WAV_SECONDS,
        "duration_limit_reason": (
            "Preserve the established one-forward-pass whole-WAV decoder; longer "
            "records would require unvalidated ASR chunking or excessive attention memory"
        ),
    }


def load_locator_text(mapping):
    manifest_path = Path(mapping["manifest_path"])
    require(file_sha256(manifest_path) == mapping["manifest_sha256"],
            f"Manifest changed after corpus discovery: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    locator = mapping["locator"]
    record = manifest["records"][locator["record_index"]]
    if locator["kind"] == "record_field":
        text = record[locator["text_field"]]
        expected_hash = record.get(locator["hash_field"])
    else:
        unit = manifest["conditions"][locator["condition"]]["units"][
            locator["unit_index"] - 1
        ]
        text = unit["text"]
        expected_hash = unit.get("text_sha256")
    actual_hash = text_sha256(text)
    if expected_hash is not None:
        require(actual_hash == expected_hash,
                f"Intended text hash differs: {mapping['mapping_id']}")
    return text, actual_hash


def load_intended_after_all_inference(results):
    """Resolve and verify every intended transcript after all ASR calls finish."""
    intended = {}
    for result in results:
        mappings = [result] + result["aliases"]
        texts = [load_locator_text(mapping) for mapping in mappings]
        han_texts = {"".join(intended_han(text)) for text, _ in texts}
        require(len(han_texts) == 1,
                f"Duplicate WAV maps to different Han text: {result['id']}")
        text, digest = texts[0]
        tokens = intended_han(text)
        require(tokens, f"No intended Han text: {result['id']}")
        intended[result["id"]] = {
            "normalized_text": text,
            "normalized_text_sha256": digest,
            "comparison_normalization": (
                "Unicode NFC; retain Han characters only; preserve order"
            ),
            "comparison_tokens": tokens,
            "comparison_text": "".join(tokens),
            "comparison_length": len(tokens),
            "target_phrase": "",
            "target_start": 0,
            "target_end": 0,
            "alias_text_sha256s": sorted({value[1] for value in texts}),
        }
    return intended


def maximum_deletion_span(alignment):
    return max((group["expected_end"] - group["expected_start"]
                for group in alignment["edit_groups"]
                if group["operation"] == "deletion"), default=0)


def summarize(results):
    distribution = Counter()
    candidates = []
    short_deletion_units = []
    for result in results:
        alignment = result["alignment"]
        maximum = maximum_deletion_span(alignment)
        distribution[maximum] += 1
        deletions = [group for group in alignment["edit_groups"]
                     if group["operation"] == "deletion"]
        if alignment["flagged"]:
            candidates.append({
                "id": result["id"],
                "path": result["path"],
                "wav_sha256": result["sha256"],
                "collection": result["collection"],
                "cer": alignment["cer"],
                "maximum_deletion_span": maximum,
                "flagged_deletion_groups": alignment["flagged_deletion_groups"],
            })
        elif maximum:
            short_deletion_units.append({
                "id": result["id"],
                "path": result["path"],
                "wav_sha256": result["sha256"],
                "maximum_deletion_span": maximum,
                "deletion_groups": deletions,
            })
    return {
        "total_unique_wavs_checked": len(results),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "units_with_only_1_to_3_character_deletions": len(short_deletion_units),
        "short_deletion_units": short_deletion_units,
        "maximum_deletion_span_distribution": {
            str(length): distribution[length] for length in sorted(distribution)
        },
        "rule": f"contiguous expected deletion length >= {DELETION_FLAG_THRESHOLD}",
        "classification_note": (
            "Flags are listening candidates, not automatic TTS-omission findings"
        ),
    }


def execute(output_parent):
    corpus, selection_audit = discover_audio_only_corpus()
    run = output_parent / (
        "mandarin_asr_history_audit_"
        + datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f")
    )
    run.mkdir(parents=True)
    report_path = run / "report.json"
    started = time.perf_counter()
    report = {
        "schema_version": 1,
        "experiment_id": "mandarin_asr_existing_history_audit",
        "status": "initializing",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": file_sha256(Path(__file__)),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "inference_contract": (
            "Independent whole-WAV audio-only greedy CTC; inference function does "
            "not accept intended text; all intended text resolved after every decode"
        ),
        "intended_text_used_during_inference": False,
        "threshold_fixed_before_inference": True,
        "deletion_flag_threshold": DELETION_FLAG_THRESHOLD,
        "selection_audit": selection_audit,
        "corpus": corpus,
        "results": [],
    }
    save_json(report_path, report)
    try:
        model_started = time.perf_counter()
        torch, tokenizer, feature_extractor, model, model_metadata = load_asr_once()
        report["model_load_seconds"] = time.perf_counter() - model_started
        report["model"] = model_metadata
        report["status"] = "transcribing_without_intended_text"
        save_json(report_path, report)
        results = infer_audio_only(
            corpus, torch, tokenizer, feature_extractor, model
        )
        report["all_audio_inference_completed_before_intended_text_load"] = True
        intended = load_intended_after_all_inference(results)
        report["status"] = "aligning_after_inference"
        for result in results:
            result["intended"] = intended[result["id"]]
            align_result(result, intended[result["id"]])
            result["alignment"]["maximum_deletion_span"] = maximum_deletion_span(
                result["alignment"]
            )
        report["results"] = results
        report["summary"] = summarize(results)
        report["runtime"] = {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
        }
        report["status"] = "completed"
        report["total_wall_seconds"] = time.perf_counter() - started
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(report_path, report)
        print(f"Completed: {report_path}", flush=True)
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(error)),
        }
        report["total_wall_seconds"] = time.perf_counter() - started
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(report_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--output-parent", type=Path, default=OUTPUT_ROOT,
    )
    args = parser.parse_args()
    corpus, selection_audit = discover_audio_only_corpus()
    if not args.execute:
        print(json.dumps({
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "device": "cuda",
            "decoder": "independent whole-WAV greedy CTC",
            "deletion_flag_threshold": DELETION_FLAG_THRESHOLD,
            "corpus_count": len(corpus),
            "selection_audit": selection_audit,
        }, ensure_ascii=False, indent=2))
        return 0
    return execute(args.output_parent.expanduser().resolve())


if __name__ == "__main__":
    raise SystemExit(main())
