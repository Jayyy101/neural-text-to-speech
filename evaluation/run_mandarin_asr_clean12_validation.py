"""False-positive validation of the fixed independent-ASR omission rule."""

import argparse
from datetime import datetime, timezone
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


CLEAN12 = ROOT / "outputs/evaluation/cosyvoice_clean12_2026-09-20_08-36-03_246121"
RETRIES = ROOT / "outputs/evaluation/cosyvoice_clean12_retries_2026-09-20_09-07-08_605116"

CLEAN12_WAV_HASHES = (
    "4844da5fe47416bbc886e8c5058e03ae11443bc4e14bac7cf3d974278220867c",
    "b385ad20373ec69f4ca97a369d860f3d511d7556391a71328ede11efa24d407a",
    "b2d602306f8d9315db85243cf5f0945ef9319f0fe0ebe4a229b0f2b2cd48fe8b",
    "44b259f464d52e6db29a34e48af59e70e4a84171e246ce181d1f9a0e1a13b85a",
    "594b749576799d12eae786db98e50538856f29838da51ee668c508eb8515aac3",
    "4e2850ed1daa6a7fad83375c37370c478c1a416654cae5d7605cf45ca435b830",
    "4e5a36a9da94e1a57c8d37086f5c84c5323881a506e88bd3aa8894f0c5bc655a",
    "3ef2c2ae70cfc06925456eb1475cb0b23b61a72e5d2cb3be30d9b11a49ebcef8",
    "3efbd2062e78ff0937debc5803d856a0193d4a9ac421f90f05651e36e6354b97",
    "68545b8d51c1d9c0e6594ac2b271ba7a272e960ad0ca4361a1bc10dc60cdd3b2",
    "0f729e52120a0e4c202681a895416891e894185b604bcb133360985b7d5d10af",
    "a646e0677b5290de16a59efc4ab6fdb5e84827556e82cfb89560e398445f3cf5",
)

RETRY_WAVS = (
    (2026091705, "0907a2c2e0b09f8499116396269f55fd0c18596f0282d792472d5d9cec9712ae"),
    (2026091706, "2da53582d232d8124b17c192a3fb1a1d7c884fb46f20008c1e7e424edf667bcf"),
)


def fixed_corpus():
    corpus = []
    for unit, digest in enumerate(CLEAN12_WAV_HASHES, 1):
        corpus.append({
            "id": f"clean12_unit_{unit:02d}",
            "unit": unit,
            "source": "clean12",
            "expected_label": "known_bad" if unit == 9 else "known_good",
            "path": CLEAN12 / f"unit_{unit:02d}.wav",
            "sha256": digest,
        })
    for seed, digest in RETRY_WAVS:
        corpus.append({
            "id": f"unit_09_retry_seed_{seed}",
            "unit": 9,
            "seed": seed,
            "source": "unit09_retry",
            "expected_label": "known_good",
            "path": RETRIES / f"unit_09_seed_{seed}.wav",
            "sha256": digest,
        })
    return corpus


def validate_audio_corpus():
    require(DELETION_FLAG_THRESHOLD == 4, "The predefined deletion threshold changed")
    corpus = fixed_corpus()
    require(len(corpus) == 14, "Expected 14 fixed validation WAVs")
    require(sum(item["expected_label"] == "known_good" for item in corpus) == 13,
            "Expected 13 known-good WAVs")
    result = []
    for item in corpus:
        require(item["path"].is_file(), f"Missing fixed WAV: {item['path']}")
        require(file_sha256(item["path"]) == item["sha256"],
                f"Fixed WAV hash differs: {item['id']}")
        result.append({**item, "path": str(item["path"].resolve())})
    return result


def intended_record(text, expected_hash):
    require(text_sha256(text) == expected_hash, "Intended text hash differs")
    tokens = intended_han(text)
    require(tokens, "Intended Han comparison text is empty")
    return {
        "normalized_text": text,
        "normalized_text_sha256": expected_hash,
        "comparison_normalization": (
            "Unicode NFC; retain Han characters only; preserve order"
        ),
        "comparison_tokens": tokens,
        "comparison_text": "".join(tokens),
        "comparison_length": len(tokens),
        "target_phrase": "",
        "target_start": 0,
        "target_end": 0,
    }


def load_intended_after_all_inference(results):
    """Load intended text only after every independent ASR call has completed."""
    clean_manifest = json.loads(
        (CLEAN12 / "manifest.json").read_text(encoding="utf-8")
    )
    retry_manifest = json.loads(
        (RETRIES / "manifest.json").read_text(encoding="utf-8")
    )
    require(clean_manifest.get("status") == "completed", "Clean12 run is incomplete")
    require(retry_manifest.get("status") == "completed", "Retry run is incomplete")
    clean_records = clean_manifest.get("records")
    retry_records = retry_manifest.get("records")
    require(isinstance(clean_records, list) and len(clean_records) == 12,
            "Clean12 manifest must contain 12 records")
    require(isinstance(retry_records, list), "Retry manifest records are invalid")
    clean_by_unit = {record["unit_index"]: record for record in clean_records}
    retry_by_seed = {
        record["seed"]: record for record in retry_records if record.get("unit") == 9
    }
    require(set(clean_by_unit) == set(range(1, 13)), "Clean12 unit coverage differs")
    require(set(retry_by_seed) == {seed for seed, _ in RETRY_WAVS},
            "Unit 09 retry controls differ")

    intended = {}
    for result in results:
        if result["source"] == "clean12":
            record = clean_by_unit[result["unit"]]
            require(record["wav_sha256"] == result["sha256"],
                    f"Manifest WAV hash differs: {result['id']}")
            intended[result["id"]] = intended_record(
                record["normalized_text"], record["text_sha256"]
            )
        else:
            record = retry_by_seed[result["seed"]]
            require(record["wav_sha256"] == result["sha256"],
                    f"Retry manifest WAV hash differs: {result['id']}")
            intended[result["id"]] = intended_record(
                record["text"], record["text_sha256"]
            )
    require(
        intended["clean12_unit_09"]["normalized_text"]
        == intended["unit_09_retry_seed_2026091705"]["normalized_text"]
        == intended["unit_09_retry_seed_2026091706"]["normalized_text"],
        "Unit 09 intended texts differ",
    )
    return intended


def summarize(results):
    known_good = [item for item in results if item["expected_label"] == "known_good"]
    false_positives = [item["id"] for item in known_good if item["alignment"]["flagged"]]
    bad = next(item for item in results if item["id"] == "clean12_unit_09")
    retries = [item for item in results if item["source"] == "unit09_retry"]
    return {
        "known_good_total": len(known_good),
        "known_good_false_positive_count": len(false_positives),
        "known_good_false_positive_ids": false_positives,
        "known_good_pass_count": len(known_good) - len(false_positives),
        "known_bad_unit_09_flagged": bad["alignment"]["flagged"],
        "corrected_unit_09_retries_passed": all(
            not item["alignment"]["flagged"] for item in retries
        ),
        "fixed_rule_validation_passed": (
            not false_positives
            and bad["alignment"]["flagged"]
            and all(not item["alignment"]["flagged"] for item in retries)
        ),
    }


def execute(output_parent):
    corpus = validate_audio_corpus()
    run = output_parent / (
        "mandarin_asr_clean12_validation_"
        + datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f")
    )
    run.mkdir(parents=True)
    report_path = run / "report.json"
    started = time.perf_counter()
    report = {
        "schema_version": 1,
        "experiment_id": "mandarin_asr_clean12_false_positive_validation",
        "status": "initializing",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": file_sha256(Path(__file__)),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "inference_contract": (
            "All 14 WAVs decoded by independent audio-only greedy CTC before "
            "any intended transcript is loaded"
        ),
        "intended_text_used_during_inference": False,
        "threshold_fixed_before_inference": True,
        "deletion_flag_threshold": DELETION_FLAG_THRESHOLD,
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
        "--output-parent", type=Path,
        default=ROOT / "outputs/evaluation",
    )
    args = parser.parse_args()
    corpus = validate_audio_corpus()
    if not args.execute:
        print(json.dumps({
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "device": "cuda",
            "decoder": "independent greedy CTC",
            "intended_text_loaded_after_all_inference": True,
            "deletion_flag_threshold": DELETION_FLAG_THRESHOLD,
            "corpus_count": len(corpus),
            "known_good_count": sum(
                item["expected_label"] == "known_good" for item in corpus
            ),
            "corpus": corpus,
        }, ensure_ascii=False, indent=2))
        return 0
    return execute(args.output_parent.expanduser().resolve())


if __name__ == "__main__":
    raise SystemExit(main())
