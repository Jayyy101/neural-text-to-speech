"""Independent greedy-CTC completeness feasibility test for clean12 unit 09."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.align_mandarin_ctc import ANALYSIS_RATE, read_analysis_audio


MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
MODEL_REVISION = "99ccb2737be22b8bb50dcfcc39ad4d567fb90cfd"
DELETION_FLAG_THRESHOLD = 4
TARGET_PHRASE = "而上品神通骨"
CLEAN12 = ROOT / "outputs/evaluation/cosyvoice_clean12_2026-09-20_08-36-03_246121/manifest.json"
RETRIES = ROOT / "outputs/evaluation/cosyvoice_clean12_retries_2026-09-20_09-07-08_605116/manifest.json"
CORPUS = (
    {
        "id": "bad_original",
        "expected_label": "known_bad",
        "path": CLEAN12.parent / "unit_09.wav",
        "sha256": "3efbd2062e78ff0937debc5803d856a0193d4a9ac421f90f05651e36e6354b97",
    },
    {
        "id": "correct_seed_2026091705",
        "expected_label": "known_correct",
        "path": RETRIES.parent / "unit_09_seed_2026091705.wav",
        "sha256": "0907a2c2e0b09f8499116396269f55fd0c18596f0282d792472d5d9cec9712ae",
    },
    {
        "id": "correct_seed_2026091706",
        "expected_label": "known_correct",
        "path": RETRIES.parent / "unit_09_seed_2026091706.wav",
        "sha256": "2da53582d232d8124b17c192a3fb1a1d7c884fb46f20008c1e7e424edf667bcf",
    },
)


def require(value, message):
    if not value:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def is_han(character):
    name = unicodedata.name(character, "")
    return (name.startswith("CJK UNIFIED IDEOGRAPH-")
            or name.startswith("CJK COMPATIBILITY IDEOGRAPH-")
            or character == "〇")


def intended_han(text):
    return [character for character in unicodedata.normalize("NFC", text) if is_han(character)]


def validate_corpus():
    result = []
    for item in CORPUS:
        require(item["path"].is_file(), f"Missing fixed corpus WAV: {item['path']}")
        require(file_sha256(item["path"]) == item["sha256"], f"Fixed WAV hash differs: {item['id']}")
        result.append({**item, "path": str(item["path"].resolve())})
    return result


def load_intended_after_inference():
    clean12 = json.loads(CLEAN12.read_text(encoding="utf-8"))
    retries = json.loads(RETRIES.read_text(encoding="utf-8"))
    require(clean12["status"] == retries["status"] == "completed", "Source run is incomplete")
    original = next(record for record in clean12["records"] if record["unit_index"] == 9)
    retry_records = [record for record in retries["records"] if record["unit"] == 9]
    require(len(retry_records) == 2, "Expected exactly two unit 09 retries")
    texts = [original["normalized_text"]] + [record["text"] for record in retry_records]
    hashes = [original["text_sha256"]] + [record["text_sha256"] for record in retry_records]
    require(len(set(texts)) == len(set(hashes)) == 1, "Unit 09 intended text differs across runs")
    text = texts[0]
    require(text_sha256(text) == hashes[0], "Intended text hash differs")
    tokens = intended_han(text)
    phrase_tokens = list(TARGET_PHRASE)
    start = "".join(tokens).index(TARGET_PHRASE)
    require(start == 91 and len(tokens) == 107, "Known intended comparison span changed")
    return {
        "normalized_text": text,
        "normalized_text_sha256": hashes[0],
        "comparison_normalization": "Unicode NFC; retain Han characters only; preserve order",
        "comparison_tokens": tokens,
        "comparison_text": "".join(tokens),
        "comparison_length": len(tokens),
        "target_phrase": TARGET_PHRASE,
        "target_start": start,
        "target_end": start + len(phrase_tokens),
    }


def load_asr_once():
    import torch
    from transformers import (
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2ForCTC,
    )

    require(torch.cuda.is_available(), "CUDA is unavailable")
    common = {"revision": MODEL_REVISION, "local_files_only": True}
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(MODEL_ID, **common)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, **common)
    model, loading = Wav2Vec2ForCTC.from_pretrained(
        MODEL_ID, output_loading_info=True, **common
    )
    require(not loading.get("missing_keys") and not loading.get("mismatched_keys"),
            "ASR checkpoint has missing or mismatched weights")
    require(feature_extractor.sampling_rate == ANALYSIS_RATE,
            "ASR feature extractor does not expect 16 kHz")
    require(model.config.pad_token_id == tokenizer.pad_token_id,
            "Model/tokenizer blank IDs differ")
    require(model.config.vocab_size == len(tokenizer.get_vocab()),
            "Model/tokenizer vocabulary sizes differ")
    require(getattr(model.config, "_commit_hash", None) == MODEL_REVISION,
            "Resolved ASR revision differs")
    model.to("cuda").eval()
    return torch, tokenizer, feature_extractor, model, {
        "model_id": MODEL_ID,
        "requested_revision": MODEL_REVISION,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "feature_extractor_class": type(feature_extractor).__name__,
        "vocabulary_size": model.config.vocab_size,
        "blank_token_id": model.config.pad_token_id,
        "unk_token": tokenizer.unk_token,
        "unk_token_id": tokenizer.unk_token_id,
        "word_delimiter_token": tokenizer.word_delimiter_token,
        "word_delimiter_token_id": tokenizer.word_delimiter_token_id,
        "device": "cuda",
        "device_name": torch.cuda.get_device_name(0),
        "loading_info": {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in loading.items()
        },
    }


def collapse_greedy_ids(frame_ids, blank_id):
    emitted = []
    start = 0
    for index in range(1, len(frame_ids) + 1):
        if index == len(frame_ids) or frame_ids[index] != frame_ids[start]:
            token_id = frame_ids[start]
            if token_id != blank_id:
                emitted.append({"token_id": token_id, "start_frame": start, "end_frame": index})
            start = index
    return emitted


def normalize_recognized(emitted, tokenizer, frame_seconds):
    raw_parts = []
    comparison = []
    ignored = []
    for item in emitted:
        token = tokenizer.convert_ids_to_tokens(item["token_id"])
        token_record = {
            **item,
            "token": token,
            "start_seconds": item["start_frame"] * frame_seconds,
            "end_seconds": item["end_frame"] * frame_seconds,
        }
        if token == tokenizer.word_delimiter_token:
            raw_parts.append(" ")
            ignored.append({**token_record, "reason": "ctc_word_delimiter"})
        elif token == tokenizer.unk_token:
            raw_parts.append(tokenizer.unk_token)
            comparison.append({**token_record, "comparison_token": tokenizer.unk_token})
        elif token in tokenizer.all_special_tokens:
            raw_parts.append(token)
            ignored.append({**token_record, "reason": "non_unk_special_token"})
        else:
            raw_parts.append(token)
            kept = [character for character in unicodedata.normalize("NFC", token) if is_han(character)]
            if kept:
                for character in kept:
                    comparison.append({**token_record, "comparison_token": character})
            else:
                ignored.append({**token_record, "reason": "non_han_token"})
    return "".join(raw_parts).strip(), comparison, ignored


def infer_audio_only(corpus, torch, tokenizer, feature_extractor, model):
    """No intended text or tokens are accepted by this audio-only inference path."""
    results = []
    stride = math.prod(model.config.conv_stride)
    frame_seconds = stride / ANALYSIS_RATE
    for item in corpus:
        samples, audio = read_analysis_audio(item["path"])
        inputs = feature_extractor(samples, sampling_rate=ANALYSIS_RATE, return_tensors="pt")
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            logits = model(**inputs).logits[0]
        frame_ids = logits.argmax(dim=-1).cpu().tolist()
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - started
        emitted = collapse_greedy_ids(frame_ids, model.config.pad_token_id)
        raw_transcript, comparison, ignored = normalize_recognized(
            emitted, tokenizer, frame_seconds
        )
        results.append({
            **item,
            "audio": audio,
            "inference_seconds": inference_seconds,
            "emission_frames": len(frame_ids),
            "ctc_frame_seconds": frame_seconds,
            "raw_transcript": raw_transcript,
            "raw_emitted_tokens": [
                {**entry, "token": tokenizer.convert_ids_to_tokens(entry["token_id"])}
                for entry in emitted
            ],
            "comparison_tokens": comparison,
            "comparison_text": "".join(entry["comparison_token"] for entry in comparison),
            "ignored_tokens": ignored,
        })
        print(f"Decoded {item['id']}: {audio['duration_seconds']:.2f}s audio, "
              f"{inference_seconds:.3f}s inference", flush=True)
    return results


def levenshtein_steps(expected, recognized):
    n, m = len(expected), len(recognized)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    choices = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        costs[i][0] = i
        choices[i][0] = "deletion"
    for j in range(1, m + 1):
        costs[0][j] = j
        choices[0][j] = "insertion"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if expected[i - 1] == recognized[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
                choices[i][j] = "match"
            else:
                # Stable tie order: substitution, deletion, insertion.
                candidates = (
                    (costs[i - 1][j - 1] + 1, 0, "substitution"),
                    (costs[i - 1][j] + 1, 1, "deletion"),
                    (costs[i][j - 1] + 1, 2, "insertion"),
                )
                cost, _, operation = min(candidates)
                costs[i][j] = cost
                choices[i][j] = operation
    steps = []
    i, j = n, m
    while i or j:
        operation = choices[i][j]
        if operation in ("match", "substitution"):
            steps.append({"operation": operation, "expected_index": i - 1,
                          "recognized_index": j - 1, "expected": expected[i - 1],
                          "recognized": recognized[j - 1]})
            i -= 1
            j -= 1
        elif operation == "deletion":
            steps.append({"operation": operation, "expected_index": i - 1,
                          "recognized_index": None, "recognized_position": j,
                          "expected": expected[i - 1], "recognized": None})
            i -= 1
        elif operation == "insertion":
            steps.append({"operation": operation, "expected_index": None,
                          "expected_position": i, "recognized_index": j - 1,
                          "expected": None, "recognized": recognized[j - 1]})
            j -= 1
        else:
            raise RuntimeError("Levenshtein traceback is incomplete")
    steps.reverse()
    return steps, costs[n][m]


def token_time(comparison_tokens, index):
    if index is None or not 0 <= index < len(comparison_tokens):
        return None
    token = comparison_tokens[index]
    return [token["start_seconds"], token["end_seconds"]]


def group_edits(steps, expected, recognized_meta, target_start, target_end):
    groups = []
    current = None
    recognized = [item["comparison_token"] for item in recognized_meta]
    for step in steps:
        if step["operation"] == "match":
            if current:
                groups.append(current)
                current = None
            continue
        if current is None or current["operation"] != step["operation"]:
            if current:
                groups.append(current)
            current = {"operation": step["operation"], "steps": []}
        current["steps"].append(step)
    if current:
        groups.append(current)

    records = []
    for group_index, group in enumerate(groups, 1):
        expected_indices = [s["expected_index"] for s in group["steps"]
                            if s["expected_index"] is not None]
        recognized_indices = [s["recognized_index"] for s in group["steps"]
                              if s["recognized_index"] is not None]
        if expected_indices:
            expected_start = min(expected_indices)
            expected_end = max(expected_indices) + 1
        else:
            expected_start = expected_end = group["steps"][0]["expected_position"]
        if recognized_indices:
            recognized_start = min(recognized_indices)
            recognized_end = max(recognized_indices) + 1
        else:
            recognized_start = recognized_end = group["steps"][0]["recognized_position"]
        overlap = max(0, min(expected_end, target_end) - max(expected_start, target_start))
        left_time = token_time(recognized_meta, recognized_start - 1)
        right_time = token_time(recognized_meta, recognized_start)
        records.append({
            "group_index": group_index,
            "operation": group["operation"],
            "expected_start": expected_start,
            "expected_end": expected_end,
            "expected_text": "".join(expected[index] for index in expected_indices),
            "recognized_start": recognized_start,
            "recognized_end": recognized_end,
            "recognized_text": "".join(recognized[index] for index in recognized_indices),
            "expected_context": "".join(expected[max(0, expected_start - 8):
                                                   min(len(expected), expected_end + 8)]),
            "recognized_context": "".join(recognized[max(0, recognized_start - 8):
                                                       min(len(recognized), recognized_end + 8)]),
            "recognized_time_span_seconds": (
                [recognized_meta[recognized_start]["start_seconds"],
                 recognized_meta[recognized_end - 1]["end_seconds"]]
                if recognized_indices else None
            ),
            "deletion_neighbor_times_seconds": (
                {"left": left_time, "right": right_time}
                if group["operation"] == "deletion" else None
            ),
            "target_overlap_characters": overlap,
        })
    return records


def align_result(result, intended):
    expected = intended["comparison_tokens"]
    recognized_meta = result["comparison_tokens"]
    recognized = [item["comparison_token"] for item in recognized_meta]
    steps, distance = levenshtein_steps(expected, recognized)
    counts = {
        operation: sum(step["operation"] == operation for step in steps)
        for operation in ("match", "deletion", "insertion", "substitution")
    }
    groups = group_edits(
        steps, expected, recognized_meta,
        intended["target_start"], intended["target_end"],
    )
    flagged_groups = [group for group in groups
                      if group["operation"] == "deletion"
                      and group["expected_end"] - group["expected_start"] >= DELETION_FLAG_THRESHOLD]
    target_steps = [step for step in steps
                    if step["expected_index"] is not None
                    and intended["target_start"] <= step["expected_index"] < intended["target_end"]]
    target_alignment = [{
        "expected_index": step["expected_index"],
        "expected": step["expected"],
        "operation": step["operation"],
        "recognized": step["recognized"],
        "recognized_index": step["recognized_index"],
        "recognized_time_seconds": token_time(recognized_meta, step["recognized_index"]),
    } for step in target_steps]
    result["alignment"] = {
        "algorithm": "unit-cost Levenshtein; deterministic traceback tie order substitution, deletion, insertion",
        "expected_length": len(expected),
        "recognized_length": len(recognized),
        "edit_distance": distance,
        "cer": distance / len(expected),
        "counts": counts,
        "edit_groups": groups,
        "flag_rule": f"contiguous expected deletion length >= {DELETION_FLAG_THRESHOLD}",
        "flagged": bool(flagged_groups),
        "flagged_deletion_groups": flagged_groups,
        "target_phrase": intended["target_phrase"],
        "target_start": intended["target_start"],
        "target_end": intended["target_end"],
        "target_alignment": target_alignment,
        "target_deleted_characters": sum(step["operation"] == "deletion" for step in target_steps),
        "maximum_flagged_target_overlap": max(
            (group["target_overlap_characters"] for group in flagged_groups), default=0
        ),
    }
    return result


def execute(output_parent):
    corpus = validate_corpus()
    run = output_parent / (
        "mandarin_asr_unit09_feasibility_"
        + datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f")
    )
    run.mkdir(parents=True)
    report_path = run / "report.json"
    started = time.perf_counter()
    report = {
        "schema_version": 1,
        "experiment_id": "mandarin_asr_unit09_feasibility",
        "status": "initializing",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": file_sha256(Path(__file__)),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "inference_contract": "Independent audio-only greedy CTC; intended transcript is not accepted by inference function",
        "threshold_fixed_before_inference": True,
        "deletion_flag_threshold": DELETION_FLAG_THRESHOLD,
        "corpus": corpus,
        "results": [],
    }
    save_json(report_path, report)
    print("Output:", run, flush=True)
    try:
        load_started = time.perf_counter()
        torch, tokenizer, feature_extractor, model, model_metadata = load_asr_once()
        report["model_load_seconds"] = time.perf_counter() - load_started
        report["model"] = model_metadata
        report["runtime"] = {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
        }
        report["versions"] = {}
        for package in ("torch", "numpy", "scipy", "soundfile", "transformers"):
            try:
                report["versions"][package] = version(package)
            except PackageNotFoundError:
                report["versions"][package] = None
        report["status"] = "transcribing"
        save_json(report_path, report)

        # This phase is deliberately completed before intended text is loaded.
        transcription_results = infer_audio_only(
            corpus, torch, tokenizer, feature_extractor, model
        )
        intended = load_intended_after_inference()
        vocabulary = tokenizer.get_vocab()
        intended["characters_missing_from_asr_vocabulary"] = sorted(
            {character for character in intended["comparison_tokens"] if character not in vocabulary}
        )
        require(all(character in vocabulary for character in TARGET_PHRASE),
                "Target phrase contains an out-of-vocabulary character")
        report["intended"] = intended
        report["results"] = [align_result(result, intended) for result in transcription_results]

        bad = next(result for result in report["results"] if result["expected_label"] == "known_bad")
        correct = [result for result in report["results"] if result["expected_label"] == "known_correct"]
        success = (
            bad["alignment"]["flagged"]
            and bad["alignment"]["maximum_flagged_target_overlap"] >= DELETION_FLAG_THRESHOLD
            and all(not result["alignment"]["flagged"] for result in correct)
        )
        report["success_criteria"] = {
            "bad_original_flagged": bad["alignment"]["flagged"],
            "bad_flagged_group_overlaps_target_by_at_least_threshold": (
                bad["alignment"]["maximum_flagged_target_overlap"] >= DELETION_FLAG_THRESHOLD
            ),
            "both_correct_retries_not_flagged": all(
                not result["alignment"]["flagged"] for result in correct
            ),
            "experiment_success": success,
        }
        report["conclusion"] = (
            "feasibility_passed" if success else "feasibility_failed"
        )
        report.update(
            status="completed",
            total_wall_seconds=time.perf_counter() - started,
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        save_json(report_path, report)
        print("Completed:", report_path, flush=True)
        return 0
    except Exception as error:
        report.update(
            status="failed",
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            error={"type": type(error).__name__, "message": str(error),
                   "traceback": traceback.format_exc()},
        )
        save_json(report_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-parent", type=Path, default=ROOT / "outputs/evaluation")
    args = parser.parse_args()
    corpus = validate_corpus()
    print(json.dumps({
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "device": "cuda",
        "decoder": "independent greedy CTC",
        "intended_text_used_during_inference": False,
        "deletion_flag_threshold": DELETION_FLAG_THRESHOLD,
        "corpus": corpus,
    }, ensure_ascii=True, indent=2))
    if not args.execute:
        return 0
    return execute(args.output_parent.expanduser().resolve())


if __name__ == "__main__":
    raise SystemExit(main())
