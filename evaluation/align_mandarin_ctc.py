"""Align a known Han transcript to a WAV; write JSON evidence, never audio.

Only Han characters are acoustic targets. Explicit punctuation and whitespace
are omitted from targets but retain their exact source offsets. No automatic
normalization, free transcription, pause insertion, or heuristic valley search.
Each target character is one ctc-segmentation segment, with blanks between
segments. Returned spans use the library's midpoint / 0.5 s boundary rules;
they are estimates, not phone boundaries. A zero estimated gap does not prove
absence of a pause. Raw CTC character anchors are also retained.
Model dependencies are lazy: --help and text mapping require only Python.
Running alignment may download the selected Hugging Face model/tokenizer.
"""

import argparse
from bisect import bisect_left
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import sys
import unicodedata


DEFAULT_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
ANALYSIS_RATE = 16000
PUNCTUATION = frozenset('。？！？，、：；“”‘’「」『』（）《》〈〉【】〔〕［］｛｝…—–-·.,?!:;"\'()[]{}')


def json_safe(value):
    """Recursively convert container metadata to deterministic JSON values."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [json_safe(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            ),
        )
    return value


def json_text(value):
    return json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def map_source(text):
    """Indices are zero-based Python Unicode code-point offsets, never bytes."""
    tokens, punctuation, failures = [], [], []
    source_to_token = [None] * len(text)
    for index, character in enumerate(text):
        name = unicodedata.name(character, "")
        if (name.startswith("CJK UNIFIED IDEOGRAPH-")
                or name.startswith("CJK COMPATIBILITY IDEOGRAPH-") or character == "〇"):
            source_to_token[index] = len(tokens)
            tokens.append({"token_index": len(tokens), "character": character, "source_index": index})
        elif character in PUNCTUATION:
            punctuation.append({"character": character, "source_index": index})
        elif not character.isspace():
            failures.append({"character": character, "source_index": index,
                             "codepoint": f"U+{ord(character):04X}",
                             "reason": "Only Han targets are supported; explicit spoken-form normalization is required."})
    indices = [token["source_index"] for token in tokens]
    for mark in punctuation:
        following = bisect_left(indices, mark["source_index"])
        mark["preceding_token_index"] = following - 1 if following else None
        mark["following_token_index"] = following if following < len(tokens) else None
    return {"exact_source_text": text, "source_index_unit": "Unicode code point (zero-based)",
            "acoustic_text": "".join(token["character"] for token in tokens),
            "tokens": tokens, "source_to_token": source_to_token,
            "punctuation": punctuation, "normalization_failures": failures,
            "unsupported_characters": []}


def validate_vocabulary(mapping, tokenizer):
    """Validate exact identities, not just tokenizer output (which may be <unk>)."""
    vocab = tokenizer.get_vocab()
    if (not vocab or any(type(i) is not int or i < 0 for i in vocab.values())
            or sorted(vocab.values()) != list(range(len(vocab)))):
        raise ValueError("Model vocabulary must have unique contiguous nonnegative IDs.")
    blank = tokenizer.pad_token_id
    if type(blank) is not int or blank not in vocab.values():
        raise ValueError("Tokenizer must expose its CTC blank as pad_token_id.")
    special_ids = set(tokenizer.all_special_ids)
    unsupported = []
    token_ids = []
    for token in mapping["tokens"]:
        character = token["character"]
        token_id = vocab.get(character)
        if token_id is None or token_id in special_ids or token_id == blank:
            unsupported.append({**token, "codepoint": f"U+{ord(character):04X}",
                                "reason": "Missing acoustic character in model vocabulary."})
        elif tokenizer.encode(character, add_special_tokens=False) != [token_id]:
            unsupported.append({**token, "reason": "Tokenizer does not preserve this character as one exact token."})
        else:
            token_ids.append(token_id)
    mapping["unsupported_characters"] = unsupported
    if unsupported:
        details = ", ".join(f"{t['character']} (source index {t['source_index']})" for t in unsupported)
        raise ValueError(f"Unsupported source characters: {details}. No characters were dropped or substituted.")
    for token, token_id in zip(mapping["tokens"], token_ids):
        token["token_id"] = token_id
    labels = [None] * len(vocab)
    for character, token_id in vocab.items():
        labels[token_id] = character
    metadata = {"class": type(tokenizer).__name__, "vocabulary_size": len(vocab),
                "vocabulary_sha256": hashlib.sha256(json_text(labels).encode("utf-8")).hexdigest(),
                "blank_token_id": blank, "unk_token_id": tokenizer.unk_token_id,
                "resolved_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")}
    return token_ids, labels, metadata


def read_analysis_audio(path, start_seconds=0.0, end_seconds=None):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    for name, value in (("start-seconds", start_seconds), ("end-seconds", end_seconds)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be a finite nonnegative number.")
    with sf.SoundFile(str(path), "r") as audio:
        if audio.format not in ("WAV", "WAVEX", "RF64"):
            raise ValueError("Input must be a WAV file.")
        if audio.channels not in (1, 2) or audio.frames == 0:
            raise ValueError("Input WAV must be nonempty mono or stereo.")
        rate, total, channels = audio.samplerate, audio.frames, audio.channels
        end = total / rate if end_seconds is None else end_seconds
        if not 0 <= start_seconds < end <= total / rate:
            raise ValueError("Require 0 <= start-seconds < end-seconds <= WAV duration.")
        first, last = round(start_seconds * rate), round(end * rate)
        if last <= first:
            raise ValueError("Selected span rounds to zero audio frames.")
        audio.seek(first)
        samples = audio.read(last - first, dtype="float32", always_2d=True)
        if len(samples) != last - first:
            raise ValueError("Input WAV has a truncated selected span.")
        subtype = audio.subtype
    if not np.isfinite(samples).all():
        raise ValueError("Input contains non-finite audio samples.")
    mono = samples.mean(axis=1, dtype=np.float64).astype(np.float32)
    if not np.any(mono):
        raise ValueError("Selected mono analysis signal is silent, possibly from stereo cancellation.")
    if rate != ANALYSIS_RATE:
        divisor = math.gcd(rate, ANALYSIS_RATE)
        mono = resample_poly(mono, ANALYSIS_RATE // divisor, rate // divisor).astype(np.float32)
    if len(mono) < 400:
        raise ValueError("Analysis span is too short for Wav2Vec2 (minimum 400 samples at 16 kHz).")
    metadata = {"original_sample_rate": rate, "analysis_sample_rate": ANALYSIS_RATE,
                "channels": channels, "subtype": subtype, "source_frames": total,
                "source_duration_seconds": total / rate,
                "requested_start_seconds": start_seconds, "requested_end_seconds": end_seconds,
                "start_frame": first, "end_frame_exclusive": last,
                "start_seconds": first / rate, "end_seconds": last / rate,
                "duration_seconds": (last - first) / rate, "analysis_samples": len(mono),
                "downmix": "arithmetic channel mean", "resampling": "scipy.signal.resample_poly"}
    return mono, metadata


def load_tokenizer(model_id, revision):
    from transformers import Wav2Vec2CTCTokenizer
    return Wav2Vec2CTCTokenizer.from_pretrained(model_id, revision=revision)


def infer_emissions(samples, model_id, revision, device, vocabulary_size, blank):
    """Real ML path. Called only after exact target vocabulary validation."""
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; select --device cpu explicitly.")
    frontend = Wav2Vec2FeatureExtractor.from_pretrained(model_id, revision=revision)
    if frontend.sampling_rate != ANALYSIS_RATE:
        raise ValueError("Model feature extractor must expect 16000 Hz.")
    model, loading = Wav2Vec2ForCTC.from_pretrained(model_id, revision=revision, output_loading_info=True)
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        raise ValueError("Checkpoint has missing/mismatched weights; refusing randomly initialized alignment weights.")
    config = model.config
    if config.vocab_size != vocabulary_size or config.pad_token_id != blank:
        raise ValueError("Model CTC output vocabulary/blank does not match the validated tokenizer.")
    if getattr(config, "add_adapter", False):
        raise ValueError("Adapter downsampling is not supported by this initial timing adapter.")
    stride = math.prod(config.conv_stride)
    expected_frames = len(samples)
    for kernel, step in zip(config.conv_kernel, config.conv_stride):
        expected_frames = (expected_frames - kernel) // step + 1
    inputs = frontend(samples, sampling_rate=ANALYSIS_RATE, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    model.to(device).eval()
    with torch.inference_mode():
        logits = model(**inputs).logits
        log_probs = logits[0].float().log_softmax(dim=-1).cpu().numpy()
    if len(log_probs) != expected_frames:
        raise ValueError("Unexpected emission frame count; cannot safely map model frames to seconds.")
    metadata = {"class": type(model).__name__, "resolved_revision": getattr(config, "_commit_hash", None),
                "config": config.to_dict(), "feature_extractor": frontend.to_dict(),
                "loading_info": loading, "ctc_frame_seconds": stride / ANALYSIS_RATE,
                "emission_frames": len(log_probs), "dtype": "float32",
                "frame_time_convention": "CTC frame index * convolution stride / 16000; no fitted time shift",
                "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
    return log_probs, stride / ANALYSIS_RATE, metadata


def align_emissions(log_probs, token_ids, labels, blank, frame_seconds):
    """No decoder: supplied exact token IDs constrain the CTC segmentation."""
    import numpy as np
    import ctc_segmentation as ctc

    lpz = np.asarray(log_probs, dtype=np.float32)
    if (lpz.ndim != 2 or lpz.shape[1] != len(labels) or not np.isfinite(lpz).all()
            or np.any(lpz > 1e-6)):
        raise ValueError("Expected finite CTC log probabilities with the validated vocabulary width.")
    if not token_ids or not math.isfinite(frame_seconds) or frame_seconds <= 0:
        raise ValueError("Alignment requires nonempty tokens and a finite positive frame duration.")
    config = ctc.CtcSegmentationParameters(char_list=labels, blank=blank, index_duration=frame_seconds)
    # One utterance per character exposes character spans via the public API.
    targets = [np.asarray([token_id], dtype=np.int64) for token_id in token_ids]
    ground_truth, begins = ctc.prepare_token_list(config, targets)
    if len(ground_truth) > len(lpz):
        raise ValueError("Too few CTC frames for the complete character/blank target sequence.")
    anchors, frame_scores, _ = ctc.ctc_segmentation(config, lpz, ground_truth)
    segments = ctc.determine_utterance_segments(config, begins, frame_scores, anchors, targets)
    return [{"start": float(start), "end": float(end), "score": float(score),
             "ctc_anchor": float(anchors[begins[i] + 1])}
            for i, (start, end, score) in enumerate(segments)]


def attach_timings(mapping, segments, audio):
    if len(segments) != len(mapping["tokens"]):
        raise ValueError("Alignment returned a different number of spans than source tokens.")
    tokens = []
    previous_end = 0.0
    for token, segment in zip(mapping["tokens"], segments):
        start, end, anchor, score = (segment[key] for key in ("start", "end", "ctc_anchor", "score"))
        if (not all(math.isfinite(v) for v in (start, end, anchor, score))
                or not 0 <= start < end <= audio["duration_seconds"]
                or start + 1e-9 < previous_end or not 0 <= anchor <= audio["duration_seconds"]
                or score <= -1e9):
            raise ValueError(f"Invalid/nonmonotonic alignment span for token {token['token_index']}.")
        tokens.append({**token, "local_start_seconds": start, "local_end_seconds": end,
                       "source_start_seconds": audio["start_seconds"] + start,
                       "source_end_seconds": audio["start_seconds"] + end,
                       "ctc_anchor_local_seconds": anchor,
                       "ctc_anchor_source_seconds": audio["start_seconds"] + anchor,
                       "segment_log_score": score})
        previous_end = end
    boundaries = []
    for mark in mapping["punctuation"]:
        left_index, right_index = mark["preceding_token_index"], mark["following_token_index"]
        left = tokens[left_index] if left_index is not None else None
        right = tokens[right_index] if right_index is not None else None
        has_pair = left is not None and right is not None
        boundaries.append({**mark, "preceding_token": left, "following_token": right,
                           "preceding_token_end_seconds": left["source_end_seconds"] if left else None,
                           "following_token_start_seconds": right["source_start_seconds"] if right else None,
                           "interval_local_seconds": [left["local_end_seconds"], right["local_start_seconds"]] if has_pair else None,
                           "interval_source_seconds": [left["source_end_seconds"], right["source_start_seconds"]] if has_pair else None,
                           "estimated_gap_seconds": right["local_start_seconds"] - left["local_end_seconds"] if has_pair else None,
                           "status": "adjacent_tokens" if has_pair else "missing_left_or_right_token"})
    return tokens, boundaries


def run_alignment(input_path, text_path, model_id=DEFAULT_MODEL, device="cpu", start_seconds=0.0,
                  end_seconds=None, revision="main"):
    """Return success or failure evidence; failed validation never runs inference."""
    report = {"schema_version": 1, "status": "failed", "input_wav": str(Path(input_path).resolve()),
              "input_wav_sha256": None, "transcript_file": str(Path(text_path).resolve()),
              "model_id": model_id, "requested_revision": revision, "device": device,
              "audio": None, "text_mapping": None, "tokenizer": None, "model": None,
              "token_timings": [], "punctuation_boundaries": [], "error": None,
              "method": "known-transcript ctc-segmentation; one segment per Han character",
              "timing_limitations": "Library midpoint/0.5 s span estimates, not phone boundaries or validated silence. Raw anchors retained. Scores are log scores, not calibrated confidence.",
              "versions": {}}
    for package in ("torch", "numpy", "scipy", "soundfile", "transformers", "ctc-segmentation"):
        try:
            report["versions"][package] = version(package)
        except PackageNotFoundError:
            report["versions"][package] = None
    stage = "input"
    try:
        report["input_wav_sha256"] = file_sha256(input_path)
        raw_text = Path(text_path).read_bytes()
        report["transcript_file_sha256"] = hashlib.sha256(raw_text).hexdigest()
        mapping = map_source(raw_text.decode("utf-8"))
        report["text_mapping"] = mapping
        stage = "text_validation"
        if mapping["normalization_failures"]:
            details = ", ".join(f"{f['character']!r} at {f['source_index']}" for f in mapping["normalization_failures"])
            raise ValueError(f"Unresolved non-Han input: {details}. No automatic normalization is performed.")
        if not mapping["tokens"]:
            raise ValueError("Transcript contains no Han acoustic tokens.")
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda.")
        stage = "audio_validation"
        samples, audio = read_analysis_audio(input_path, start_seconds, end_seconds)
        report["audio"] = audio
        stage = "vocabulary_validation"
        tokenizer = load_tokenizer(model_id, revision)
        token_ids, labels, token_meta = validate_vocabulary(mapping, tokenizer)
        report["tokenizer"] = token_meta
        stage = "model_inference"
        emissions, frame_seconds, model_meta = infer_emissions(samples, model_id, revision, device, len(labels), token_meta["blank_token_id"])
        report["model"] = model_meta
        stage = "ctc_alignment"
        segments = align_emissions(emissions, token_ids, labels, token_meta["blank_token_id"], frame_seconds)
        report["token_timings"], report["punctuation_boundaries"] = attach_timings(mapping, segments, audio)
        report["status"] = "aligned_listening_pending"
    except Exception as error:
        report["error"] = {"stage": stage, "type": type(error).__name__, "message": str(error)}
    return report


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Original WAV, read-only.")
    parser.add_argument("--text-file", required=True, type=Path, help="Exact UTF-8 transcript for the selected audio span.")
    parser.add_argument("--output", required=True, type=Path, help="New JSON evidence file, also written on alignment failure.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID or local directory.")
    parser.add_argument("--revision", default="main", help="Model/tokenizer revision (prefer a pinned commit for repeat runs).")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--end-seconds", type=float, help="Exclusive slice end; default is WAV end.")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.resolve() in (args.input.resolve(), args.text_file.resolve()):
        parser.error("Output must be a new file, different from the WAV and transcript; refusing overwrite.")
    # Reserve output before expensive work, protecting against races and bad paths.
    try:
        with args.output.open("x", encoding="utf-8") as output:
            report = run_alignment(args.input, args.text_file, args.model, args.device,
                                   args.start_seconds, args.end_seconds, args.revision)
            output.write(json_text(report))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if report["error"]:
        print(f"Alignment failed ({report['error']['stage']}): {report['error']['message']}", file=sys.stderr)
    print(f"{report['status']}: {len(report['token_timings'])} tokens; "
          f"{len(report['punctuation_boundaries'])} punctuation records -> {args.output}")
    return 1 if report["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
