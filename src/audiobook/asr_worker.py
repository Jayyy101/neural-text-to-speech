"""Audio-only JSON-lines Mandarin CTC worker for the tts-align environment."""

import contextlib
import hashlib
import json
from pathlib import Path
import sys

from . import profiling

profiling.mark("asr.entry")

from .content_qc import MODEL_ID, MODEL_REVISION


def _load_model():
    from evaluation.run_mandarin_asr_unit09_feasibility import load_asr_once
    return load_asr_once()


def _infer(corpus, torch, tokenizer, feature_extractor, model):
    from evaluation.run_mandarin_asr_unit09_feasibility import infer_audio_only
    return infer_audio_only(corpus, torch, tokenizer, feature_extractor, model)


def _write(value):
    sys.stdout.write(json.dumps(value, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _sha256(path):
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            total += len(block)
    profiling.annotate(bytes_read=total)
    return digest.hexdigest()


def run():
    # The model is loaded once. All subsequent requests reuse these objects.
    with profiling.span("asr.model_initialization"):
        torch, tokenizer, feature_extractor, model, metadata = _load_model()
    if metadata.get("model_id") != MODEL_ID or metadata.get("resolved_revision") != MODEL_REVISION:
        raise RuntimeError("ASR model identity differs from the pinned contract.")
    _write({"type": "ready", "model": metadata})
    for line in sys.stdin:
        request_span = profiling.begin("asr.request")
        request_id = None
        responded = False
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or set(request) != {
                "type", "request_id", "audio_path", "wav_sha256"
            } or request["type"] != "recognize":
                raise ValueError("Invalid audio-only ASR request schema.")
            request_id = request["request_id"]
            request_span.add(request_id=request_id)
            path = Path(request["audio_path"]).resolve(strict=True)
            with profiling.span("asr.wav_hash"):
                if not path.is_file() or _sha256(path) != request["wav_sha256"]:
                    raise ValueError("ASR request WAV is missing or differs from its SHA-256.")
            corpus = [{"id": request_id, "path": str(path),
                       "sha256": request["wav_sha256"]}]
            # The validated audit helper prints progress; reserve stdout for JSON.
            with contextlib.redirect_stdout(sys.stderr):
                with profiling.span("asr.recognition"):
                    result = _infer(
                        corpus, torch, tokenizer, feature_extractor, model
                    )[0]
            with profiling.span("asr.response_serialize"):
                _write({
                    "type": "recognized", "request_id": request_id,
                    "wav_sha256": request["wav_sha256"],
                    "raw_transcript": result["raw_transcript"],
                    "comparison_tokens": result["comparison_tokens"],
                    "comparison_text": result["comparison_text"],
                    "raw_emitted_tokens": result["raw_emitted_tokens"],
                    "ignored_tokens": result["ignored_tokens"],
                    "audio": result["audio"],
                    "inference_seconds": result["inference_seconds"],
                    "emission_frames": result["emission_frames"],
                    "ctc_frame_seconds": result["ctc_frame_seconds"],
                })
                responded = True
        except Exception as error:
            if not responded:
                _write({"type": "error", "request_id": request_id,
                        "message": f"{type(error).__name__}: {error}"})
            request_span.finish("error", error_type=type(error).__name__)
        else:
            request_span.finish()


if __name__ == "__main__":
    run()
