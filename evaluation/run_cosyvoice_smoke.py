"""Opt-in CosyVoice3 smoke test for the Mandarin diagnostic corpus or custom text."""

import argparse
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

from src.audiobook.cosyvoice import (
    PROMPT_PREFIX,
    create_cosyvoice_model,
    file_sha256 as file_hash,
    git_head,
    infer_zero_shot,
    load_cosyvoice_runtime,
    normalize_prompt_transcript,
    wav_info,
    write_pcm16_wav,
)

CORPUS = ROOT / "evaluation/inputs/mandarin_diagnostics.json"

COSYVOICE_ROOT = Path.home() / "CosyVoice"
MODEL_DIR = COSYVOICE_ROOT / "pretrained_models/Fun-CosyVoice3-0.5B"
PROMPT_WAV = COSYVOICE_ROOT / "asset/zero_shot_prompt.wav"

PROMPT_TRANSCRIPT = "希望你以后能够做的比我还好呦。"


def create_run_directory(parent, now=None):
    parent.mkdir(parents=True, exist_ok=True)
    stem = "cosyvoice_baseline_" + (
        now or datetime.now()
    ).strftime("%Y-%m-%d_%H-%M-%S")

    number = 1
    while True:
        candidate = parent / (
            stem if number == 1 else f"{stem}_{number:02d}"
        )
        try:
            candidate.mkdir()
            return candidate.resolve()
        except FileExistsError:
            number += 1


def plan_trials(cases, repeats):
    records = []

    for case in cases:
        label = (
            "mixed_language"
            if case["id"] == "mixed_text"
            else case["id"]
        )

        for repeat in range(1, repeats + 1):
            records.append({
                "case_id": case["id"],
                "category": case["category"],
                "text": case["text"],
                "repeat": repeat,
                "status": "pending",
                "expected_output_filename":
                    f"{label}_{repeat:02d}.wav",
                "output_filename": None,
                "trial_elapsed_seconds": None,
            })

    return records


def main(argv=None):
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))

    parser = argparse.ArgumentParser(description=__doc__)

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--case",
        choices=[c["id"] for c in corpus["cases"]],
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Run all seven diagnostic cases.",
    )
    selection.add_argument(
        "--text", help="Synthesize custom text instead of a diagnostic case.",
    )
    selection.add_argument(
        "--text-file", type=Path,
        help="Read custom text from a UTF-8 file, preserving punctuation and newlines.",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="Calls per case (default: 2).",
    )
    parser.add_argument(
        "--prompt-wav", type=Path, default=PROMPT_WAV,
        help="Reference WAV (default: stock CosyVoice zero-shot reference).",
    )
    transcript = parser.add_mutually_exclusive_group()
    transcript.add_argument(
        "--prompt-text", help="Reference transcript; the CosyVoice3 prefix is added automatically.",
    )
    transcript.add_argument(
        "--prompt-text-file", type=Path,
        help="UTF-8 reference transcript file; no CosyVoice3 prefix is needed.",
    )

    args = parser.parse_args(argv)

    if args.repeat < 1:
        parser.error("--repeat must be positive")
    prompt_wav = args.prompt_wav.expanduser().resolve()
    transcript_file = (
        args.prompt_text_file.expanduser().resolve()
        if args.prompt_text_file is not None else None
    )
    if (prompt_wav != PROMPT_WAV.resolve()
            and args.prompt_text is None and transcript_file is None):
        parser.error("A custom --prompt-wav requires --prompt-text or --prompt-text-file")

    custom_text = args.text
    if args.text_file is not None:
        try:
            custom_text = args.text_file.expanduser().read_bytes().decode("utf-8")
        except (OSError, UnicodeError) as error:
            parser.error(f"Cannot read --text-file as UTF-8: {error}")
    if custom_text is not None:
        if not custom_text.strip():
            parser.error("Custom text must not be empty or whitespace-only.")
        cases = [{"id": "custom_text", "category": "custom_text", "text": custom_text}]
    else:
        cases = [
            c for c in corpus["cases"]
            if args.all or c["id"] == (args.case or "narration")
        ]

    run_dir = create_run_directory(
        ROOT / "outputs/evaluation"
    )

    manifest = {
        "schema_version": 2,
        "run_id": run_dir.name,
        "started_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "starting",
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "corpus_id": corpus["corpus_id"] if custom_text is None else None,
        "cosyvoice_repo": str(COSYVOICE_ROOT),
        "model_dir": str(MODEL_DIR),
        "prompt_wav": str(prompt_wav),
        "prompt_transcript_file": str(transcript_file) if transcript_file else None,
        "settings": {
            "load_trt": False,
            "load_vllm": False,
            "fp16": False,
            "stream": False,
        },
        "repeats": args.repeat,
        "records": plan_trials(cases, args.repeat),
        "notes": [
            ("Same Mandarin diagnostic corpus as Melo baseline."
             if custom_text is None else "Custom text experiment; not a diagnostic corpus run."),
            "Model is loaded once per benchmark run.",
            "No application-level text chunking is added.",
            "All chunks yielded internally by CosyVoice are concatenated in order.",
            "Zero-shot prompt WAV and transcript remain fixed across every case.",
            "Human listening remains a separate evaluation step.",
            "WAV checks validate basic PCM structure, not pronunciation, completeness of narration, repetition, or quality.",
            "Peak memory is PyTorch CUDA allocated GiB, not total process/device VRAM.",
        ],
    }

    manifest_path = run_dir / "manifest.json"

    def save():
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest_path)

    save()

    print("Results:", run_dir)

    try:
        manifest["corpus_sha256"] = file_hash(CORPUS) if custom_text is None else None
        manifest["runner_sha256"] = file_hash(Path(__file__))
        prompt_transcript = normalize_prompt_transcript((
            transcript_file.read_text(encoding="utf-8-sig")
            if transcript_file else
            (args.prompt_text if args.prompt_text is not None else PROMPT_TRANSCRIPT)
        ))
        prompt_text = PROMPT_PREFIX + prompt_transcript
        manifest["prompt_transcript"] = prompt_transcript
        manifest["prompt_transcript_sha256"] = hashlib.sha256(
            prompt_transcript.encode("utf-8")
        ).hexdigest()
        manifest["prompt_text"] = prompt_text
        if transcript_file:
            manifest["prompt_transcript_file_sha256"] = file_hash(transcript_file)
        for required in (COSYVOICE_ROOT, MODEL_DIR, prompt_wav):
            if not required.exists():
                raise FileNotFoundError(required)
        manifest["cosyvoice_git_head"] = git_head(COSYVOICE_ROOT)
        manifest["model_config_sha256"] = file_hash(MODEL_DIR / "cosyvoice3.yaml")
        manifest["prompt_wav_sha256"] = file_hash(prompt_wav)
        save()

        import_started = time.perf_counter()

        torch, torchaudio, AutoModel = load_cosyvoice_runtime(COSYVOICE_ROOT)
        manifest["torch_version"] = torch.__version__
        manifest["cuda"] = {
            "available": torch.cuda.is_available(),
            "build": torch.version.cuda,
        }
        if torch.cuda.is_available():
            manifest["cuda"]["device"] = torch.cuda.get_device_name(0)
        manifest["torchaudio_version"] = torchaudio.__version__

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Run this from the "
                "cosyvoice-b WSL environment."
            )

        manifest["import_seconds"] = (
            time.perf_counter() - import_started
        )

        print("Loading CosyVoice3...")

        load_started = time.perf_counter()

        model = create_cosyvoice_model(AutoModel, MODEL_DIR, manifest["settings"])

        manifest["model_load_seconds"] = (
            time.perf_counter() - load_started
        )
        manifest["sample_rate_hz"] = model.sample_rate
        manifest["status"] = "running"

        save()

        print(
            "Model loaded in",
            round(manifest["model_load_seconds"], 2),
            "sec",
        )

        for record in manifest["records"]:
            record["status"] = "running"
            record["started_at_utc"] = datetime.now(
                timezone.utc
            ).isoformat()
            save()

            started = time.perf_counter()

            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()

                inference_started = time.perf_counter()

                speech, chunk_count = infer_zero_shot(
                    model, torch, record["text"], prompt_text, prompt_wav,
                    stream=False,
                )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                inference_seconds = (
                    time.perf_counter()
                    - inference_started
                )

                output_path = (
                    run_dir
                    / record["expected_output_filename"]
                )

                write_pcm16_wav(torchaudio, output_path, speech, model.sample_rate)

                audio = wav_info(output_path, model.sample_rate)

                record.update({
                    "status": "passed_wav_check",
                    "output_filename":
                        output_path.name,
                    "wav_sha256":
                        file_hash(output_path),
                    "audio": audio,
                    "cosyvoice_chunks":
                        chunk_count,
                    "inference_seconds":
                        inference_seconds,
                    "rtf":
                        inference_seconds
                        / audio["duration_seconds"],
                    "peak_torch_cuda_allocated_gib":
                        torch.cuda.max_memory_allocated()
                        / 1024**3,
                })

            except Exception as error:
                record["status"] = "failed"
                record["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback":
                        traceback.format_exc(),
                }

            finally:
                record["trial_elapsed_seconds"] = (
                    time.perf_counter() - started
                )
                record["finished_at_utc"] = (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                )
                save()

            print(
                f"{record['case_id']} "
                f"repetition {record['repeat']:02d}: "
                f"{record['status']}"
            )

    except (Exception, KeyboardInterrupt) as error:
        manifest["status"] = "aborted"
        manifest["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        for record in manifest["records"]:
            if record["status"] == "pending":
                record["status"] = "not_run"
            elif record["status"] == "running":
                record["status"] = "failed"
                record["error"] = manifest["error"].copy()
                record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()

        print(
            manifest["error"]["traceback"],
            file=sys.stderr,
        )

    passed = sum(r["status"] == "passed_wav_check" for r in manifest["records"])
    failed = sum(r["status"] == "failed" for r in manifest["records"])
    not_completed = len(manifest["records"]) - passed - failed
    manifest["summary"] = {
        "passed_trials": passed,
        "failed_trials": failed,
        "not_completed_trials": not_completed,
        "total_trials": len(manifest["records"]),
    }
    if manifest["status"] != "aborted":
        manifest["status"] = (
            "passed_wav_checks_listening_pending"
            if passed == len(manifest["records"])
            else "completed_with_failures"
        )
    manifest["exit_code"] = int(
        manifest["status"] == "aborted" or failed > 0 or not_completed > 0
    )
    manifest["finished_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    save()

    print("Results and manifest:", run_dir)

    return manifest["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
