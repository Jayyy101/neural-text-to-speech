"""Generate six deterministic standalone retries for clean12 units 07-09."""

import argparse
import contextlib
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

from evaluation.run_cosyvoice_clean12 import Tee
from evaluation.run_cosyvoice_cr_format_ab import require
from evaluation.run_cosyvoice_rng_sweep import (
    COSYVOICE_ROOT, MODEL_DIR, configure_local_wetext_frontend,
    save_manifest, text_sha256, validate_speech,
)
from src.audiobook.cosyvoice import (
    PROMPT_PREFIX, SETTLED_SETTINGS, create_cosyvoice_model, file_sha256,
    git_head, load_cosyvoice_runtime, normalize_prompt_transcript,
    set_cosyvoice_random_seed, wav_info, write_pcm16_wav,
)

CLEAN12 = ROOT / "outputs/evaluation/cosyvoice_clean12_2026-09-20_08-36-03_246121/manifest.json"
UNITS = (7, 8, 9)
SEEDS = (2026091705, 2026091706)


def prepare():
    clean12 = json.loads(CLEAN12.read_text(encoding="utf-8"))
    require(clean12["status"] == "completed", "Clean12 run is incomplete")
    require(clean12["actual_counts"] == {
        "model_initializations": 1, "public_calls": 1,
        "synthesis_jobs": 12, "rng_resets": 1,
    }, "Clean12 execution counts changed")
    require(dict(SETTLED_SETTINGS) == clean12["settings"], "Runtime settings changed")
    records = {record["unit_index"]: record for record in clean12["records"]}
    require(set(UNITS).issubset(records), "Affected clean12 unit is missing")
    selected = []
    for unit in UNITS:
        record = records[unit]
        text = record["normalized_text"]
        require(text == clean12["actual_synthesis_texts"][unit - 1], "Recorded synthesis input differs")
        require(text_sha256(text) == record["text_sha256"], "Recorded text hash differs")
        require("\r" not in text and "\n" not in text, "Retry input contains CR/LF")
        require(file_sha256(CLEAN12.parent / record["output_path"]) == record["wav_sha256"],
                "Existing clean12 WAV changed")
        selected.append({
            "unit": unit,
            "text": text,
            "text_sha256": record["text_sha256"],
            "characters": record["characters"],
            "existing_wav": str(CLEAN12.parent / record["output_path"]),
            "existing_wav_sha256": record["wav_sha256"],
            "existing_duration_seconds": record["audio"]["duration_seconds"],
        })
    attempts = [
        {"unit": item["unit"], "seed": seed}
        for item in selected for seed in SEEDS
    ]
    return clean12, selected, attempts


def execute(prepared, run):
    clean12, selected, attempts = prepared
    selected_by_unit = {item["unit"]: item for item in selected}
    manifest_path = run / "manifest.json"
    started = time.perf_counter()
    manifest = {
        "schema_version": 1,
        "experiment_id": "cosyvoice_clean12_targeted_retries",
        "status": "initializing",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": file_sha256(Path(__file__)),
        "clean12_manifest": str(CLEAN12),
        "clean12_manifest_sha256": file_sha256(CLEAN12),
        "units": list(UNITS),
        "seeds": list(SEEDS),
        "settings": dict(SETTLED_SETTINGS),
        "seed_policy": "Reset requested seed once immediately before each standalone one-unit zero-shot call",
        "model_lifetime": "One initialization retained for all six retries; no extra warmup synthesis",
        "planned_attempts": attempts,
        "inputs": selected,
        "rng_resets": [],
        "actual_synthesis_inputs": [],
        "records": [],
        "expected_counts": {"model_initializations": 1, "public_calls": 6,
                            "synthesis_jobs": 6, "rng_resets": 6},
    }
    save_manifest(manifest_path, manifest)
    try:
        prompt = clean12["prompt"]
        prompt_wav = Path(prompt["wav"])
        prompt_file = Path(prompt["transcript_file"])
        transcript = normalize_prompt_transcript(prompt_file.read_text(encoding="utf-8-sig"))
        require(transcript == prompt["transcript"], "Prompt transcript changed")
        require(file_sha256(prompt_wav) == prompt["wav_sha256"], "Narrator WAV changed")
        require(file_sha256(prompt_file) == prompt["transcript_file_sha256"], "Prompt transcript file changed")
        prompt_text = PROMPT_PREFIX + transcript
        require("\r" not in prompt_text, "Prompt contains CR")
        manifest["prompt"] = prompt
        manifest["cosyvoice_git_head"] = git_head(COSYVOICE_ROOT)
        manifest["model_config_sha256"] = file_sha256(MODEL_DIR / "cosyvoice3.yaml")
        require(manifest["cosyvoice_git_head"] == clean12["cosyvoice_git_head"], "CosyVoice revision changed")
        require(manifest["model_config_sha256"] == clean12["model_config_sha256"], "Model config changed")

        import_started = time.perf_counter()
        torch, torchaudio, auto_model = load_cosyvoice_runtime(COSYVOICE_ROOT)
        manifest["import_seconds"] = time.perf_counter() - import_started
        require(torch.cuda.is_available(), "CUDA unavailable")
        manifest["runtime"] = {
            "python": platform.python_version(), "executable": sys.executable,
            "torch": torch.__version__, "torchaudio": torchaudio.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        }
        require(manifest["runtime"] == clean12["runtime"], "Runtime differs from clean12")
        load_started = time.perf_counter()
        model = create_cosyvoice_model(auto_model, MODEL_DIR, SETTLED_SETTINGS)
        manifest["model_load_seconds"] = time.perf_counter() - load_started
        manifest["frontend"] = configure_local_wetext_frontend(model.frontend)
        require(manifest["frontend"]["assets"] == clean12["frontend"]["assets"], "Frontend assets changed")

        for item in selected:
            observed = model.frontend.text_normalize(item["text"], split=True)
            require(observed == [item["text"]], f"Unit {item['unit']:02d} frontend preflight changed input")
        manifest["all_pre_synthesis_checks_passed"] = True
        manifest["status"] = "generating"
        save_manifest(manifest_path, manifest)
        print("Preflight passed: units 07-09 are exact, CR-free, and each remains one frontend unit.", flush=True)

        original_frontend = model.frontend.frontend_zero_shot
        active = {}

        def observed_frontend(*args, **kwargs):
            text = args[0] if args else kwargs["tts_text"]
            actual_prompt = args[1] if len(args) > 1 else kwargs["prompt_text"]
            require(text == active["text"] and text_sha256(text) == active["text_sha256"],
                    "Actual retry synthesis input differs")
            require("\r" not in text and "\n" not in text and "\r" not in actual_prompt,
                    "Control newline reached synthesis frontend")
            manifest["actual_synthesis_inputs"].append({
                "unit": active["unit"], "seed": active["seed"],
                "text": text, "text_sha256": text_sha256(text),
            })
            return original_frontend(*args, **kwargs)

        model.frontend.frontend_zero_shot = observed_frontend
        generation_started = time.perf_counter()
        for attempt_index, attempt in enumerate(attempts, 1):
            item = selected_by_unit[attempt["unit"]]
            active.clear()
            active.update(item, seed=attempt["seed"])
            torch.cuda.synchronize()
            set_cosyvoice_random_seed(attempt["seed"])
            manifest["rng_resets"].append({
                "attempt_index": attempt_index, "unit": item["unit"], "seed": attempt["seed"]
            })
            call_started = time.perf_counter()
            generator = iter(model.inference_zero_shot(
                item["text"], prompt_text, str(prompt_wav), stream=False, text_frontend=True
            ))
            speech = validate_speech(next(generator))
            try:
                next(generator)
            except StopIteration:
                pass
            else:
                raise RuntimeError("Standalone retry yielded more than one WAV")
            torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - call_started
            output_path = run / f"unit_{item['unit']:02d}_seed_{attempt['seed']}.wav"
            write_pcm16_wav(torchaudio, output_path, speech, model.sample_rate)
            audio = wav_info(output_path, model.sample_rate)
            record = {
                "attempt_index": attempt_index, "unit": item["unit"], "seed": attempt["seed"],
                "text": item["text"], "text_sha256": item["text_sha256"],
                "characters": item["characters"], "generation_seconds": generation_seconds,
                "output_path": output_path.name, "audio": audio,
                "wav_sha256": file_sha256(output_path),
                "existing_clean12_wav": item["existing_wav"],
                "existing_clean12_wav_sha256": item["existing_wav_sha256"],
            }
            manifest["records"].append(record)
            save_manifest(manifest_path, manifest)
            print(f"{attempt_index}/6 unit {item['unit']:02d} seed {attempt['seed']}: "
                  f"{audio['duration_seconds']:.2f}s audio in {generation_seconds:.3f}s", flush=True)
        manifest["generation_seconds"] = time.perf_counter() - generation_started
        manifest["actual_counts"] = {
            "model_initializations": 1, "public_calls": len(manifest["records"]),
            "synthesis_jobs": len(manifest["records"]), "rng_resets": len(manifest["rng_resets"]),
        }
        require(manifest["actual_counts"] == manifest["expected_counts"], "Execution counts differ")
        require(len(manifest["actual_synthesis_inputs"]) == 6, "Synthesis observation count differs")
        manifest.update(status="completed", total_wall_seconds=time.perf_counter() - started,
                        finished_at_utc=datetime.now(timezone.utc).isoformat())
        save_manifest(manifest_path, manifest)
        print("Completed:", run, flush=True)
        return 0
    except Exception as error:
        manifest.update(status="failed", finished_at_utc=datetime.now(timezone.utc).isoformat(),
                        error={"type": type(error).__name__, "message": str(error),
                               "traceback": traceback.format_exc()})
        save_manifest(manifest_path, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    prepared = prepare()
    print(json.dumps({
        "units": [{"unit": item["unit"], "characters": item["characters"],
                   "text_sha256": item["text_sha256"], "text": item["text"]}
                  for item in prepared[1]],
        "seeds": list(SEEDS), "attempts": prepared[2],
        "expected_counts": {"model_initializations": 1, "public_calls": 6,
                            "synthesis_jobs": 6, "rng_resets": 6},
    }, ensure_ascii=True, indent=2))
    if not args.execute:
        return 0
    run = ROOT / "outputs/evaluation" / (
        "cosyvoice_clean12_retries_" + datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f")
    )
    run.mkdir()
    with (run / "console.log").open("x", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print("Output:", run, flush=True)
            return execute(prepared, run)


if __name__ == "__main__":
    raise SystemExit(main())
