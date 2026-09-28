"""Controlled CosyVoice zero-shot versus cross-lingual prompt experiment.

The default action prints the fixed plan and performs no model inference. Real
generation requires --execute and writes only to outputs/evaluation/.
"""

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
    SETTLED_SETTINGS,
    create_cosyvoice_model,
    file_sha256,
    git_head,
    load_cosyvoice_runtime,
    normalize_prompt_transcript,
    set_cosyvoice_random_seed,
    wav_info,
    write_pcm16_wav,
)


PLAN_PATH = ROOT / "evaluation/inputs/cosyvoice_prompt_conditioning_ab.json"
COSYVOICE_ROOT = Path.home() / "CosyVoice"
MODEL_DIR = COSYVOICE_ROOT / "pretrained_models/Fun-CosyVoice3-0.5B"
METHODS = (
    {
        "id": "A_zero_shot",
        "api": "inference_zero_shot",
        "description": "Prompt WAV plus lexical prompt transcript",
    },
    {
        "id": "B_cross_lingual",
        "api": "inference_cross_lingual",
        "description": "Same prompt WAV without lexical LLM prompt conditioning",
    },
)


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_plan(path=PLAN_PATH):
    plan = json.loads(path.read_text(encoding="utf-8"))
    units = plan.get("units")
    if plan.get("schema_version") != 1 or not isinstance(units, list) or len(units) != 4:
        raise ValueError("The diagnostic plan must contain exactly four units.")
    if len({unit["id"] for unit in units}) != len(units):
        raise ValueError("Diagnostic unit IDs must be unique.")
    for unit in units:
        seed = unit.get("seed")
        text = unit.get("normalized_text")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError(f"Invalid seed for {unit.get('id')}.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Missing normalized text for {unit.get('id')}.")
    return plan


def validate_source(plan):
    source_path = ROOT / plan["source_path"]
    source_text = source_path.read_text(encoding="utf-8-sig")
    excerpt = plan["source_excerpt"]
    if not source_text.startswith(excerpt):
        raise ValueError(
            "The benchmark source opening no longer matches the fixed diagnostic excerpt."
        )
    return source_path, source_text, excerpt


def show_plan(plan):
    print("CosyVoice lexical-prompt A/B plan")
    print("Source:", plan["source_path"])
    print("Prompt WAV:", plan["prompt_wav"])
    print("Methods:")
    for method in METHODS:
        print(f"  {method['id']}: {method['api']} — {method['description']}")
    print("Units:")
    for index, unit in enumerate(plan["units"], 1):
        areas = ", ".join(unit["artifact_areas"])
        print(f"  {index}. {unit['id']} seed={unit['seed']} areas={areas}")
        print(f"     {unit['normalized_text']}")
    print("Estimated GPU inference calls:", len(plan["units"]) * len(METHODS))
    print("Per method:", len(plan["units"]))
    print("Generation requires the explicit --execute flag.")


def create_run_directory(parent):
    parent.mkdir(parents=True, exist_ok=True)
    stem = "cosyvoice_prompt_conditioning_ab_" + datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    number = 1
    while True:
        candidate = parent / (stem if number == 1 else f"{stem}_{number:02d}")
        try:
            candidate.mkdir()
            return candidate.resolve()
        except FileExistsError:
            number += 1


def collect_speech(outputs, torch):
    chunks = [output["tts_speech"] for output in outputs]
    if not chunks:
        raise RuntimeError("CosyVoice yielded no audio.")
    speech = torch.cat(chunks, dim=1).cpu()
    if len(chunks) != 1:
        raise RuntimeError(
            f"Expected one non-streaming yield for one fixed unit; received {len(chunks)}."
        )
    return speech, len(chunks)


def configure_local_wetext_frontend(frontend):
    """Bind the isolated diagnostic to the already-installed WeText FSTs."""
    from wetext import Normalizer

    cache = Path.home() / ".cache/modelscope/hub/pengzhendong/wetext"
    paths = {
        "zh_tagger": cache / "zh/tn/tagger.fst",
        "zh_verbalizer": cache / "zh/tn/verbalizer.fst",
        "en_tagger": cache / "en/tn/tagger.fst",
        "en_verbalizer": cache / "en/tn/verbalizer.fst",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    frontend.zh_tn_model = Normalizer(
        tagger_path=str(paths["zh_tagger"]),
        verbalizer_path=str(paths["zh_verbalizer"]),
        lang="zh",
    )
    frontend.en_tn_model = Normalizer(
        tagger_path=str(paths["en_tagger"]),
        verbalizer_path=str(paths["en_verbalizer"]),
        lang="en",
    )
    frontend.text_frontend = "wetext"
    return {
        "name": "wetext",
        "asset_source": "existing_local_modelscope_cache_via_custom_paths",
        "assets": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
    }


def infer_unit(model, torch, method_id, text, prompt_text, prompt_wav):
    if method_id == "A_zero_shot":
        outputs = model.inference_zero_shot(
            text,
            prompt_text,
            str(prompt_wav),
            stream=False,
            text_frontend=False,
        )
    elif method_id == "B_cross_lingual":
        outputs = model.inference_cross_lingual(
            PROMPT_PREFIX + text,
            str(prompt_wav),
            stream=False,
            text_frontend=False,
        )
    else:
        raise ValueError(f"Unsupported method: {method_id}")
    return collect_speech(outputs, torch)


def save_manifest(path, manifest):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def execute(plan, output_parent, resume_run=None):
    source_path, source_text, excerpt = validate_source(plan)
    prompt_wav = Path(plan["prompt_wav"])
    prompt_transcript_file = Path(plan["prompt_transcript_file"])
    for required in (
        COSYVOICE_ROOT,
        MODEL_DIR,
        MODEL_DIR / "cosyvoice3.yaml",
        prompt_wav,
        prompt_transcript_file,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    prompt_transcript = normalize_prompt_transcript(
        prompt_transcript_file.read_text(encoding="utf-8-sig")
    )
    prompt_text = PROMPT_PREFIX + prompt_transcript
    planned_texts = [unit["normalized_text"] for unit in plan["units"]]
    if resume_run is None:
        run_dir = create_run_directory(output_parent)
        manifest_path = run_dir / "manifest.json"
        manifest = {
            "schema_version": 1,
            "experiment_id": plan["experiment_id"],
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "initializing",
            "source": {
                "path": str(source_path),
                "file_sha256": file_sha256(source_path),
                "excerpt": excerpt,
                "excerpt_sha256": text_sha256(excerpt),
                "full_chapter_generation": False,
            },
            "plan_sha256": file_sha256(PLAN_PATH),
            "runner_sha256": file_sha256(Path(__file__)),
            "cosyvoice_repo": str(COSYVOICE_ROOT),
            "cosyvoice_git_head": git_head(COSYVOICE_ROOT),
            "model_dir": str(MODEL_DIR),
            "model_config_sha256": file_sha256(MODEL_DIR / "cosyvoice3.yaml"),
            "prompt": {
                "wav": str(prompt_wav),
                "wav_sha256": file_sha256(prompt_wav),
                "transcript_file": str(prompt_transcript_file),
                "transcript_file_sha256": file_sha256(prompt_transcript_file),
                "transcript": prompt_transcript,
                "transcript_sha256": text_sha256(prompt_transcript),
            },
            "settings": dict(SETTLED_SETTINGS),
            "seed_policy": (
                "Reset Python, NumPy, CPU Torch, and CUDA Torch RNGs to the recorded "
                "unit seed immediately before each method call; paired A/B calls use "
                "the same seed."
            ),
            "methods": list(METHODS),
            "planned_units": [
                {
                    **unit,
                    "normalized_text_sha256": text_sha256(unit["normalized_text"]),
                }
                for unit in plan["units"]
            ],
            "records": [],
            "concatenated_outputs": [],
        }
    else:
        run_dir = resume_run.expanduser().resolve()
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("experiment_id") != plan["experiment_id"]:
            raise ValueError("Resume directory is not this diagnostic experiment.")
        if manifest["source"]["file_sha256"] != file_sha256(source_path):
            raise ValueError("Benchmark source changed since the failed run.")
        if [unit["normalized_text"] for unit in manifest["planned_units"]] != planned_texts:
            raise ValueError("Resume manifest does not match the approved units.")
        manifest["status"] = "initializing"
        manifest.pop("error", None)
        manifest.pop("finished_at_utc", None)
        manifest["resume_count"] = manifest.get("resume_count", 0) + 1
        manifest["resumed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["resume_runner_sha256"] = file_sha256(Path(__file__))
        manifest["concatenated_outputs"] = []
    del source_text
    save_manifest(manifest_path, manifest)

    method_speeches = {method["id"]: [] for method in METHODS}
    try:
        torch, torchaudio, auto_model = load_cosyvoice_runtime(COSYVOICE_ROOT)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the CosyVoice environment.")
        manifest["runtime"] = {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
        }
        load_started = time.perf_counter()
        model = create_cosyvoice_model(auto_model, MODEL_DIR, SETTLED_SETTINGS)
        manifest["model_load_seconds"] = time.perf_counter() - load_started
        manifest["sample_rate_hz"] = model.sample_rate
        manifest["frontend"] = configure_local_wetext_frontend(model.frontend)

        actual_texts = model.frontend.text_normalize(excerpt, split=True)
        manifest["actual_normalized_units"] = actual_texts
        manifest["normalization_matches_plan"] = actual_texts == planned_texts
        if actual_texts != planned_texts:
            raise RuntimeError(
                "Current frontend normalization differs from the approved four-unit plan; "
                "no inference was run."
            )

        manifest["status"] = "running"
        save_manifest(manifest_path, manifest)
        for unit in plan["units"]:
            for method in METHODS:
                method_id = method["id"]
                output_dir = run_dir / method_id
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / f"{unit['id']}.wav"
                matches = [
                    item for item in manifest["records"]
                    if item["unit_id"] == unit["id"] and item["method"] == method_id
                ]
                if len(matches) > 1:
                    raise RuntimeError("Duplicate unit/method records in resume manifest.")
                record = matches[0] if matches else {}
                if record.get("status") == "passed_wav_check":
                    if file_sha256(output_path) != record["wav_sha256"]:
                        raise RuntimeError(f"Completed resume artifact changed: {output_path}")
                    speech, sample_rate = torchaudio.load(str(output_path))
                    if sample_rate != model.sample_rate or speech.shape[0] != 1:
                        raise RuntimeError(f"Invalid completed resume artifact: {output_path}")
                    method_speeches[method_id].append(speech)
                    print("Reusing completed artifact:", output_path)
                    continue
                new_record = {
                    "unit_id": unit["id"],
                    "method": method_id,
                    "api": method["api"],
                    "seed": unit["seed"],
                    "normalized_text": unit["normalized_text"],
                    "normalized_text_sha256": text_sha256(unit["normalized_text"]),
                    "artifact_areas": unit["artifact_areas"],
                    "status": "running",
                    "output_path": str(output_path.relative_to(run_dir)),
                }
                if method_id == "B_cross_lingual":
                    new_record["target_control_prefix"] = PROMPT_PREFIX
                    new_record["lexical_prompt_transcript_used"] = False
                record.clear()
                record.update(new_record)
                if not matches:
                    manifest["records"].append(record)
                save_manifest(manifest_path, manifest)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                set_cosyvoice_random_seed(unit["seed"])
                started = time.perf_counter()
                speech, yield_count = infer_unit(
                    model,
                    torch,
                    method_id,
                    unit["normalized_text"],
                    prompt_text,
                    prompt_wav,
                )
                torch.cuda.synchronize()
                inference_seconds = time.perf_counter() - started
                write_pcm16_wav(torchaudio, output_path, speech, model.sample_rate)
                audio = wav_info(output_path, model.sample_rate)
                method_speeches[method_id].append(speech)
                record.update({
                    "status": "passed_wav_check",
                    "yield_count": yield_count,
                    "inference_seconds": inference_seconds,
                    "duration_seconds": audio["duration_seconds"],
                    "audio": audio,
                    "wav_sha256": file_sha256(output_path),
                    "peak_torch_cuda_allocated_gib": (
                        torch.cuda.max_memory_allocated() / 1024**3
                    ),
                })
                save_manifest(manifest_path, manifest)

        for method in METHODS:
            method_id = method["id"]
            output_path = run_dir / method_id / "concatenated.wav"
            speech = torch.cat(method_speeches[method_id], dim=1)
            write_pcm16_wav(torchaudio, output_path, speech, model.sample_rate)
            audio = wav_info(output_path, model.sample_rate)
            expected_frames = sum(
                record["audio"]["frames"]
                for record in manifest["records"]
                if record["method"] == method_id
            )
            if audio["frames"] != expected_frames:
                raise RuntimeError("Concatenated frame count does not equal raw unit frames.")
            manifest["concatenated_outputs"].append({
                "method": method_id,
                "output_path": str(output_path.relative_to(run_dir)),
                "operation": "torch.cat in unit order; no silence, crossfade, trim, or gain",
                "audio": audio,
                "wav_sha256": file_sha256(output_path),
            })

        manifest["status"] = "completed"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_manifest(manifest_path, manifest)
        print("Completed:", run_dir)
        return 0
    except Exception as error:
        manifest["status"] = "failed"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        save_manifest(manifest_path, manifest)
        raise


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the eight GPU inference calls. Without this flag, only print the plan.",
    )
    parser.add_argument(
        "--output-parent",
        type=Path,
        default=ROOT / "outputs/evaluation",
        help="Parent directory for the isolated diagnostic run.",
    )
    parser.add_argument(
        "--resume-run",
        type=Path,
        help="Resume a failed diagnostic run, reusing validated completed unit WAVs.",
    )
    args = parser.parse_args(argv)
    plan = load_plan()
    validate_source(plan)
    show_plan(plan)
    if not args.execute:
        return 0
    return execute(
        plan,
        args.output_parent.expanduser().resolve(),
        resume_run=args.resume_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
