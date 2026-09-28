"""Sweep global seeds across four sequential CosyVoice zero-shot units.

Without --execute this prints the fixed plan and performs no model inference.
Real generation is isolated under outputs/evaluation/.
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

from evaluation.run_cosyvoice_prompt_conditioning_ab import (
    configure_local_wetext_frontend,
)
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


SOURCE_PLAN = ROOT / "evaluation/inputs/cosyvoice_prompt_conditioning_ab.json"
COSYVOICE_ROOT = Path.home() / "CosyVoice"
MODEL_DIR = COSYVOICE_ROOT / "pretrained_models/Fun-CosyVoice3-0.5B"
GLOBAL_SEEDS = (2026091701, 2026091702, 2026091703, 2026091704)


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_plan():
    plan = json.loads(SOURCE_PLAN.read_text(encoding="utf-8"))
    units = plan.get("units")
    if not isinstance(units, list) or len(units) != 4:
        raise ValueError("The approved source plan must contain exactly four units.")
    expected_seeds = tuple(unit.get("seed") for unit in units)
    if expected_seeds != GLOBAL_SEEDS:
        raise ValueError("Approved unit seeds changed; review the sweep seed selection.")
    for unit in units:
        text = unit.get("normalized_text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Missing normalized text for {unit.get('id')}.")
    source_path = ROOT / plan["source_path"]
    source_text = source_path.read_text(encoding="utf-8-sig")
    if not source_text.startswith(plan["source_excerpt"]):
        raise ValueError("The benchmark source no longer matches the approved excerpt.")
    return plan, source_path


def show_plan(plan):
    print("CosyVoice sequential RNG-state sweep")
    print("Global seeds:", ", ".join(str(seed) for seed in GLOBAL_SEEDS))
    print("Seed policy: one RNG reset before each four-unit sequence; no per-unit resets.")
    print("Units:")
    for index, unit in enumerate(plan["units"], 1):
        print(f"  {index}. {unit['id']}")
        print(f"     {unit['normalized_text']}")
    print("Expected public inference_zero_shot calls: 4")
    print("Expected internal model.tts jobs / WAV yields: 16")
    print("Expected raw concatenations: 4")
    print("Generation requires the explicit --execute flag.")


def create_run_directory(parent):
    parent.mkdir(parents=True, exist_ok=True)
    stem = "cosyvoice_rng_sweep_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    number = 1
    while True:
        candidate = parent / (stem if number == 1 else f"{stem}_{number:02d}")
        try:
            candidate.mkdir()
            return candidate.resolve()
        except FileExistsError:
            number += 1


def save_manifest(path, manifest):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_speech(output):
    speech = output["tts_speech"]
    if speech.ndim != 2 or speech.shape[0] != 1 or speech.shape[1] <= 0:
        raise RuntimeError("CosyVoice yielded an invalid speech tensor.")
    return speech.cpu()


def execute(plan, source_path, output_parent):
    units = plan["units"]
    unit_texts = [unit["normalized_text"] for unit in units]
    combined_text = "".join(unit_texts)
    prompt_wav = Path(plan["prompt_wav"])
    prompt_text_file = Path(plan["prompt_transcript_file"])
    prompt_transcript = normalize_prompt_transcript(
        prompt_text_file.read_text(encoding="utf-8-sig")
    )
    prompt_text = PROMPT_PREFIX + prompt_transcript
    for required in (
        COSYVOICE_ROOT,
        MODEL_DIR,
        MODEL_DIR / "cosyvoice3.yaml",
        prompt_wav,
        prompt_text_file,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    run_dir = create_run_directory(output_parent)
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "experiment_id": "cosyvoice_rng_sweep",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "initializing",
        "objective": (
            "Test whether unwanted onset syllables vary with the global RNG state "
            "when four frontend units run sequentially without per-unit reseeding."
        ),
        "source": {
            "path": str(source_path),
            "file_sha256": file_sha256(source_path),
            "full_chapter_generation": False,
        },
        "source_plan": str(SOURCE_PLAN),
        "source_plan_sha256": file_sha256(SOURCE_PLAN),
        "runner_sha256": file_sha256(Path(__file__)),
        "cosyvoice_repo": str(COSYVOICE_ROOT),
        "cosyvoice_git_head": git_head(COSYVOICE_ROOT),
        "model_dir": str(MODEL_DIR),
        "model_config_sha256": file_sha256(MODEL_DIR / "cosyvoice3.yaml"),
        "prompt": {
            "wav": str(prompt_wav),
            "wav_sha256": file_sha256(prompt_wav),
            "transcript_file": str(prompt_text_file),
            "transcript_file_sha256": file_sha256(prompt_text_file),
            "transcript": prompt_transcript,
            "transcript_sha256": text_sha256(prompt_transcript),
        },
        "settings": dict(SETTLED_SETTINGS),
        "global_seeds": list(GLOBAL_SEEDS),
        "seed_policy": (
            "Call set_all_random_seed exactly once before each four-unit public "
            "inference_zero_shot generator; never reset between its yields."
        ),
        "expected_counts": {
            "rng_resets": 4,
            "public_inference_zero_shot_calls": 4,
            "internal_model_tts_jobs": 16,
            "wav_yields": 16,
            "raw_concatenations": 4,
        },
        "combined_input_text": combined_text,
        "combined_input_text_sha256": text_sha256(combined_text),
        "units": [
            {
                "id": unit["id"],
                "unit_index": index,
                "artifact_areas": unit["artifact_areas"],
                "normalized_text": unit["normalized_text"],
                "normalized_text_sha256": text_sha256(unit["normalized_text"]),
            }
            for index, unit in enumerate(units, 1)
        ],
        "records": [],
        "concatenated_outputs": [],
    }
    save_manifest(manifest_path, manifest)

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

        normalized_units = model.frontend.text_normalize(combined_text, split=True)
        manifest["normalization_validation"] = {
            "actual_units": normalized_units,
            "matches_plan": normalized_units == unit_texts,
        }
        if normalized_units != unit_texts:
            raise RuntimeError(
                "Current frontend normalization differs from the approved four-unit plan; "
                "no inference was run."
            )

        manifest["status"] = "running"
        save_manifest(manifest_path, manifest)
        generation_order = 0
        for seed_set_index, global_seed in enumerate(GLOBAL_SEEDS, 1):
            seed_dir = run_dir / f"seed_{global_seed}"
            seed_dir.mkdir()

            # The single reset for this seed set occurs here. Nothing reseeds the
            # generator while its four sequential unit yields are consumed.
            set_cosyvoice_random_seed(global_seed)
            generator = iter(model.inference_zero_shot(
                combined_text,
                prompt_text,
                str(prompt_wav),
                stream=False,
                text_frontend=True,
            ))
            sequence_started = time.perf_counter()
            yielded = []
            yield_elapsed = []
            previous = sequence_started
            for unit_index in range(1, len(units) + 1):
                output = next(generator)
                now = time.perf_counter()
                yielded.append(validate_speech(output))
                yield_elapsed.append(now - previous)
                previous = now
            try:
                next(generator)
            except StopIteration:
                pass
            else:
                raise RuntimeError(
                    f"Seed {global_seed} yielded more than four outputs."
                )
            torch.cuda.synchronize()
            sequence_seconds = time.perf_counter() - sequence_started

            for unit_index, (unit, speech, elapsed) in enumerate(
                zip(units, yielded, yield_elapsed), 1
            ):
                generation_order += 1
                output_path = seed_dir / f"unit_{unit_index:02d}.wav"
                write_pcm16_wav(torchaudio, output_path, speech, model.sample_rate)
                audio = wav_info(output_path, model.sample_rate)
                manifest["records"].append({
                    "method": "sequential_zero_shot_global_seed",
                    "seed_set_index": seed_set_index,
                    "global_seed": global_seed,
                    "public_call_index": seed_set_index,
                    "unit_index": unit_index,
                    "yield_index": unit_index,
                    "generation_order": generation_order,
                    "unit_id": unit["id"],
                    "normalized_text": unit["normalized_text"],
                    "normalized_text_sha256": text_sha256(unit["normalized_text"]),
                    "yield_elapsed_seconds": elapsed,
                    "sequence_elapsed_seconds": sequence_seconds,
                    "status": "passed_wav_check",
                    "output_path": str(output_path.relative_to(run_dir)),
                    "duration_seconds": audio["duration_seconds"],
                    "audio": audio,
                    "wav_sha256": file_sha256(output_path),
                })

            concatenated_path = seed_dir / "concatenated.wav"
            joined = torch.cat(yielded, dim=1)
            write_pcm16_wav(
                torchaudio, concatenated_path, joined, model.sample_rate
            )
            concatenated_audio = wav_info(concatenated_path, model.sample_rate)
            expected_frames = sum(
                record["audio"]["frames"]
                for record in manifest["records"]
                if record["global_seed"] == global_seed
            )
            if concatenated_audio["frames"] != expected_frames:
                raise RuntimeError(
                    f"Raw concatenation frame count mismatch for seed {global_seed}."
                )
            manifest["concatenated_outputs"].append({
                "seed_set_index": seed_set_index,
                "global_seed": global_seed,
                "output_path": str(concatenated_path.relative_to(run_dir)),
                "operation": "torch.cat in unit order; no silence, trim, crossfade, gain, or postprocessing",
                "audio": concatenated_audio,
                "wav_sha256": file_sha256(concatenated_path),
            })
            save_manifest(manifest_path, manifest)

        manifest["actual_counts"] = {
            "rng_resets": len(GLOBAL_SEEDS),
            "public_inference_zero_shot_calls": len(GLOBAL_SEEDS),
            "internal_model_tts_jobs": len(manifest["records"]),
            "wav_yields": len(manifest["records"]),
            "raw_concatenations": len(manifest["concatenated_outputs"]),
        }
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
        help="Run four public calls and sixteen expected GPU synthesis jobs.",
    )
    parser.add_argument(
        "--output-parent",
        type=Path,
        default=ROOT / "outputs/evaluation",
    )
    args = parser.parse_args(argv)
    plan, source_path = load_plan()
    show_plan(plan)
    if not args.execute:
        return 0
    return execute(plan, source_path, args.output_parent.expanduser().resolve())


if __name__ == "__main__":
    raise SystemExit(main())
