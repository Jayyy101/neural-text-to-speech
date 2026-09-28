"""Compare combined and independent CosyVoice zero-shot unit paths.

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
CONDITIONS = (
    {
        "id": "A_combined",
        "method": "combined_zero_shot",
        "public_api_calls": 1,
        "expected_yields": 4,
    },
    {
        "id": "B_independent",
        "method": "independent_zero_shot",
        "public_api_calls": 4,
        "expected_yields": 4,
    },
)


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_plan():
    source_plan = json.loads(SOURCE_PLAN.read_text(encoding="utf-8"))
    units = source_plan.get("units")
    if not isinstance(units, list) or len(units) != 4:
        raise ValueError("The approved source plan must contain exactly four units.")
    for unit in units:
        seed = unit.get("seed")
        text = unit.get("normalized_text")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError(f"Invalid seed for {unit.get('id')}.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Missing normalized text for {unit.get('id')}.")
    source_path = ROOT / source_plan["source_path"]
    source_text = source_path.read_text(encoding="utf-8-sig")
    if not source_text.startswith(source_plan["source_excerpt"]):
        raise ValueError("The benchmark source no longer matches the approved excerpt.")
    return source_plan, source_path


def show_plan(plan):
    units = plan["units"]
    print("CosyVoice combined-path versus independent-path zero-shot plan")
    print("Prompt WAV:", plan["prompt_wav"])
    print("Prompt transcript:", plan["prompt_transcript_file"])
    print("Units:")
    for index, unit in enumerate(units, 1):
        print(f"  {index}. {unit['id']} seed={unit['seed']}")
        print(f"     {unit['normalized_text']}")
    print("Condition A: one inference_zero_shot call over the joined text;")
    print("             its four frontend units are consumed as four sequential yields.")
    print("Condition B: four inference_zero_shot calls, one approved unit per call.")
    print("Expected public inference_zero_shot calls: 5 (A=1, B=4)")
    print("Expected internal model.tts jobs / WAV yields: 8 (A=4, B=4)")
    print("Generation requires the explicit --execute flag.")


def create_run_directory(parent):
    parent.mkdir(parents=True, exist_ok=True)
    stem = "cosyvoice_multiunit_path_ab_" + datetime.now().strftime(
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


def save_manifest(path, manifest):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def one_speech(output, torch):
    speech = output["tts_speech"]
    if speech.ndim != 2 or speech.shape[0] != 1 or speech.shape[1] <= 0:
        raise RuntimeError("CosyVoice yielded an invalid speech tensor.")
    return speech.cpu()


def write_record_audio(torchaudio, model, run_dir, output_path, speech, record):
    write_pcm16_wav(torchaudio, output_path, speech, model.sample_rate)
    audio = wav_info(output_path, model.sample_rate)
    record.update({
        "status": "passed_wav_check",
        "duration_seconds": audio["duration_seconds"],
        "audio": audio,
        "wav_sha256": file_sha256(output_path),
        "output_path": str(output_path.relative_to(run_dir)),
    })


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
        "experiment_id": "cosyvoice_multiunit_path_ab",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "initializing",
        "objective": (
            "Test whether grouping the four units inside one public zero-shot call "
            "changes their raw generated audio versus four independent calls."
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
        "conditions": list(CONDITIONS),
        "expected_counts": {
            "public_inference_zero_shot_calls": 5,
            "internal_model_tts_jobs": 8,
            "wav_yields": 8,
        },
        "seed_policy": (
            "For A, reset all CosyVoice RNGs to the unit seed immediately before "
            "requesting that unit's next yield from the single public generator. "
            "For B, reset to the same unit seed immediately before its independent "
            "public inference call."
        ),
        "limitations": [
            "The original chapter used the model default random sequence, so its exact random state cannot be reproduced.",
            "A keeps one Python generator alive while B exits and re-enters the public API for each unit.",
            "Seed resets do not force deterministic GPU kernels or eliminate thread-scheduling differences.",
            "A normalizes the combined text once; B invokes normalization separately for each already-normalized unit.",
            "Resetting A before each yield is a control for this experiment and differs from the original unseeded chapter run."
        ],
        "combined_input_text": combined_text,
        "combined_input_text_sha256": text_sha256(combined_text),
        "units": [
            {
                **unit,
                "normalized_text_sha256": text_sha256(unit["normalized_text"]),
            }
            for unit in units
        ],
        "records": [],
        "concatenated_outputs": [],
    }
    save_manifest(manifest_path, manifest)

    speeches = {condition["id"]: [] for condition in CONDITIONS}
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

        combined_units = model.frontend.text_normalize(combined_text, split=True)
        independent_units = [
            model.frontend.text_normalize(text, split=True) for text in unit_texts
        ]
        manifest["normalization_validation"] = {
            "combined_units": combined_units,
            "combined_matches_plan": combined_units == unit_texts,
            "independent_units": independent_units,
            "independent_match_plan": independent_units == [[text] for text in unit_texts],
        }
        if combined_units != unit_texts or independent_units != [[text] for text in unit_texts]:
            raise RuntimeError(
                "Current frontend normalization differs from the approved four-unit plan; "
                "no inference was run."
            )

        manifest["status"] = "running"
        save_manifest(manifest_path, manifest)
        combined_dir = run_dir / "A_combined"
        combined_dir.mkdir()
        combined_generator = iter(model.inference_zero_shot(
            combined_text,
            prompt_text,
            str(prompt_wav),
            stream=False,
            text_frontend=True,
        ))
        for yield_index, unit in enumerate(units, 1):
            set_cosyvoice_random_seed(unit["seed"])
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = next(combined_generator)
            torch.cuda.synchronize()
            speech = one_speech(output, torch)
            record = {
                "condition": "A_combined",
                "method": "combined_zero_shot",
                "public_call_index": 1,
                "condition_call_index": 1,
                "yield_index": yield_index,
                "unit_id": unit["id"],
                "seed": unit["seed"],
                "normalized_text": unit["normalized_text"],
                "normalized_text_sha256": text_sha256(unit["normalized_text"]),
                "inference_seconds": time.perf_counter() - started,
                "status": "writing",
            }
            output_path = combined_dir / f"yield_{yield_index:02d}_{unit['id']}.wav"
            write_record_audio(
                torchaudio, model, run_dir, output_path, speech, record
            )
            speeches["A_combined"].append(speech)
            manifest["records"].append(record)
            save_manifest(manifest_path, manifest)
        try:
            next(combined_generator)
        except StopIteration:
            pass
        else:
            raise RuntimeError("Combined call yielded more than four outputs.")

        independent_dir = run_dir / "B_independent"
        independent_dir.mkdir()
        for condition_call_index, unit in enumerate(units, 1):
            set_cosyvoice_random_seed(unit["seed"])
            torch.cuda.synchronize()
            started = time.perf_counter()
            generator = iter(model.inference_zero_shot(
                unit["normalized_text"],
                prompt_text,
                str(prompt_wav),
                stream=False,
                text_frontend=True,
            ))
            output = next(generator)
            try:
                next(generator)
            except StopIteration:
                pass
            else:
                raise RuntimeError(
                    f"Independent call for {unit['id']} yielded more than once."
                )
            torch.cuda.synchronize()
            speech = one_speech(output, torch)
            record = {
                "condition": "B_independent",
                "method": "independent_zero_shot",
                "public_call_index": condition_call_index + 1,
                "condition_call_index": condition_call_index,
                "yield_index": 1,
                "unit_id": unit["id"],
                "seed": unit["seed"],
                "normalized_text": unit["normalized_text"],
                "normalized_text_sha256": text_sha256(unit["normalized_text"]),
                "inference_seconds": time.perf_counter() - started,
                "status": "writing",
            }
            output_path = independent_dir / (
                f"call_{condition_call_index:02d}_{unit['id']}.wav"
            )
            write_record_audio(
                torchaudio, model, run_dir, output_path, speech, record
            )
            speeches["B_independent"].append(speech)
            manifest["records"].append(record)
            save_manifest(manifest_path, manifest)

        for condition in CONDITIONS:
            condition_id = condition["id"]
            output_path = run_dir / condition_id / "concatenated.wav"
            joined = torch.cat(speeches[condition_id], dim=1)
            write_pcm16_wav(torchaudio, output_path, joined, model.sample_rate)
            audio = wav_info(output_path, model.sample_rate)
            expected_frames = sum(
                record["audio"]["frames"]
                for record in manifest["records"]
                if record["condition"] == condition_id
            )
            if audio["frames"] != expected_frames:
                raise RuntimeError("Raw concatenated frame count does not match its yields.")
            manifest["concatenated_outputs"].append({
                "condition": condition_id,
                "output_path": str(output_path.relative_to(run_dir)),
                "operation": "torch.cat in unit order; no silence, trim, crossfade, gain, or postprocessing",
                "audio": audio,
                "wav_sha256": file_sha256(output_path),
            })

        manifest["actual_counts"] = {
            "public_inference_zero_shot_calls": 5,
            "internal_model_tts_jobs": len(manifest["records"]),
            "wav_yields": len(manifest["records"]),
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
        help="Run the five public calls/eight expected GPU synthesis jobs.",
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
