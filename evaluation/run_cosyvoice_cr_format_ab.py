"""Compare frozen clean/CR opening units; default is model-free verification."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
import wave

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.run_cosyvoice_rng_sweep import (
    COSYVOICE_ROOT, MODEL_DIR, configure_local_wetext_frontend,
    save_manifest, text_sha256, validate_speech,
)
from src.audiobook.cosyvoice import (
    PROMPT_PREFIX, SETTLED_SETTINGS, create_cosyvoice_model, file_sha256,
    git_head, load_cosyvoice_runtime, normalize_prompt_transcript,
    set_cosyvoice_random_seed, wav_info, write_pcm16_wav,
)

SEED = 2026091702
SWEEP = ROOT / "outputs/evaluation/cosyvoice_rng_sweep_2026-09-18_02-58-45/manifest.json"
GIANT = ROOT / "outputs/evaluation/cosyvoice_longform_ab/01_giant_seed_2026091702/benchmark_manifest.json"
GIANT_RUN = GIANT.parent / "milestone_d_runs/shentongzhe_ch01_longform_ab/giant_seed_2026091702"


def require(value, message):
    if not value:
        raise ValueError(message)


def pcm(path, start=0, count=None):
    with wave.open(str(path), "rb") as reader:
        require(reader.getnchannels() == 1 and reader.getsampwidth() == 2
                and reader.getframerate() == 24000, f"Unexpected PCM format: {path}")
        reader.setpos(start)
        data = reader.readframes(reader.getnframes() - start if count is None else count)
        if count is not None:
            require(len(data) == count * 2, f"Truncated PCM read: {path}")
        return data


def load_inputs():
    sweep = json.loads(SWEEP.read_text(encoding="utf-8"))
    giant = json.loads(GIANT.read_text(encoding="utf-8"))
    require(sweep["status"] == giant["status"] == "completed", "Historical run incomplete")
    require(giant["seed"] == SEED, "Wrong giant seed")
    clean = [item["normalized_text"] for item in sweep["units"]]
    cr = [item["text"] for item in giant["scenes"][0]["normalized_units"][:4]]
    require(len(clean) == len(cr) == 4, "Expected exactly four units")
    history = sorted((r for r in sweep["records"] if r["global_seed"] == SEED),
                     key=lambda r: r["unit_index"])
    require(len(history) == 4, "Historical sweep unit count differs")
    checks = []
    for index, (a, b, old) in enumerate(zip(clean, cr, history)):
        require("\r" not in a and "\n" not in a and "\n" not in b, "Unexpected line endings")
        require(b.replace("\r", "") == a, "Inputs differ beyond carriage returns")
        require(b.count("\r") == 4, "Expected four historical CRs per unit")
        require(text_sha256(a) == sweep["units"][index]["normalized_text_sha256"]
                == old["normalized_text_sha256"], "Historical clean text hash mismatch")
        require(text_sha256(b) == giant["scenes"][0]["normalized_units"][index]["text_sha256"],
                "Historical CR text hash mismatch")
        require(file_sha256(SWEEP.parent / old["output_path"]) == old["wav_sha256"],
                "Historical sweep WAV hash mismatch")
        checks.append({"unit_index": index + 1, "clean_length": len(a), "cr_length": len(b),
                       "cr_positions": [i for i, c in enumerate(b) if c == "\r"],
                       "equal_after_removing_only_cr": True})
    giant_wav = GIANT_RUN / giant["scenes"][0]["output_path"]
    require(file_sha256(giant_wav) == giant["scenes"][0]["wav_sha256"], "Giant WAV hash mismatch")
    return sweep, giant, history, {"A_clean": clean, "B_cr": cr}, checks, giant_wav


def new_directory(parent):
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / ("cosyvoice_cr_format_ab_" + datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f"))
    path.mkdir()
    return path


def execute(inputs, output_parent, completed_a=None):
    sweep, giant, history, conditions, checks, giant_wav = inputs
    run = new_directory(output_parent)
    path = run / "manifest.json"
    started = time.perf_counter()
    manifest = {
        "schema_version": 1, "experiment_id": "cosyvoice_cr_format_ab", "status": "initializing",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "seed_policy": "Once per four-unit condition, after model initialization; no per-unit reseeding",
        "runner_sha256": file_sha256(Path(__file__)),
        "historical_manifests": {"sweep": {"path": str(SWEEP), "sha256": file_sha256(SWEEP)},
                                 "giant": {"path": str(GIANT), "sha256": file_sha256(GIANT)}},
        "input_checks": checks, "settings": dict(SETTLED_SETTINGS),
        "conditions": {name: {"units": [{"text": text, "text_sha256": text_sha256(text)}
                                          for text in texts],
                               "combined_text_sha256": text_sha256("".join(texts))}
                       for name, texts in conditions.items()},
        "rng_resets": [], "records": [], "concatenated_outputs": [],
        "expected_counts": {"public_calls": 2, "synthesis_jobs": 8, "rng_resets": 2},
        "model_lifetime": "One initialization, resident across A then B; no extra warmup synthesis",
    }
    save_manifest(path, manifest)
    print("Output:", run, flush=True)
    try:
        if completed_a is not None:
            previous_path = completed_a / "manifest.json"
            previous_run = json.loads(previous_path.read_text(encoding="utf-8"))
            require(previous_run["seed"] == SEED and len(previous_run["records"]) == 4
                    and all(r["condition"] == "A_clean" for r in previous_run["records"]),
                    "Reuse requires exactly four completed A records and no B records")
            require(previous_run["conditions"]["A_clean"]["actual_synthesis_texts"] == conditions["A_clean"],
                    "Reused A inputs differ")
            require(previous_run["historical_manifests"] == manifest["historical_manifests"],
                    "Historical evidence changed")
            require(previous_run["rng_resets"] == [{"condition": "A_clean", "seed": SEED, "before_unit": 1}],
                    "Unexpected reused RNG policy")
            destination = run / "A_clean"
            destination.mkdir()
            for record in previous_run["records"]:
                source = completed_a / record["output_path"]
                require(file_sha256(source) == record["wav_sha256"] == record["historical_wav_sha256"],
                        "Reused A WAV differs")
                shutil.copy2(source, destination / source.name)
            source = completed_a / "A_clean/concatenated.wav"
            require(pcm(source) == b"".join(pcm(completed_a / r["output_path"])
                                           for r in previous_run["records"]), "Reused A assembly differs")
            shutil.copy2(source, destination / source.name)
            old_concat = next(r for r in sweep["concatenated_outputs"] if r["global_seed"] == SEED)
            manifest["records"] = previous_run["records"]
            manifest["conditions"]["A_clean"] = previous_run["conditions"]["A_clean"]
            manifest["rng_resets"] = previous_run["rng_resets"]
            manifest["concatenated_outputs"].append({
                "condition": "A_clean", "output_path": "A_clean/concatenated.wav",
                "audio": wav_info(source, 24000), "wav_sha256": file_sha256(source),
                "artificial_silence_ms": 0, "pcm_matches_ordered_unit_concatenation": True,
                "historical_wav_byte_identical": file_sha256(source) == old_concat["wav_sha256"],
            })
            manifest["reused_A"] = {"manifest": str(previous_path), "sha256": file_sha256(previous_path),
                                    "original_status": previous_run["status"],
                                    "original_error": previous_run.get("error"),
                                    "model_load_seconds": previous_run["model_load_seconds"],
                                    "import_seconds": previous_run["import_seconds"],
                                    "runtime": previous_run["runtime"], "provenance": previous_run["provenance"]}
            manifest["model_lifetime"] = "Separate model initialization for A and B after post-A comparison failure; model resident throughout each four-unit sequence; no extra synthesis"
        prompt = sweep["prompt"]
        prompt_wav = Path(prompt["wav"])
        prompt_file = Path(prompt["transcript_file"])
        transcript = normalize_prompt_transcript(prompt_file.read_text(encoding="utf-8-sig"))
        provenance = {"cosyvoice_git_head": git_head(COSYVOICE_ROOT),
                      "model_config_sha256": file_sha256(MODEL_DIR / "cosyvoice3.yaml"),
                      "prompt_wav_sha256": file_sha256(prompt_wav),
                      "prompt_transcript_sha256": text_sha256(transcript)}
        for key in ("cosyvoice_git_head", "model_config_sha256"):
            require(provenance[key] == sweep[key] == giant["backend_provenance"][key],
                    f"Historical provenance differs: {key}")
        require(provenance["prompt_wav_sha256"] == prompt["wav_sha256"]
                == giant["backend_provenance"]["prompt_wav_sha256"], "Narrator reference differs")
        require(transcript == prompt["transcript"] == giant["backend_provenance"]["prompt_transcript"],
                "Narrator transcript differs")
        require(dict(SETTLED_SETTINGS) == sweep["settings"] == giant["backend_provenance"]["settings"],
                "Settings differ")
        manifest["provenance"] = provenance
        if completed_a is not None:
            require(provenance == manifest["reused_A"]["provenance"], "A/B provenance differs")
        manifest["prompt"] = prompt
        imports_started = time.perf_counter()
        torch, torchaudio, auto_model = load_cosyvoice_runtime(COSYVOICE_ROOT)
        manifest["import_seconds"] = time.perf_counter() - imports_started
        require(torch.cuda.is_available(), "CUDA unavailable")
        manifest["runtime"] = {"python": platform.python_version(), "executable": sys.executable,
                               "torch": torch.__version__, "torchaudio": torchaudio.__version__,
                               "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
        load_started = time.perf_counter()
        model = create_cosyvoice_model(auto_model, MODEL_DIR, SETTLED_SETTINGS)
        manifest["model_load_seconds"] = time.perf_counter() - load_started
        manifest["frontend"] = configure_local_wetext_frontend(model.frontend)
        require(manifest["frontend"]["assets"] == sweep["frontend"]["assets"]
                == giant["frontend"]["assets"], "Frontend assets differ")
        for name, texts in conditions.items():
            observed = model.frontend.text_normalize("".join(texts), split=True)
            manifest["conditions"][name]["preflight_normalized_units"] = observed
            require(observed == texts, f"Frontend changed frozen {name} text/boundaries")
        save_manifest(path, manifest)
        original_frontend = model.frontend.frontend_zero_shot
        for name, texts in conditions.items():
            if completed_a is not None and name == "A_clean":
                continue
            directory = run / name
            directory.mkdir()
            actual = []

            def observed_frontend(*args, **kwargs):
                text = args[0] if args else kwargs["tts_text"]
                require(len(actual) < 4 and text == texts[len(actual)], "Actual synthesis input differs")
                actual.append(text)
                return original_frontend(*args, **kwargs)

            model.frontend.frontend_zero_shot = observed_frontend
            torch.cuda.synchronize()
            set_cosyvoice_random_seed(SEED)
            manifest["rng_resets"].append({"condition": name, "seed": SEED, "before_unit": 1})
            generator = iter(model.inference_zero_shot("".join(texts), PROMPT_PREFIX + transcript,
                                                      str(prompt_wav), stream=False, text_frontend=True))
            sequence_started = previous = time.perf_counter()
            speeches, elapsed = [], []
            for index in range(4):
                output = next(generator)
                now = time.perf_counter()
                speeches.append(validate_speech(output))
                elapsed.append(now - previous)
                previous = now
                print(f"{name} unit {index + 1}/4: {elapsed[-1]:.3f}s", flush=True)
            try:
                next(generator)
            except StopIteration:
                pass
            else:
                raise RuntimeError("Unexpected fifth yield")
            torch.cuda.synchronize()
            sequence_seconds = time.perf_counter() - sequence_started
            model.frontend.frontend_zero_shot = original_frontend
            require(actual == texts, "Incomplete observed synthesis inputs")
            manifest["conditions"][name].update(actual_synthesis_texts=actual,
                                                  sequence_seconds=sequence_seconds)
            for index, (text, speech, seconds) in enumerate(zip(texts, speeches, elapsed), 1):
                wav = directory / f"unit_{index:02d}.wav"
                write_pcm16_wav(torchaudio, wav, speech, model.sample_rate)
                audio = wav_info(wav, model.sample_rate)
                record = {"condition": name, "unit_index": index, "text_sha256": text_sha256(text),
                          "yield_elapsed_seconds": seconds, "audio": audio,
                          "output_path": str(wav.relative_to(run)), "wav_sha256": file_sha256(wav)}
                if name == "A_clean":
                    old = history[index - 1]
                    record.update(historical_wav_sha256=old["wav_sha256"],
                                  historical_wav_byte_identical=record["wav_sha256"] == old["wav_sha256"],
                                  historical_pcm_identical=pcm(wav) == pcm(SWEEP.parent / old["output_path"]))
                else:
                    clean_record = manifest["records"][index - 1]
                    record["matches_A_wav"] = record["wav_sha256"] == clean_record["wav_sha256"]
                    record["matches_A_pcm"] = pcm(wav) == pcm(run / clean_record["output_path"])
                manifest["records"].append(record)
            joined_path = directory / "concatenated.wav"
            write_pcm16_wav(torchaudio, joined_path, torch.cat(speeches, dim=1), model.sample_rate)
            audio = wav_info(joined_path, model.sample_rate)
            records = [r for r in manifest["records"] if r["condition"] == name]
            require(audio["frames"] == sum(r["audio"]["frames"] for r in records), "Assembly frame mismatch")
            require(pcm(joined_path) == b"".join(pcm(run / r["output_path"]) for r in records),
                    "Assembly PCM mismatch")
            joined = {"condition": name, "output_path": str(joined_path.relative_to(run)),
                      "audio": audio, "wav_sha256": file_sha256(joined_path),
                      "artificial_silence_ms": 0, "pcm_matches_ordered_unit_concatenation": True}
            if name == "A_clean":
                old = next(r for r in sweep["concatenated_outputs"] if r["global_seed"] == SEED)
                joined["historical_wav_byte_identical"] = joined["wav_sha256"] == old["wav_sha256"]
                joined["historical_comparison_basis"] = "Recorded SHA256; individual historical unit files verified separately"
            manifest["concatenated_outputs"].append(joined)
            save_manifest(path, manifest)

        cr_concat = run / "B_cr/concatenated.wav"
        frames = manifest["concatenated_outputs"][1]["audio"]["frames"]
        giant_prefix = pcm(giant_wav, count=frames)
        comparison_path = run / "giant_opening_prefix_same_B_duration.wav"
        with wave.open(str(comparison_path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(model.sample_rate)
            writer.writeframes(giant_prefix)
        comparison = {"historical_scene_wav": str(giant_wav), "historical_scene_sha256": file_sha256(giant_wav),
                      "historical_unit_frame_boundaries_available": False,
                      "comparison_method": "Compare scene PCM prefix at B concatenation length; no alignment inferred if unequal",
                      "prefix_frames": frames, "prefix_pcm_sha256": hashlib.sha256(giant_prefix).hexdigest(),
                      "B_pcm_sha256": hashlib.sha256(pcm(cr_concat)).hexdigest(),
                      "B_equals_giant_opening_pcm": pcm(cr_concat) == giant_prefix,
                      "listening_prefix_path": str(comparison_path.relative_to(run)),
                      "listening_prefix_wav_sha256": file_sha256(comparison_path)}
        comparison["per_B_unit_interval_pcm_equal"] = []
        offset = 0
        for record in manifest["records"][4:]:
            count = record["audio"]["frames"]
            comparison["per_B_unit_interval_pcm_equal"].append(
                pcm(run / record["output_path"]) == giant_prefix[offset * 2:(offset + count) * 2])
            offset += count
        manifest["giant_comparison"] = comparison
        manifest["actual_counts"] = {"public_calls": 2, "synthesis_jobs": len(manifest["records"]),
                                     "rng_resets": len(manifest["rng_resets"])}
        manifest["new_synthesis_jobs_this_invocation"] = 4 if completed_a is not None else 8
        require(manifest["actual_counts"] == manifest["expected_counts"], "Unexpected execution counts")
        manifest["status"] = "completed"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["total_wall_seconds"] = time.perf_counter() - started
        save_manifest(path, manifest)
        print("Completed:", run, flush=True)
        return 0
    except Exception as error:
        manifest.update(status="failed", error={"type": type(error).__name__, "message": str(error),
                                                 "traceback": traceback.format_exc()})
        save_manifest(path, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-parent", type=Path, default=ROOT / "outputs/evaluation")
    parser.add_argument("--reuse-completed-a", type=Path,
                        help="Reuse the verified four A WAVs after a post-generation comparison failure; synthesize B only")
    args = parser.parse_args()
    inputs = load_inputs()
    print(json.dumps({"seed": SEED, "input_checks": inputs[4], "conditions": inputs[3],
                      "expected_public_calls": 2, "expected_jobs": 8}, ensure_ascii=True, indent=2))
    if args.execute:
        return execute(inputs, args.output_parent.resolve(),
                       args.reuse_completed_a.resolve() if args.reuse_completed_a else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
