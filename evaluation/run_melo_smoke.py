"""Opt-in real-model smoke test; --help needs only the Python standard library."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import wave


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "evaluation/inputs/mandarin_diagnostics.json"


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_run_directory(parent, now=None):
    """Readable local timestamp; an exclusive mkdir prevents same-second reuse."""
    parent.mkdir(parents=True, exist_ok=True)
    stem = "melo_baseline_" + (now or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    number = 1
    while True:
        candidate = parent / (stem if number == 1 else f"{stem}_{number:02d}")
        try:
            candidate.mkdir()
            return candidate.resolve()
        except FileExistsError:
            number += 1


def plan_trials(cases, settings, repeats):
    records = []
    for case in cases:
        label = "mixed_language" if case["id"] == "mixed_text" else case["id"]
        for repeat in range(1, repeats + 1):
            records.append({
                "case_id": case["id"], "text": case["text"], "repeat": repeat,
                "settings": dict(settings), "status": "pending",
                "expected_output_filename": f"{label}_{repeat:02d}.wav",
                "output_filename": None,
                "trial_directory": f"_trials/{case['id']}/{repeat:02d}",
                "trial_elapsed_seconds": None,
            })
    return records


def run_case(synthesize, case, settings, trial_dir, output_path):
    """Isolate relative output paths without modifying the synthesis function."""
    trial_dir = trial_dir.resolve()
    trial_dir.mkdir(parents=True, exist_ok=False)
    previous_cwd = Path.cwd()
    started = time.perf_counter()
    try:
        os.chdir(trial_dir)
        result = synthesize(text=case["text"], **settings)
        audio_path = Path(result["output_path"]).resolve()
        audio_path.relative_to(trial_dir)  # Only organize audio from this new trial.
        with wave.open(str(audio_path), "rb") as audio:
            frames = audio.getnframes()
            rate = audio.getframerate()
            if frames <= 0 or rate <= 0:
                raise ValueError("Synthesis returned an empty WAV.")
            audio_info = {
                "sample_rate_hz": rate,
                "channels": audio.getnchannels(),
                "sample_width_bytes": audio.getsampwidth(),
                "frames": frames,
                "duration_seconds": frames / rate,
            }
        digest = file_hash(audio_path)
        if output_path.exists():
            raise FileExistsError(output_path)
        audio_path.rename(output_path)
        return {
            "backend_result": result,
            "output_filename": output_path.name,
            "wav_sha256": digest,
            "audio": audio_info,
            "call_and_wav_check_seconds": time.perf_counter() - started,
        }
    finally:
        os.chdir(previous_cwd)


def run_trials(synthesize, run_dir, manifest, save):
    """Record individual errors and continue, including subsequent repetitions."""
    for record in manifest["records"]:
        record["status"] = "running"
        record["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        started = time.perf_counter()
        try:
            record.update(run_case(
                synthesize, record, record["settings"],
                run_dir / record["trial_directory"],
                run_dir / record["expected_output_filename"],
            ))
            record["status"] = "passed_wav_check"
        except Exception as error:
            record["status"] = "failed"
            record["error"] = {
                "type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(),
            }
        finally:
            record["trial_elapsed_seconds"] = time.perf_counter() - started
            record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            save()
        print(f"{record['case_id']} repetition {record['repeat']:02d}: {record['status']}")


def summarize(manifest):
    records = manifest["records"]
    case_ids = list(dict.fromkeys(r["case_id"] for r in records))
    passed_cases, failed_cases, incomplete_cases = [], [], []
    for case_id in case_ids:
        states = [r["status"] for r in records if r["case_id"] == case_id]
        if all(s == "passed_wav_check" for s in states):
            passed_cases.append(case_id)
        elif "failed" in states:
            failed_cases.append(case_id)
        else:
            incomplete_cases.append(case_id)
    summary = {
        "passed_trials": sum(r["status"] == "passed_wav_check" for r in records),
        "failed_trials": sum(r["status"] == "failed" for r in records),
        "not_completed_trials": sum(r["status"] not in ("passed_wav_check", "failed") for r in records),
        "passed_cases": passed_cases, "failed_cases": failed_cases,
        "incomplete_cases": incomplete_cases,
    }
    manifest["summary"] = summary
    print(f"Summary: {summary['passed_trials']} passed, {summary['failed_trials']} failed, "
          f"{summary['not_completed_trials']} not completed trials.")
    for label in ("passed_cases", "failed_cases", "incomplete_cases"):
        print(label.replace("_", " ").capitalize() + ": " + (", ".join(summary[label]) or "none"))
    return int(bool(summary["failed_trials"] or summary["not_completed_trials"]))


def open_run_folder(run_dir):
    """Optional convenience; an Explorer failure must not alter trial results."""
    try:
        if os.name != "nt":
            return "--open-output is supported only on Windows."
        os.startfile(str(run_dir))
    except Exception as error:
        return f"Could not open run folder: {type(error).__name__}: {error}"
    return None


def main(argv=None):
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case", choices=[c["id"] for c in corpus["cases"]])
    selection.add_argument("--all", action="store_true", help="Run all seven cases.")
    parser.add_argument("--repeat", type=int, default=2, help="Calls per case (default: 2).")
    parser.add_argument("--open-output", action="store_true", help="Open the finished run folder in Windows Explorer.")
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be positive")

    cases = [c for c in corpus["cases"] if args.all or c["id"] == (args.case or "narration")]
    settings = {"language": corpus["language"], "speaker_name": corpus["speaker"], "speed": corpus["speed"]}
    run_dir = create_run_directory(ROOT / "outputs/evaluation")
    manifest = {
        "schema_version": 2,
        "run_id": run_dir.name,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "directory_timestamp_timezone": "local",
        "status": "starting",
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "settings": settings,
        "corpus_id": corpus["corpus_id"],
        "corpus_sha256": file_hash(CORPUS),
        "backend_sha256": file_hash(ROOT / "src/generate_melo.py"),
        "baseline_record_sha256": file_hash(ROOT / "evaluation/baseline/melo-environment.json"),
        "repeats": args.repeat,
        "records": plan_trials(cases, settings, args.repeat),
        "notes": [
            "Baseline-record hash is provenance, not verification of current package/model equality.",
            "No seed is forced; waveform equality is not expected.",
            "Backend inference_time excludes model loading. Call timing includes WAV checking.",
            "Human listening and complete asset provenance remain separate checks.",
        ],
    }
    manifest_path = run_dir / "manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save()
    print("Results:", run_dir)
    # Limit HF/Transformers to cached assets; do not silently fetch newer weights.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    manifest["environment"] = {name: os.environ.get(name) for name in (
        "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HOME", "HF_HUB_CACHE",
        "CUDA_VISIBLE_DEVICES", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "TORCH_FORCE_WEIGHTS_ONLY_LOAD",
    )}
    try:
        import_started = time.perf_counter()
        import torch

        manifest["packages"] = {name: metadata.version(name) for name in (
            "melotts", "torch", "torchaudio", "transformers", "numpy", "soundfile",
        )}
        manifest["cuda"] = {"available": torch.cuda.is_available(), "build": torch.version.cuda}
        if not torch.cuda.is_available():
            raise RuntimeError("This GPU smoke test requires CUDA. Check that the melo environment is active.")
        manifest["cuda"]["device"] = torch.cuda.get_device_name(0)
        manifest["melotts_install_source"] = json.loads(metadata.distribution("melotts").read_text("direct_url.json") or "null")
        sys.path.insert(0, str(ROOT / "src"))
        from generate_melo import synthesize_melo

        manifest["import_seconds"] = time.perf_counter() - import_started
        manifest["status"] = "running"
        save()
        run_trials(synthesize_melo, run_dir, manifest, save)
        manifest["status"] = ("completed_with_failures" if any(r["status"] == "failed" for r in manifest["records"])
                              else "passed_wav_checks_listening_pending")
    except Exception as error:
        manifest["status"] = "aborted"
        manifest["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        for record in manifest["records"]:
            if record["status"] == "pending":
                record["status"] = "not_run"
                record["reason"] = "Run aborted; see top-level error."
        print(manifest["error"]["traceback"], file=sys.stderr)
    exit_code = summarize(manifest)
    if manifest["status"] == "aborted":
        exit_code = 1
    manifest["exit_code"] = exit_code
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print("Results and manifest:", run_dir)
    if args.open_output:
        warning = open_run_folder(run_dir)
        if warning:
            manifest["open_output_warning"] = warning
            save()
            print(warning, file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
