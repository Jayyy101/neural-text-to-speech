"""Milestone D2 orchestration for first-attempt scene generation."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

from .cosyvoice import file_sha256, wav_info
from .planning import build_plan


GENERATION_SCHEMA_VERSION = 2
ATTEMPT_ID = "attempt_001"


class GenerationError(ValueError):
    """Raised when an existing run cannot enter D2 generation."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def error_record(error):
    return {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    }


def save_manifest(path, manifest):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_planned_run(run_directory):
    run_directory = Path(run_directory).expanduser().resolve()
    manifest_path = run_directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GenerationError(f"Cannot read D1 manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise GenerationError("D1 manifest must be a JSON object.")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "planned":
        raise GenerationError("Generation requires an untouched schema-version 1 planned run.")
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise GenerationError("D1 manifest must contain at least one planned scene.")
    for index, scene in enumerate(scenes, 1):
        expected_id = f"scene_{index:04d}"
        if (not isinstance(scene, dict) or scene.get("id") != expected_id
                or scene.get("order") != index):
            raise GenerationError("D1 scenes must have stable IDs and contiguous order.")
        text = scene.get("narration_text")
        if not isinstance(text, str) or not text.strip():
            raise GenerationError(f"{expected_id} has invalid narration text.")
        if scene.get("text_sha256") != file_sha256_bytes(text.encode("utf-8")):
            raise GenerationError(f"{expected_id} narration text does not match its hash.")

    source = manifest.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("snapshot_path"), str):
        raise GenerationError("D1 manifest has invalid source metadata.")
    snapshot = (run_directory / source["snapshot_path"]).resolve()
    try:
        snapshot.relative_to(run_directory)
    except ValueError as error:
        raise GenerationError("Source snapshot path must stay inside the run directory.") from error
    if not snapshot.is_file() or file_sha256(snapshot) != source.get("sha256"):
        raise GenerationError("Source snapshot is missing or does not match its hash.")
    derived_plan = build_plan(snapshot.read_bytes())
    if (manifest.get("plan_hash") != derived_plan["plan_hash"]
            or scenes != derived_plan["scenes"]):
        raise GenerationError("D1 manifest does not match the deterministic source plan.")

    for scene in scenes:
        attempt_dir = run_directory / "scenes" / scene["id"] / ATTEMPT_ID
        if attempt_dir.exists():
            raise GenerationError(f"Generation output already exists: {attempt_dir}")
    return run_directory, manifest_path, manifest


def file_sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def generate_planned_run(run_directory, backend, clock=utc_now):
    """Generate attempt_001 for every scene, continuing after scene failures."""
    run_directory, manifest_path, manifest = load_planned_run(run_directory)
    started_at = clock()
    manifest["schema_version"] = GENERATION_SCHEMA_VERSION
    manifest["status"] = "generating"
    manifest["generation"] = {
        "status": "initializing",
        "started_at_utc": started_at,
        "configuration": backend.configuration(),
    }
    for scene in manifest["scenes"]:
        relative_output = (
            Path("scenes") / scene["id"] / ATTEMPT_ID / "generated.wav"
        ).as_posix()
        scene["generation"] = {
            "status": "pending",
            "attempt": {
                "id": ATTEMPT_ID,
                "output_path": relative_output,
            },
        }
    save_manifest(manifest_path, manifest)

    try:
        backend_metadata = backend.initialize()
    except Exception as error:
        manifest["status"] = "generation_failed"
        manifest["generation"].update({
            "status": "failed",
            "finished_at_utc": clock(),
            "error": error_record(error),
        })
        for scene in manifest["scenes"]:
            scene["generation"]["status"] = "not_run"
        save_manifest(manifest_path, manifest)
        return manifest

    expected_rate = backend_metadata.get("sample_rate_hz")
    if not isinstance(expected_rate, int) or expected_rate <= 0:
        error = ValueError("Backend metadata must provide a positive sample_rate_hz.")
        manifest["status"] = "generation_failed"
        manifest["generation"].update({
            "status": "failed", "finished_at_utc": clock(),
            "backend": backend_metadata, "error": error_record(error),
        })
        for scene in manifest["scenes"]:
            scene["generation"]["status"] = "not_run"
        save_manifest(manifest_path, manifest)
        return manifest

    manifest["generation"].pop("configuration")
    manifest["generation"].update({"status": "running", "backend": backend_metadata})
    save_manifest(manifest_path, manifest)

    failed = 0
    for scene in manifest["scenes"]:
        generation = scene["generation"]
        attempt = generation["attempt"]
        output_path = run_directory / attempt["output_path"]
        generation["status"] = "running"
        attempt["started_at_utc"] = clock()
        save_manifest(manifest_path, manifest)
        try:
            output_path.parent.mkdir(parents=True)
            result = backend.generate_scene(scene["narration_text"], output_path)
            audio = wav_info(output_path, expected_rate)
            attempt.update({
                "finished_at_utc": clock(),
                "wav_sha256": file_sha256(output_path),
                "audio": audio,
            })
            for key in (
                "cosyvoice_chunks", "inference_seconds", "rtf",
                "peak_torch_cuda_allocated_gib",
            ):
                if key in result:
                    attempt[key] = result[key]
            generation["status"] = "generated"
        except Exception as error:
            failed += 1
            generation["status"] = "failed"
            attempt.update({
                "finished_at_utc": clock(),
                "error": error_record(error),
            })
        save_manifest(manifest_path, manifest)

    manifest["status"] = "generated" if failed == 0 else "generation_failed"
    manifest["generation"].update({
        "status": manifest["status"],
        "finished_at_utc": clock(),
        "summary": {
            "generated_scenes": len(manifest["scenes"]) - failed,
            "failed_scenes": failed,
            "total_scenes": len(manifest["scenes"]),
        },
    })
    save_manifest(manifest_path, manifest)
    return manifest
