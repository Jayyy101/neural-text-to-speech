"""Audiobook scene generation, recovery, and targeted regeneration."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import traceback
import wave

from .cosyvoice import file_sha256, wav_info
from .planning import build_plan


GENERATION_SCHEMA_VERSION = 4
FIRST_ATTEMPT_ID = "attempt_001"
ATTEMPT_PATTERN = re.compile(r"^attempt_([0-9]{3,})$")
REPAIR_PATTERN = re.compile(r"^repair_([0-9]{3,})$")
PLANNING_SCENE_KEYS = (
    "id", "order", "source_span", "narration_text", "text_sha256",
)
ATTEMPT_RESULT_KEYS = (
    "cosyvoice_chunks", "inference_seconds", "rtf",
    "peak_torch_cuda_allocated_gib",
)


class GenerationError(ValueError):
    """Raised when persisted generation state is unsafe or inconsistent."""


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


def read_manifest(run_directory):
    run_directory = Path(run_directory).expanduser().resolve()
    manifest_path = run_directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GenerationError(f"Cannot read audiobook manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise GenerationError("Audiobook manifest must be a JSON object.")
    return run_directory, manifest_path, manifest


def file_sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def validate_plan_identity(run_directory, manifest):
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise GenerationError("Manifest must contain at least one planned scene.")
    planned_scenes = []
    for index, scene in enumerate(scenes, 1):
        expected_id = f"scene_{index:04d}"
        if (not isinstance(scene, dict) or scene.get("id") != expected_id
                or scene.get("order") != index):
            raise GenerationError("Scenes must have stable IDs and contiguous order.")
        text = scene.get("narration_text")
        if not isinstance(text, str) or not text.strip():
            raise GenerationError(f"{expected_id} has invalid narration text.")
        if scene.get("text_sha256") != file_sha256_bytes(text.encode("utf-8")):
            raise GenerationError(f"{expected_id} narration text does not match its hash.")
        try:
            planned_scenes.append({key: scene[key] for key in PLANNING_SCENE_KEYS})
        except KeyError as error:
            raise GenerationError(f"{expected_id} is missing D1 planning metadata.") from error

    source = manifest.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("snapshot_path"), str):
        raise GenerationError("Manifest has invalid source metadata.")
    snapshot = (run_directory / source["snapshot_path"]).resolve()
    try:
        snapshot.relative_to(run_directory)
    except ValueError as error:
        raise GenerationError("Source snapshot path must stay inside the run directory.") from error
    if not snapshot.is_file() or file_sha256(snapshot) != source.get("sha256"):
        raise GenerationError("Source snapshot is missing or does not match its hash.")
    derived_plan = build_plan(snapshot.read_bytes())
    if (manifest.get("plan_hash") != derived_plan["plan_hash"]
            or planned_scenes != derived_plan["scenes"]):
        raise GenerationError("Manifest does not match the deterministic source plan.")
    return scenes


def attempt_output_path(scene_id, attempt_id):
    return (
        Path("scenes") / scene_id / attempt_id / "generated.wav"
    ).as_posix()


def load_planned_run(run_directory):
    run_directory, manifest_path, manifest = read_manifest(run_directory)
    if manifest.get("schema_version") != 1 or manifest.get("status") != "planned":
        raise GenerationError("Generation requires an untouched schema-version 1 planned run.")
    scenes = validate_plan_identity(run_directory, manifest)
    for scene in scenes:
        attempt_dir = run_directory / "scenes" / scene["id"] / FIRST_ATTEMPT_ID
        if attempt_dir.exists():
            raise GenerationError(f"Generation output already exists: {attempt_dir}")
    return run_directory, manifest_path, manifest


def migrate_d2_scene(scene):
    generation = scene.get("generation")
    if not isinstance(generation, dict):
        raise GenerationError(f"{scene['id']} has invalid generation state.")
    if "attempt" in generation:
        attempt = generation.pop("attempt")
        if not isinstance(attempt, dict):
            raise GenerationError(f"{scene['id']} has invalid D2 attempt state.")
        old_status = generation.get("status")
        attempt["status"] = (
            "generated" if old_status == "generated"
            else "failed" if old_status == "failed"
            else old_status if old_status in {"pending", "running", "not_run"}
            else "not_run"
        )
        generation["attempts"] = [attempt]
        generation["selected_attempt_id"] = (
            attempt.get("id") if attempt["status"] == "generated" else None
        )
    if not isinstance(generation.get("attempts"), list):
        raise GenerationError(f"{scene['id']} must contain an attempts list.")
    if "selected_attempt_id" not in generation:
        raise GenerationError(f"{scene['id']} must identify its selected attempt.")
    return generation


def validate_attempt_history(scene):
    generation = migrate_d2_scene(scene)
    previous_number = 0
    seen = set()
    for attempt in generation["attempts"]:
        if not isinstance(attempt, dict):
            raise GenerationError(f"{scene['id']} contains invalid attempt metadata.")
        attempt_id = attempt.get("id")
        match = ATTEMPT_PATTERN.fullmatch(attempt_id) if isinstance(attempt_id, str) else None
        if not match:
            raise GenerationError(f"{scene['id']} contains an invalid attempt ID.")
        number = int(match.group(1))
        if number <= previous_number or attempt_id in seen:
            raise GenerationError(f"{scene['id']} attempt IDs must increase monotonically.")
        previous_number = number
        seen.add(attempt_id)
        if attempt.get("output_path") != attempt_output_path(scene["id"], attempt_id):
            raise GenerationError(f"{scene['id']} attempt output path is not deterministic.")
        if attempt.get("status") not in {
            "pending", "running", "not_run", "generated", "failed",
        }:
            raise GenerationError(f"{scene['id']} contains an invalid attempt status.")
    selected = generation["selected_attempt_id"]
    if selected is not None and selected not in seen:
        raise GenerationError(f"{scene['id']} selected attempt does not exist.")
    return generation


def validate_repair_history(scene):
    repair_state = scene.get("repair")
    if repair_state is None:
        return
    if not isinstance(repair_state, dict) or not isinstance(
            repair_state.get("repairs"), list):
        raise GenerationError(f"{scene['id']} has invalid repair state.")
    previous_number = 0
    seen = set()
    for repair in repair_state["repairs"]:
        repair_id = repair.get("id") if isinstance(repair, dict) else None
        match = REPAIR_PATTERN.fullmatch(repair_id) if isinstance(repair_id, str) else None
        if not match:
            raise GenerationError(f"{scene['id']} contains an invalid repair ID.")
        number = int(match.group(1))
        if number <= previous_number or repair_id in seen:
            raise GenerationError(f"{scene['id']} repair IDs must increase monotonically.")
        previous_number = number
        seen.add(repair_id)
    selected = repair_state.get("selected_repair_id")
    if selected is not None and selected not in seen:
        raise GenerationError(f"{scene['id']} selected repair does not exist.")


def load_generation_run(run_directory):
    run_directory, manifest_path, manifest = read_manifest(run_directory)
    if manifest.get("schema_version") not in {2, 3, GENERATION_SCHEMA_VERSION}:
        raise GenerationError("Operation requires a schema-version 2, 3, or 4 generation run.")
    if not isinstance(manifest.get("generation"), dict):
        raise GenerationError("Manifest has invalid run generation metadata.")
    scenes = validate_plan_identity(run_directory, manifest)
    for scene in scenes:
        validate_attempt_history(scene)
        validate_repair_history(scene)
    manifest["schema_version"] = GENERATION_SCHEMA_VERSION
    return run_directory, manifest_path, manifest


def validate_attempt_artifact(run_directory, attempt, expected_rate):
    if attempt.get("status") != "generated":
        return False, "Attempt was not recorded as successfully generated."
    output_path = (run_directory / attempt["output_path"]).resolve()
    try:
        output_path.relative_to(run_directory)
    except ValueError:
        return False, "Attempt output path leaves the run directory."
    try:
        audio = wav_info(output_path, expected_rate)
        digest = file_sha256(output_path)
    except (OSError, EOFError, ValueError, wave.Error) as error:
        return False, str(error)
    if digest != attempt.get("wav_sha256"):
        return False, "Generated WAV does not match its recorded SHA-256."
    if audio != attempt.get("audio"):
        return False, "Generated WAV metadata does not match the manifest."
    return True, None


def selected_attempt(scene):
    generation = scene["generation"]
    selected_id = generation["selected_attempt_id"]
    if selected_id is None:
        return None
    for attempt in generation["attempts"]:
        if attempt["id"] == selected_id:
            return attempt
    raise GenerationError(f"{scene['id']} selected attempt does not exist.")


def mark_assembly_stale(manifest, reason):
    assembly = manifest.get("assembly")
    if isinstance(assembly, dict) and assembly.get("status") == "assembled":
        assembly["status"] = "stale"
        assembly["stale_reason"] = reason


def selected_attempt_changed(manifest, scene, previous_attempt_id, next_attempt_id):
    if previous_attempt_id == next_attempt_id:
        return
    repair_state = scene.get("repair")
    if isinstance(repair_state, dict):
        repair_state["selected_repair_id"] = None
    mark_assembly_stale(
        manifest,
        f"{scene['id']} selected attempt changed from "
        f"{previous_attempt_id or 'none'} to {next_attempt_id or 'none'}.",
    )


def invalidate_selected(scene, attempt, reason, manifest=None):
    previous_attempt_id = scene["generation"]["selected_attempt_id"]
    attempt["artifact_status"] = "invalid"
    attempt["artifact_error"] = reason
    scene["generation"]["selected_attempt_id"] = None
    scene["generation"]["status"] = "incomplete"
    if manifest is not None:
        selected_attempt_changed(manifest, scene, previous_attempt_id, None)


def next_attempt_id(scene):
    attempts = scene["generation"]["attempts"]
    number = 1 if not attempts else int(ATTEMPT_PATTERN.fullmatch(attempts[-1]["id"]).group(1)) + 1
    return f"attempt_{number:03d}"


def expected_rate_from_manifest(manifest, required):
    backend = manifest["generation"].get("backend")
    rate = backend.get("sample_rate_hz") if isinstance(backend, dict) else None
    if isinstance(rate, int) and rate > 0:
        return rate
    if required:
        raise GenerationError("Generation manifest has no valid backend sample rate.")
    return None


def verify_backend_configuration(manifest, backend):
    generation = manifest["generation"]
    recorded = generation.get("backend") or generation.get("configuration")
    if not isinstance(recorded, dict):
        raise GenerationError("Generation manifest has no recorded backend configuration.")
    requested = backend.configuration()
    if any(recorded.get(key) != value for key, value in requested.items()):
        raise GenerationError("Requested backend configuration differs from the existing run.")


def initialize_backend(manifest, backend):
    verify_backend_configuration(manifest, backend)
    metadata = backend.initialize()
    rate = metadata.get("sample_rate_hz")
    if not isinstance(rate, int) or rate <= 0:
        raise ValueError("Backend metadata must provide a positive sample_rate_hz.")
    recorded_rate = expected_rate_from_manifest(manifest, required=False)
    if recorded_rate is not None and rate != recorded_rate:
        raise GenerationError("Backend sample rate differs from the existing run.")
    if "backend" not in manifest["generation"]:
        if "error" in manifest["generation"]:
            manifest["generation"]["initialization_failure"] = {
                "finished_at_utc": manifest["generation"].pop("finished_at_utc", None),
                "error": manifest["generation"].pop("error"),
            }
        manifest["generation"]["backend"] = metadata
        manifest["generation"].pop("configuration", None)
    return rate


def validate_next_attempt_paths(run_directory, scenes):
    for scene in scenes:
        attempt_id = next_attempt_id(scene)
        attempt_dir = run_directory / "scenes" / scene["id"] / attempt_id
        if attempt_dir.exists():
            raise GenerationError(
                f"Untracked next-attempt output already exists: {attempt_dir}"
            )


def run_attempt(run_directory, manifest_path, manifest, scene, backend, expected_rate, clock):
    generation = scene["generation"]
    previous_attempt_id = generation["selected_attempt_id"]
    attempt_id = next_attempt_id(scene)
    output_path = run_directory / attempt_output_path(scene["id"], attempt_id)
    attempt = {
        "id": attempt_id,
        "status": "running",
        "output_path": output_path.relative_to(run_directory).as_posix(),
        "started_at_utc": clock(),
    }
    generation["attempts"].append(attempt)
    generation["status"] = "running"
    save_manifest(manifest_path, manifest)
    try:
        output_path.parent.mkdir(parents=True)
        result = backend.generate_scene(scene["narration_text"], output_path)
        audio = wav_info(output_path, expected_rate)
        attempt.update({
            "status": "generated",
            "finished_at_utc": clock(),
            "wav_sha256": file_sha256(output_path),
            "audio": audio,
        })
        for key in ATTEMPT_RESULT_KEYS:
            if key in result:
                attempt[key] = result[key]
        generation["selected_attempt_id"] = attempt_id
        generation["status"] = "generated"
        selected_attempt_changed(manifest, scene, previous_attempt_id, attempt_id)
        succeeded = True
    except Exception as error:
        attempt.update({
            "status": "failed",
            "finished_at_utc": clock(),
            "error": error_record(error),
        })
        generation["status"] = (
            "generated" if generation["selected_attempt_id"] is not None else "failed"
        )
        succeeded = False
    save_manifest(manifest_path, manifest)
    return attempt, succeeded


def refresh_run_status(manifest):
    generated = sum(
        scene["generation"]["selected_attempt_id"] is not None
        for scene in manifest["scenes"]
    )
    total = len(manifest["scenes"])
    manifest["status"] = "generated" if generated == total else "generation_failed"
    manifest["generation"].update({
        "status": manifest["status"],
        "summary": {
            "generated_scenes": generated,
            "failed_scenes": total - generated,
            "total_scenes": total,
        },
    })


def start_operation(manifest, operation_type, clock, scene_id=None):
    operation = {
        "type": operation_type,
        "status": "running",
        "started_at_utc": clock(),
    }
    if scene_id is not None:
        operation["scene_id"] = scene_id
    manifest["generation"]["last_operation"] = operation
    return operation


def finish_operation(operation, attempted, succeeded, clock, error=None):
    operation.update({
        "status": "failed" if error is not None or succeeded < attempted else (
            "no_work" if attempted == 0 else "succeeded"
        ),
        "finished_at_utc": clock(),
        "attempted_scenes": attempted,
        "successful_attempts": succeeded,
        "failed_attempts": attempted - succeeded,
    })
    if error is not None:
        operation["error"] = error_record(error)


def generate_planned_run(run_directory, backend, clock=utc_now):
    """Generate attempt_001 for every scene, continuing after scene failures."""
    run_directory, manifest_path, manifest = load_planned_run(run_directory)
    manifest["schema_version"] = GENERATION_SCHEMA_VERSION
    manifest["status"] = "generating"
    manifest["generation"] = {
        "status": "initializing",
        "started_at_utc": clock(),
        "configuration": backend.configuration(),
    }
    for scene in manifest["scenes"]:
        scene["generation"] = {
            "status": "pending",
            "selected_attempt_id": None,
            "attempts": [],
        }
    save_manifest(manifest_path, manifest)

    try:
        metadata = backend.initialize()
        expected_rate = metadata.get("sample_rate_hz")
        if not isinstance(expected_rate, int) or expected_rate <= 0:
            raise ValueError("Backend metadata must provide a positive sample_rate_hz.")
    except Exception as error:
        manifest["status"] = "generation_failed"
        manifest["generation"].update({
            "status": "failed", "finished_at_utc": clock(),
            "error": error_record(error),
        })
        for scene in manifest["scenes"]:
            scene["generation"]["status"] = "not_run"
        save_manifest(manifest_path, manifest)
        return manifest

    manifest["generation"].pop("configuration")
    manifest["generation"].update({"status": "running", "backend": metadata})
    save_manifest(manifest_path, manifest)
    for scene in manifest["scenes"]:
        run_attempt(
            run_directory, manifest_path, manifest, scene, backend, expected_rate, clock
        )
    refresh_run_status(manifest)
    manifest["generation"]["finished_at_utc"] = clock()
    save_manifest(manifest_path, manifest)
    return manifest


def resume_generation(run_directory, backend, clock=utc_now):
    """Make one new attempt for each scene without a valid selected artifact."""
    run_directory, manifest_path, manifest = load_generation_run(run_directory)
    expected_rate = expected_rate_from_manifest(manifest, required=False)
    targets = []
    for scene in manifest["scenes"]:
        selected = selected_attempt(scene)
        if selected is not None:
            if expected_rate is None:
                raise GenerationError("Cannot validate selected audio without a sample rate.")
            valid, reason = validate_attempt_artifact(run_directory, selected, expected_rate)
            if valid:
                continue
            invalidate_selected(scene, selected, reason, manifest)
        targets.append(scene)

    validate_next_attempt_paths(run_directory, targets)
    operation = start_operation(manifest, "resume", clock)
    save_manifest(manifest_path, manifest)
    if not targets:
        finish_operation(operation, 0, 0, clock)
        refresh_run_status(manifest)
        save_manifest(manifest_path, manifest)
        return manifest

    try:
        expected_rate = initialize_backend(manifest, backend)
    except Exception as error:
        finish_operation(operation, 0, 0, clock, error=error)
        refresh_run_status(manifest)
        save_manifest(manifest_path, manifest)
        return manifest

    succeeded = 0
    for scene in targets:
        _, success = run_attempt(
            run_directory, manifest_path, manifest, scene, backend, expected_rate, clock
        )
        succeeded += int(success)
    finish_operation(operation, len(targets), succeeded, clock)
    refresh_run_status(manifest)
    save_manifest(manifest_path, manifest)
    return manifest


def regenerate_scene(run_directory, scene_id, backend, clock=utc_now):
    """Make exactly one new attempt for one explicitly requested scene."""
    run_directory, manifest_path, manifest = load_generation_run(run_directory)
    requested = next(
        (scene for scene in manifest["scenes"] if scene["id"] == scene_id), None
    )
    if requested is None:
        raise GenerationError(f"Scene does not exist: {scene_id}")

    expected_rate = expected_rate_from_manifest(manifest, required=False)
    for scene in manifest["scenes"]:
        selected = selected_attempt(scene)
        if selected is None:
            continue
        if expected_rate is None:
            raise GenerationError("Cannot validate selected audio without a sample rate.")
        valid, reason = validate_attempt_artifact(run_directory, selected, expected_rate)
        if valid:
            continue
        if scene is not requested:
            raise GenerationError(
                f"{scene['id']} has an invalid selected artifact; run resume first."
            )
        invalidate_selected(scene, selected, reason, manifest)

    validate_next_attempt_paths(run_directory, [requested])
    operation = start_operation(manifest, "regenerate", clock, scene_id=scene_id)
    save_manifest(manifest_path, manifest)
    try:
        expected_rate = initialize_backend(manifest, backend)
    except Exception as error:
        finish_operation(operation, 0, 0, clock, error=error)
        refresh_run_status(manifest)
        save_manifest(manifest_path, manifest)
        return manifest

    _, success = run_attempt(
        run_directory, manifest_path, manifest, requested, backend, expected_rate, clock
    )
    finish_operation(operation, 1, int(success), clock)
    refresh_run_status(manifest)
    save_manifest(manifest_path, manifest)
    return manifest
