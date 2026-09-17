"""Read-only application layer for inspecting Milestone D audiobook runs."""

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import wave

from .audiobook.cosyvoice import file_sha256, wav_info
from .audiobook.pipeline import (
    GENERATION_SCHEMA_VERSION,
    GenerationError,
    expected_rate_from_manifest,
    load_generation_run,
    read_manifest,
    validate_attempt_artifact,
    validate_plan_identity,
)
from .audiobook.postprocessing import path_inside_run, resolve_scene_artifact


@dataclass(frozen=True)
class AttemptInspection:
    id: str
    status: str
    selected: bool
    seed: object
    seed_recorded: bool
    random_policy: str | None
    output_path: str | None
    audio: dict | None
    error: dict | None
    artifact_valid: bool | None
    artifact_error: str | None
    duplicate_of_attempt_id: str | None


@dataclass(frozen=True)
class SceneInspection:
    id: str
    order: int
    text: str
    source_span: dict
    generation_status: str
    selected_attempt_id: str | None
    attempts: tuple[AttemptInspection, ...]
    selected_repair_id: str | None
    selected_repair: dict | None
    resolved_artifact_type: str | None
    resolved_artifact_id: str | None
    resolved_audio_path: Path | None
    resolution_error: str | None


@dataclass(frozen=True)
class AssemblyInspection:
    status: str
    playable: bool
    audio_path: Path | None
    audio: dict | None
    error: str | None
    stale_reason: str | None


@dataclass(frozen=True)
class RunInspection:
    run_directory: Path
    schema_version: int
    chapter_id: str
    run_id: str
    created_at_utc: str
    source: dict
    source_path: Path
    plan_hash: str
    run_status: str
    generation_status: str
    generation_summary: dict | None
    latest_operation: dict | None
    assembly: AssemblyInspection
    scenes: tuple[SceneInspection, ...]


def _attempt_inspection(run_directory, attempt, selected, expected_rate):
    valid = None
    reason = None
    if expected_rate is not None and attempt.get("status") == "generated":
        valid, reason = validate_attempt_artifact(run_directory, attempt, expected_rate)
    random_state = attempt.get("random_state")
    seed_recorded = isinstance(random_state, dict) and "seed" in random_state
    return AttemptInspection(
        id=attempt.get("id", "(unknown)"),
        status=attempt.get("status", "unknown"),
        selected=selected,
        seed=random_state.get("seed") if seed_recorded else None,
        seed_recorded=seed_recorded,
        random_policy=random_state.get("policy") if isinstance(random_state, dict) else None,
        output_path=attempt.get("output_path"),
        audio=attempt.get("audio") if isinstance(attempt.get("audio"), dict) else None,
        error=attempt.get("error") if isinstance(attempt.get("error"), dict) else None,
        artifact_valid=valid,
        artifact_error=reason,
        duplicate_of_attempt_id=attempt.get("duplicate_of_attempt_id"),
    )


def _scene_inspection(run_directory, manifest, scene, expected_rate):
    generation = scene.get("generation")
    if not isinstance(generation, dict):
        return SceneInspection(
            id=scene["id"], order=scene["order"], text=scene["narration_text"],
            source_span=scene["source_span"], generation_status="not_started",
            selected_attempt_id=None, attempts=(), selected_repair_id=None,
            selected_repair=None, resolved_artifact_type=None,
            resolved_artifact_id=None, resolved_audio_path=None,
            resolution_error="Scene has not been generated.",
        )

    selected_id = generation.get("selected_attempt_id")
    attempts = tuple(
        _attempt_inspection(run_directory, attempt, attempt.get("id") == selected_id, expected_rate)
        for attempt in generation["attempts"]
    )
    repair_state = scene.get("repair")
    selected_repair_id = (
        repair_state.get("selected_repair_id") if isinstance(repair_state, dict) else None
    )
    selected_repair = None
    if selected_repair_id is not None:
        selected_repair = next(
            (item for item in repair_state["repairs"] if item.get("id") == selected_repair_id),
            None,
        )

    resolved_type = None
    resolved_id = None
    resolved_path = None
    resolution_error = None
    try:
        relative_path, resolved_type, resolved_id, _ = resolve_scene_artifact(
            run_directory, manifest, scene
        )
        resolved_path = path_inside_run(run_directory, relative_path, "Scene artifact")
    except (GenerationError, OSError, EOFError, ValueError, wave.Error) as error:
        resolution_error = str(error)

    return SceneInspection(
        id=scene["id"], order=scene["order"], text=scene["narration_text"],
        source_span=scene["source_span"],
        generation_status=generation.get("status", "unknown"),
        selected_attempt_id=selected_id, attempts=attempts,
        selected_repair_id=selected_repair_id, selected_repair=selected_repair,
        resolved_artifact_type=resolved_type, resolved_artifact_id=resolved_id,
        resolved_audio_path=resolved_path, resolution_error=resolution_error,
    )


def _assembly_inspection(run_directory, manifest, scenes):
    assembly = manifest.get("assembly")
    if assembly is None:
        return AssemblyInspection("not_assembled", False, None, None, None, None)
    if not isinstance(assembly, dict):
        return AssemblyInspection(
            "invalid", False, None, None, "Assembly state is not an object.", None
        )
    status = assembly.get("status", "unknown")
    stale_reason = assembly.get("stale_reason")
    if status != "assembled":
        error = stale_reason or f"Assembly status is {status}."
        return AssemblyInspection(status, False, None, assembly.get("audio"), error, stale_reason)

    try:
        path = path_inside_run(run_directory, assembly.get("output_path"), "Assembly output")
        recorded_audio = assembly.get("audio")
        if not isinstance(recorded_audio, dict):
            raise GenerationError("Assembly has no valid audio metadata.")
        rate = recorded_audio.get("sample_rate_hz")
        if not isinstance(rate, int) or rate <= 0:
            raise GenerationError("Assembly has no valid sample rate.")
        actual_audio = wav_info(path, rate)
        if actual_audio != recorded_audio:
            raise GenerationError("Final chapter WAV metadata does not match the manifest.")
        if file_sha256(path) != assembly.get("wav_sha256"):
            raise GenerationError("Final chapter WAV does not match its recorded SHA-256.")
        recorded_scenes = assembly.get("scenes")
        if not isinstance(recorded_scenes, list) or len(recorded_scenes) != len(scenes):
            raise GenerationError("Assembly scene records do not match the current plan.")
        for scene, record in zip(scenes, recorded_scenes):
            if not isinstance(record, dict):
                raise GenerationError("Assembly contains an invalid scene record.")
            if scene.resolution_error is not None:
                raise GenerationError(
                    f"{scene.id} current selection is invalid: {scene.resolution_error}"
                )
            expected = {
                "scene_id": scene.id,
                "artifact_type": scene.resolved_artifact_type,
                "artifact_id": scene.resolved_artifact_id,
            }
            if any(record.get(key) != value for key, value in expected.items()):
                raise GenerationError("Assembly does not match the current scene selections.")
    except (GenerationError, OSError, EOFError, ValueError, wave.Error) as error:
        return AssemblyInspection(status, False, None, assembly.get("audio"), str(error), None)
    return AssemblyInspection(status, True, path, recorded_audio, None, None)


def inspect_run(run_directory):
    """Validate and interpret one existing run without changing persisted state."""
    run_directory, _, raw_manifest = read_manifest(run_directory)
    schema_version = raw_manifest.get("schema_version")
    if schema_version == 1:
        if raw_manifest.get("status") != "planned":
            raise GenerationError("Schema-version 1 run must have planned status.")
        validate_plan_identity(run_directory, raw_manifest)
        manifest = raw_manifest
        expected_rate = None
        generation_status = "not_started"
        generation_summary = None
        latest_operation = None
    elif schema_version in {2, 3, GENERATION_SCHEMA_VERSION}:
        run_directory, _, manifest = load_generation_run(run_directory)
        expected_rate = expected_rate_from_manifest(manifest, required=False)
        generation = manifest["generation"]
        generation_status = generation.get("status", "unknown")
        generation_summary = generation.get("summary")
        latest_operation = generation.get("last_operation")
    else:
        raise GenerationError(f"Unsupported audiobook schema version: {schema_version!r}")

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise GenerationError("Manifest has invalid source metadata.")
    source_path = path_inside_run(run_directory, source.get("snapshot_path"), "Source snapshot")
    scenes = tuple(
        _scene_inspection(run_directory, manifest, scene, expected_rate)
        for scene in manifest["scenes"]
    )
    assembly = _assembly_inspection(run_directory, manifest, scenes)
    return RunInspection(
        run_directory=run_directory, schema_version=schema_version,
        chapter_id=str(manifest.get("chapter_id", "")),
        run_id=str(manifest.get("run_id", "")),
        created_at_utc=str(manifest.get("created_at_utc", "")),
        source=dict(source), source_path=source_path,
        plan_hash=str(manifest.get("plan_hash", "")),
        run_status=str(manifest.get("status", "unknown")),
        generation_status=generation_status,
        generation_summary=generation_summary if isinstance(generation_summary, dict) else None,
        latest_operation=latest_operation if isinstance(latest_operation, dict) else None,
        assembly=assembly, scenes=scenes,
    )


def open_audio_file(path, opener=None):
    """Open a previously validated audio path in the platform's system player."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if opener is not None:
        opener(path)
    elif os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
