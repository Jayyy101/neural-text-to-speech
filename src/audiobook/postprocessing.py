"""Manual pause repair and exact PCM chapter assembly."""

import hashlib
import json
from pathlib import Path
import wave

from evaluation.apply_pause_plan import apply_pause_plan

from .cosyvoice import file_sha256, wav_info
from .pipeline import (
    GENERATION_SCHEMA_VERSION,
    GenerationError,
    expected_rate_from_manifest,
    load_generation_run,
    mark_assembly_stale,
    save_manifest,
    selected_attempt,
    utc_now,
    validate_attempt_artifact,
)


DEFAULT_PERIOD_ADD_MS = 140
DEFAULT_SEARCH_MS = 250.0


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def path_inside_run(run_directory, relative_path, label):
    if not isinstance(relative_path, str):
        raise GenerationError(f"{label} path is invalid.")
    path = (run_directory / relative_path).resolve()
    try:
        path.relative_to(run_directory)
    except ValueError as error:
        raise GenerationError(f"{label} path leaves the run directory.") from error
    return path


def load_bound_plan(plan_path, scene_id, attempt):
    plan_path = Path(plan_path).expanduser().resolve()
    try:
        supplied_bytes = plan_path.read_bytes()
        plan = json.loads(supplied_bytes.decode("utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GenerationError(f"Cannot read UTF-8 JSON pause plan: {error}") from error
    if not isinstance(plan, dict) or not isinstance(plan.get("source"), dict):
        raise GenerationError("Pause plan must contain a source identity object.")
    expected = {
        "scene_id": scene_id,
        "attempt_id": attempt["id"],
        "wav_sha256": attempt["wav_sha256"],
    }
    if any(plan["source"].get(key) != value for key, value in expected.items()):
        raise GenerationError(
            "Pause plan source identity is stale or does not match the selected attempt."
        )

    pauses = plan.get("pauses")
    if not isinstance(pauses, list) or not pauses:
        raise GenerationError("Pause plan must contain a non-empty 'pauses' list.")
    applied = json.loads(json.dumps(plan))
    for entry in applied["pauses"]:
        if (isinstance(entry, dict) and "add_ms" not in entry
                and entry.get("label") == "period"):
            entry["add_ms"] = DEFAULT_PERIOD_ADD_MS
    applied_bytes = (
        json.dumps(applied, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    return supplied_bytes, applied_bytes


def next_repair_id(scene):
    repair_state = scene.get("repair")
    repairs = repair_state["repairs"] if isinstance(repair_state, dict) else []
    number = 1 if not repairs else int(repairs[-1]["id"].split("_", 1)[1]) + 1
    return f"repair_{number:03d}"


def selected_repair(scene):
    repair_state = scene.get("repair")
    if not isinstance(repair_state, dict):
        return None
    selected_id = repair_state.get("selected_repair_id")
    if selected_id is None:
        return None
    for repair in repair_state["repairs"]:
        if repair["id"] == selected_id:
            return repair
    raise GenerationError(f"{scene['id']} selected repair does not exist.")


def validate_repair_artifact(run_directory, scene, repair, attempt, expected_rate):
    if repair.get("status") != "repaired":
        return False, "Repair was not recorded as successful."
    source = repair.get("source")
    if source != {
        "attempt_id": attempt["id"],
        "wav_sha256": attempt["wav_sha256"],
    }:
        return False, "Repair is bound to a different generated attempt."
    expected_path = (
        Path("scenes") / scene["id"] / "repairs" / repair["id"] / "repaired.wav"
    ).as_posix()
    if repair.get("output_path") != expected_path:
        return False, "Repair output path is not deterministic."
    output_path = path_inside_run(run_directory, expected_path, "Repair output")
    try:
        audio = wav_info(output_path, expected_rate)
        digest = file_sha256(output_path)
    except (OSError, EOFError, ValueError, wave.Error) as error:
        return False, str(error)
    if digest != repair.get("wav_sha256") or audio != repair.get("audio"):
        return False, "Repaired WAV does not match its recorded identity."
    return True, None


def repair_scene(run_directory, scene_id, plan_path, clock=utc_now):
    """Apply one source-bound manual plan without modifying generated audio."""
    run_directory, manifest_path, manifest = load_generation_run(run_directory)
    scene = next((item for item in manifest["scenes"] if item["id"] == scene_id), None)
    if scene is None:
        raise GenerationError(f"Scene does not exist: {scene_id}")
    attempt = selected_attempt(scene)
    if attempt is None:
        raise GenerationError(f"{scene_id} has no selected generated attempt.")
    expected_rate = expected_rate_from_manifest(manifest, required=True)
    valid, reason = validate_attempt_artifact(run_directory, attempt, expected_rate)
    if not valid:
        raise GenerationError(f"Selected generated attempt is invalid: {reason}")

    supplied_plan, applied_plan = load_bound_plan(plan_path, scene_id, attempt)
    repair_id = next_repair_id(scene)
    relative_dir = Path("scenes") / scene_id / "repairs" / repair_id
    repair_dir = run_directory / relative_dir
    if repair_dir.exists():
        raise GenerationError(f"Repair output already exists: {repair_dir}")
    source_path = path_inside_run(run_directory, attempt["output_path"], "Attempt output")
    original_sha = file_sha256(source_path)
    repair_dir.mkdir(parents=True)
    supplied_path = repair_dir / "source-plan.json"
    applied_path = repair_dir / "applied-plan.json"
    partial_path = repair_dir / "repaired.partial.wav"
    output_path = repair_dir / "repaired.wav"
    try:
        supplied_path.write_bytes(supplied_plan)
        applied_path.write_bytes(applied_plan)
        result = apply_pause_plan(source_path, applied_path, partial_path, DEFAULT_SEARCH_MS)
        audio = wav_info(partial_path, expected_rate)
        output_sha = file_sha256(partial_path)
        if file_sha256(source_path) != original_sha:
            raise GenerationError("Generated source WAV changed while applying the repair.")
        partial_path.replace(output_path)
    except Exception:
        for path in (partial_path, output_path, applied_path, supplied_path):
            path.unlink(missing_ok=True)
        repair_dir.rmdir()
        raise
    result = dict(result)
    result["output_path"] = output_path.relative_to(run_directory).as_posix()
    repair = {
        "id": repair_id,
        "status": "repaired",
        "created_at_utc": clock(),
        "source": {
            "attempt_id": attempt["id"],
            "wav_sha256": attempt["wav_sha256"],
        },
        "source_plan_path": supplied_path.relative_to(run_directory).as_posix(),
        "source_plan_sha256": sha256_bytes(supplied_plan),
        "applied_plan_path": applied_path.relative_to(run_directory).as_posix(),
        "applied_plan_sha256": sha256_bytes(applied_plan),
        "output_path": output_path.relative_to(run_directory).as_posix(),
        "wav_sha256": output_sha,
        "audio": audio,
        "result": result,
    }
    repair_state = scene.setdefault("repair", {"selected_repair_id": None, "repairs": []})
    repair_state["repairs"].append(repair)
    repair_state["selected_repair_id"] = repair_id
    manifest["schema_version"] = GENERATION_SCHEMA_VERSION
    mark_assembly_stale(manifest, f"{scene_id} selected repair changed to {repair_id}.")
    save_manifest(manifest_path, manifest)
    return manifest


def resolve_scene_artifact(run_directory, manifest, scene):
    attempt = selected_attempt(scene)
    if attempt is None:
        raise GenerationError(f"{scene['id']} has no selected generated attempt.")
    expected_rate = expected_rate_from_manifest(manifest, required=True)
    valid, reason = validate_attempt_artifact(run_directory, attempt, expected_rate)
    if not valid:
        raise GenerationError(f"{scene['id']} selected attempt is invalid: {reason}")
    repair = selected_repair(scene)
    if repair is not None:
        valid, reason = validate_repair_artifact(
            run_directory, scene, repair, attempt, expected_rate
        )
        if not valid:
            raise GenerationError(f"{scene['id']} selected repair is invalid: {reason}")
        return repair["output_path"], "repair", repair["id"], repair
    return attempt["output_path"], "generation", attempt["id"], attempt


def read_wav_payload(path):
    try:
        with wave.open(str(path), "rb") as audio:
            frames = audio.getnframes()
            params = {
                "sample_rate_hz": audio.getframerate(),
                "channels": audio.getnchannels(),
                "sample_width_bytes": audio.getsampwidth(),
                "compression_type": audio.getcomptype(),
            }
            payload = audio.readframes(frames)
    except (OSError, EOFError, wave.Error) as error:
        raise GenerationError(f"Cannot read scene WAV: {error}") from error
    expected_bytes = frames * params["channels"] * params["sample_width_bytes"]
    if frames <= 0 or len(payload) != expected_bytes:
        raise GenerationError("Scene WAV is empty or truncated.")
    return params, frames, payload


def assemble_chapter(run_directory, clock=utc_now):
    """Concatenate selected PCM payloads in planned order with no added frames."""
    run_directory, manifest_path, manifest = load_generation_run(run_directory)
    clips = []
    required_params = None
    cursor = 0
    for scene in manifest["scenes"]:
        relative_path, kind, artifact_id, artifact = resolve_scene_artifact(
            run_directory, manifest, scene
        )
        path = path_inside_run(run_directory, relative_path, "Scene artifact")
        params, frames, payload = read_wav_payload(path)
        if required_params is None:
            required_params = params
        elif params != required_params:
            raise GenerationError("Selected scene WAV formats are incompatible.")
        clips.append({
            "scene_id": scene["id"],
            "artifact_type": kind,
            "artifact_id": artifact_id,
            "artifact_path": relative_path,
            "artifact_wav_sha256": artifact["wav_sha256"],
            "start_frame": cursor,
            "end_frame_exclusive": cursor + frames,
            "frame_count": frames,
            "payload": payload,
        })
        cursor += frames

    final_dir = run_directory / "final"
    final_dir.mkdir(exist_ok=True)
    output_path = final_dir / "chapter.wav"
    partial_path = final_dir / "chapter.partial.wav"
    if partial_path.exists():
        raise GenerationError(f"Incomplete assembly output already exists: {partial_path}")
    try:
        with wave.open(str(partial_path), "wb") as output:
            output.setparams((
                required_params["channels"], required_params["sample_width_bytes"],
                required_params["sample_rate_hz"], 0, "NONE", "not compressed",
            ))
            for clip in clips:
                output.writeframesraw(clip["payload"])
        audio = wav_info(partial_path, required_params["sample_rate_hz"])
        if audio["frames"] != cursor:
            raise GenerationError("Assembled WAV frame count is incorrect.")
        output_sha = file_sha256(partial_path)
        partial_path.replace(output_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise

    for clip in clips:
        clip.pop("payload")
    manifest["schema_version"] = GENERATION_SCHEMA_VERSION
    manifest["assembly"] = {
        "status": "assembled",
        "created_at_utc": clock(),
        "output_path": output_path.relative_to(run_directory).as_posix(),
        "wav_sha256": output_sha,
        "audio": audio,
        "extra_silence_ms_between_scenes": 0,
        "scenes": clips,
    }
    save_manifest(manifest_path, manifest)
    return manifest
