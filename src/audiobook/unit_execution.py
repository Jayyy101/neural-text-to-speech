"""Schema-5 synthesis-unit execution, recovery, and exact PCM assembly."""

import hashlib
import json
from pathlib import Path
import secrets
import wave

from .cosyvoice import file_sha256, wav_info
from .content_qc import (
    ASRWorkerClient, DEFAULT_ASR_PYTHON, MAX_WHOLE_WAV_SECONDS,
    MODEL_ID, MODEL_REVISION, audio_request,
    compare_recognition, policy_record,
)
from .pipeline import (
    ATTEMPT_PATTERN, GenerationError, error_record, read_manifest,
    save_manifest, utc_now, validate_attempt_artifact, validate_seed,
)
from .postprocessing import read_wav_payload
from .unit_planning import (
    UNIT_PLAN_SCHEMA_VERSION, validate_synthesis_unit_plan,
)
from . import profiling


UNIT_SEED_POLICY = "sha256_root_plan_unit_take_v1"
QC_EXECUTION_POLICY = "content_qc_required_v1"
RETRY_EXECUTION_POLICY = "content_qc_bounded_retries_v1"
CONTENT_RETRY_POLICY = {
    "policy": "bounded_content_qc_retries_v1",
    "max_total_attempts_per_unit": 3,
    "trigger": "validated_contiguous_expected_han_deletion_v1",
}
def derive_unit_seed(root_seed, plan_hash, unit_id, take_index=1):
    """Derive an explicit 32-bit seed independent of execution order."""
    validate_seed(root_seed)
    if not isinstance(take_index, int) or isinstance(take_index, bool) or take_index < 1:
        raise GenerationError("Unit take index must be a positive integer.")
    payload = json.dumps(
        [UNIT_SEED_POLICY, root_seed, plan_hash, unit_id, take_index],
        separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _units(manifest):
    for scene in manifest["scenes"]:
        for unit in scene["synthesis_units"]:
            yield scene, unit


def _attempt_path(scene_id, unit_id, attempt_id):
    return (Path("units") / scene_id / unit_id / attempt_id / "generated.wav").as_posix()


def _load(run_directory):
    run_directory, path, manifest = read_manifest(run_directory)
    if manifest.get("schema_version") != UNIT_PLAN_SCHEMA_VERSION:
        raise GenerationError("Operation requires a schema-version 5 unit run.")
    validate_synthesis_unit_plan(run_directory, manifest)
    generation = manifest.get("generation")
    execution_policy = manifest.get("unit_execution_policy")
    if execution_policy not in (None, QC_EXECUTION_POLICY, RETRY_EXECUTION_POLICY):
        raise GenerationError("Unit execution policy is invalid.")
    if generation is not None:
        if not isinstance(generation, dict) or generation.get("seed_policy") != UNIT_SEED_POLICY:
            raise GenerationError("Unit generation seed policy is invalid.")
        root_seed = validate_seed(generation.get("root_seed"))
        qc_config = generation.get("content_qc")
        if (execution_policy in {QC_EXECUTION_POLICY, RETRY_EXECUTION_POLICY}) != (
                qc_config is not None):
            raise GenerationError("Unit content-QC requirement and configuration differ.")
        if qc_config is not None and qc_config != policy_record(
                qc_config.get("asr_python") if isinstance(qc_config, dict) else None):
            raise GenerationError("Recorded content-QC policy or model revision differs.")
        retry_policy = generation.get("content_retry")
        if ((execution_policy == RETRY_EXECUTION_POLICY) != (retry_policy is not None)
                or retry_policy is not None and retry_policy != CONTENT_RETRY_POLICY):
            raise GenerationError("Recorded content-retry policy differs.")
        for scene, unit in _units(manifest):
            state = unit.get("generation")
            if not isinstance(state, dict) or not isinstance(state.get("attempts"), list):
                raise GenerationError(f"{unit['id']} has invalid generation state.")
            seen = set()
            previous = 0
            previous_attempt = None
            if retry_policy is not None and len(state["attempts"]) > retry_policy[
                    "max_total_attempts_per_unit"]:
                raise GenerationError(f"{unit['id']} exceeds its content-retry attempt limit.")
            for attempt in state["attempts"]:
                attempt_id = attempt.get("id") if isinstance(attempt, dict) else None
                match = ATTEMPT_PATTERN.fullmatch(attempt_id) if isinstance(attempt_id, str) else None
                if match is None or int(match.group(1)) <= previous:
                    raise GenerationError(f"{unit['id']} has invalid attempt history.")
                if retry_policy is not None and int(match.group(1)) != previous + 1:
                    raise GenerationError(f"{unit['id']} has a gap in bounded attempt history.")
                previous = int(match.group(1))
                seen.add(attempt_id)
                if attempt.get("output_path") != _attempt_path(scene["id"], unit["id"], attempt_id):
                    raise GenerationError(f"{unit['id']} has an invalid attempt path.")
                take_index = attempt.get("take_index")
                if retry_policy is not None:
                    if (previous_attempt is not None
                            and previous_attempt["status"] == "generated"
                            and previous_attempt["content_qc"]["status"] != "rejected"):
                        raise GenerationError(f"{unit['id']} has unauthorized retry history.")
                    expected_take = (
                        1 if previous_attempt is None else
                        previous_attempt["take_index"] + (
                            1 if previous_attempt["status"] == "generated"
                            and previous_attempt["content_qc"]["status"] == "rejected"
                            else 0
                        )
                    )
                    if take_index != expected_take:
                        raise GenerationError(f"{unit['id']} has invalid logical take history.")
                if attempt.get("seed") != derive_unit_seed(
                    root_seed, manifest["synthesis_unit_plan"]["ordered_unit_plan_sha256"],
                    unit["id"], take_index,
                ):
                    raise GenerationError(f"{unit['id']} has an invalid attempt seed.")
                if attempt.get("status") not in {"running", "generated", "failed"}:
                    raise GenerationError(f"{unit['id']} has an invalid attempt status.")
                if (attempt.get("normalized_text_sha256")
                        != unit["normalized_text_sha256"]
                        or attempt.get("frontend_bypass") is not True):
                    raise GenerationError(f"{unit['id']} attempt does not match frozen text.")
                title = manifest.get("title_synthesis_override")
                expected_synthesis_hash = (
                    title["synthesis_text_sha256"]
                    if title is not None and unit["id"] == title["unit_id"] else None
                )
                if attempt.get("synthesis_text_override_sha256") != expected_synthesis_hash:
                    raise GenerationError(f"{unit['id']} synthesis text override differs.")
                if qc_config is not None:
                    qc = attempt.get("content_qc")
                    if not isinstance(qc, dict) or qc.get("status") not in {
                        "pending", "running", "passed", "rejected", "error",
                    }:
                        raise GenerationError(f"{unit['id']} has invalid attempt QC state.")
                previous_attempt = attempt
            selected = state.get("selected_attempt_id")
            if selected is not None and selected not in seen:
                raise GenerationError(f"{unit['id']} has an unknown selected attempt.")
            if not isinstance(state.get("selection_history"), list):
                raise GenerationError(f"{unit['id']} has invalid selection history.")
            if retry_policy is not None and not isinstance(state.get("retry_state"), dict):
                raise GenerationError(f"{unit['id']} has invalid content-retry state.")
    return run_directory, path, manifest


def _refresh(manifest):
    selected = 0
    total = 0
    for scene in manifest["scenes"]:
        units = scene["synthesis_units"]
        if manifest["generation"].get("content_retry") is not None:
            for unit in units:
                _refresh_retry_state(unit)
        count = sum(unit["generation"]["selected_attempt_id"] is not None for unit in units)
        scene["generation"] = {
            "status": "generated" if count == len(units) else "incomplete",
            "selected_units": count,
            "total_units": len(units),
        }
        selected += count
        total += len(units)
    manifest["status"] = "generated" if selected == total else "generation_failed"
    manifest["generation"]["status"] = manifest["status"]
    manifest["generation"]["summary"] = {
        "generated_units": selected, "failed_units": total - selected,
        "total_units": total,
        "generated_scenes": sum(scene["generation"]["status"] == "generated"
                                for scene in manifest["scenes"]),
        "total_scenes": len(manifest["scenes"]),
    }


def _refresh_retry_state(unit):
    state = unit["generation"]
    attempts = state["attempts"]
    maximum = CONTENT_RETRY_POLICY["max_total_attempts_per_unit"]
    if state["selected_attempt_id"] is not None:
        status, reason = "selected", "content_qc_passed"
    elif not attempts:
        status, reason = "not_started", "no_attempts"
    else:
        latest = attempts[-1]
        qc_status = latest["content_qc"]["status"]
        if latest["status"] == "generated" and qc_status == "rejected":
            if len(attempts) == maximum:
                if all(item["status"] == "generated" and
                       item["content_qc"]["status"] == "rejected"
                       for item in attempts):
                    status, reason = "exhausted", "all_allowed_attempts_rejected_for_content"
                else:
                    status, reason = "attempt_limit_reached", "synthesis_attempt_limit"
            else:
                status, reason = "retry_pending", "validated_content_rejection"
        elif latest["status"] == "generated" and qc_status == "error":
            status, reason = "qc_error", "retry_qc_on_existing_wav"
        elif latest["status"] == "generated":
            status, reason = "qc_pending", "retry_qc_on_existing_wav"
        elif len(attempts) == maximum:
            status, reason = "attempt_limit_reached", "synthesis_attempt_limit"
        else:
            status, reason = "synthesis_pending", "retry_same_logical_take"
    state["retry_state"] = {
        "status": status, "reason": reason,
        "attempts_used": len(attempts), "max_total_attempts": maximum,
        "last_attempt_id": attempts[-1]["id"] if attempts else None,
    }


def _select(manifest, unit, attempt, clock, reason):
    with profiling.span("unit.selection", unit_id=unit["id"],
                        attempt_id=attempt["id"], reason=reason):
        return _select_impl(manifest, unit, attempt, clock, reason)


def _select_impl(manifest, unit, attempt, clock, reason):
    state = unit["generation"]
    previous = state["selected_attempt_id"]
    state["selected_attempt_id"] = attempt["id"]
    if previous != attempt["id"]:
        state["selection_history"].append({
            "at_utc": clock(), "previous_attempt_id": previous,
            "selected_attempt_id": attempt["id"], "reason": reason,
        })
        assembly = manifest.get("assembly")
        if isinstance(assembly, dict) and assembly.get("status") == "assembled":
            assembly["status"] = "stale"
            assembly["stale_reason"] = f"{unit['id']} selected attempt changed."


def _verify_unit_artifact(run_directory, attempt, rate):
    valid, reason = validate_attempt_artifact(run_directory, attempt, rate)
    if not valid:
        raise GenerationError(f"Selected unit WAV is invalid: {reason}")


def _qc_path(attempt):
    return (Path(attempt["output_path"]).parent / "content_qc.json").as_posix()


def _read_qc_evidence(run_directory, manifest, unit, attempt):
    qc = attempt["content_qc"]
    expected_path = _qc_path(attempt)
    if qc.get("evidence_path") not in (None, expected_path):
        raise GenerationError("Unit QC evidence path is invalid.")
    path = run_directory / expected_path
    if not path.exists():
        if qc.get("status") in {"passed", "rejected"}:
            raise GenerationError("Passed or rejected unit QC evidence is missing.")
        return None
    digest = file_sha256(path)
    if qc.get("evidence_sha256") is not None and digest != qc["evidence_sha256"]:
        raise GenerationError("Unit QC evidence SHA-256 differs.")
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GenerationError(f"Unit QC evidence cannot be read: {error}") from error
    if not isinstance(evidence, dict):
        raise GenerationError("Unit QC evidence must be a JSON object.")
    binding = evidence.get("binding")
    expected_binding = {
        "unit_id": unit["id"], "attempt_id": attempt["id"],
        "wav_path": attempt["output_path"],
        "wav_sha256": attempt["wav_sha256"],
        "normalized_text_sha256": unit["normalized_text_sha256"],
    }
    if binding != expected_binding or evidence.get("policy") != manifest[
            "generation"]["content_qc"]:
        raise GenerationError("Unit QC evidence binding or policy differs.")
    model = evidence.get("model")
    if (not isinstance(model, dict) or model.get("model_id") != MODEL_ID
            or model.get("resolved_revision") != MODEL_REVISION):
        raise GenerationError("Unit QC evidence model revision differs.")
    recognition = evidence.get("recognition")
    if not isinstance(recognition, dict) or recognition.get("wav_sha256") != attempt["wav_sha256"]:
        raise GenerationError("Unit QC recognition audio identity differs.")
    tokens = recognition.get("comparison_tokens")
    if not isinstance(tokens, list) or not all(
        isinstance(item, dict) and isinstance(item.get("comparison_token"), str)
        for item in tokens
    ):
        raise GenerationError("Unit QC recognition comparison tokens are invalid.")
    if recognition.get("comparison_text") != "".join(
        item.get("comparison_token", "")
        for item in tokens
    ):
        raise GenerationError("Unit QC recognition comparison text differs.")
    try:
        comparison = compare_recognition(unit["normalized_text"], recognition)
    except (KeyError, TypeError, ValueError) as error:
        raise GenerationError(f"Unit QC comparison evidence is invalid: {error}") from error
    if comparison != evidence.get("comparison") or evidence.get("decision") != comparison["decision"]:
        raise GenerationError("Unit QC comparison or decision differs from evidence.")
    if qc.get("status") in {"passed", "rejected"} and (
        qc.get("evidence_sha256") != digest
        or qc.get("status") != comparison["decision"]
    ):
        raise GenerationError("Unit QC manifest state differs from its evidence.")
    return evidence, digest


def _run_qc(run_directory, manifest_path, manifest, unit, attempt,
            worker, rate, clock):
    with profiling.span("unit.qc", unit_id=unit["id"],
                        attempt_id=attempt["id"]) as timing:
        decision = _run_qc_impl(run_directory, manifest_path, manifest, unit,
                                attempt, worker, rate, clock)
        timing.add(decision=decision)
        return decision


def _run_qc_impl(run_directory, manifest_path, manifest, unit, attempt,
                 worker, rate, clock):
    _verify_unit_artifact(run_directory, attempt, rate)
    recovered = _read_qc_evidence(run_directory, manifest, unit, attempt)
    if recovered is not None:
        evidence, digest = recovered
        decision = evidence["decision"]
        attempt["content_qc"].update({
            "status": decision, "decision": decision,
            "evidence_path": _qc_path(attempt), "evidence_sha256": digest,
            "finished_at_utc": clock(), "recovery": "validated_existing_evidence",
        })
        if decision == "passed":
            _select(manifest, unit, attempt, clock, "content_qc_passed")
        save_manifest(manifest_path, manifest)
        return decision
    qc = attempt["content_qc"]
    qc.update({"status": "running", "started_at_utc": clock()})
    save_manifest(manifest_path, manifest)
    try:
        if attempt["audio"]["duration_seconds"] > MAX_WHOLE_WAV_SECONDS:
            raise GenerationError(
                "Unit WAV exceeds the validated 30-second whole-WAV ASR limit."
            )
        # Recognition receives only path and WAV hash. Intended text is used below,
        # after the independent worker returns.
        request = audio_request(
            run_directory / attempt["output_path"], attempt["wav_sha256"],
            f"{unit['id']}:{attempt['id']}",
        )
        recognition = worker.recognize(request)
        if (recognition.get("type") != "recognized"
                or recognition.get("request_id") != request["request_id"]
                or recognition.get("wav_sha256") != request["wav_sha256"]):
            raise GenerationError("ASR recognition response audio identity differs.")
        if (worker.model.get("model_id") != MODEL_ID
                or worker.model.get("resolved_revision") != MODEL_REVISION):
            raise GenerationError("ASR worker model revision differs.")
        with profiling.span("asr.content_comparison"):
            comparison = compare_recognition(unit["normalized_text"], recognition)
        decision = comparison["decision"]
        evidence = {
            "schema_version": 1,
            "binding": {
                "unit_id": unit["id"], "attempt_id": attempt["id"],
                "wav_path": attempt["output_path"],
                "wav_sha256": attempt["wav_sha256"],
                "normalized_text_sha256": unit["normalized_text_sha256"],
            },
            "policy": manifest["generation"]["content_qc"],
            "model": worker.model,
            "recognition": recognition,
            "comparison": comparison,
            "decision": decision,
            "created_at_utc": clock(),
        }
        evidence_path = run_directory / _qc_path(attempt)
        temporary = evidence_path.with_suffix(".json.tmp")
        if evidence_path.exists() or temporary.exists():
            raise GenerationError("Untracked unit QC evidence already exists.")
        with profiling.span("unit.qc_evidence_write"):
            temporary.write_text(
                json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            temporary.replace(evidence_path)
        qc.update({
            "status": decision, "decision": decision,
            "evidence_path": _qc_path(attempt),
            "evidence_sha256": file_sha256(evidence_path),
            "finished_at_utc": clock(),
        })
        if decision == "passed":
            _select(manifest, unit, attempt, clock, "content_qc_passed")
        save_manifest(manifest_path, manifest)
        return decision
    except Exception as error:
        qc.update({"status": "error", "finished_at_utc": clock(),
                   "error": error_record(error)})
        save_manifest(manifest_path, manifest)
        return "error"


def _recover_or_target(run_directory, manifest, unit, rate, clock):
    with profiling.span("unit.recovery_check", unit_id=unit["id"]):
        return _recover_or_target_impl(run_directory, manifest, unit, rate, clock)


def _recover_or_target_impl(run_directory, manifest, unit, rate, clock):
    state = unit["generation"]
    attempts = state["attempts"]
    qc_enabled = "content_qc" in manifest["generation"]
    retry_policy = manifest["generation"].get("content_retry")
    if retry_policy is not None:
        # A later take is authorized only by intact, independently verifiable
        # rejection evidence from every earlier rejected attempt.
        for prior in attempts:
            if prior["status"] == "generated" and prior["content_qc"]["status"] == "rejected":
                _verify_unit_artifact(run_directory, prior, rate)
                _read_qc_evidence(run_directory, manifest, unit, prior)
    selected_id = state["selected_attempt_id"]
    if selected_id is not None:
        selected = next(item for item in attempts if item["id"] == selected_id)
        if rate is None:
            raise GenerationError("Cannot validate selected unit without a sample rate.")
        _verify_unit_artifact(run_directory, selected, rate)
        if qc_enabled:
            if selected["content_qc"]["status"] != "passed":
                raise GenerationError("Selected unit has not passed content QC.")
            _read_qc_evidence(run_directory, manifest, unit, selected)
        return "none"
    if attempts:
        for generated in reversed(attempts):
            if generated["status"] == "generated":
                if rate is None:
                    raise GenerationError("Cannot recover unit selection without a sample rate.")
                _verify_unit_artifact(run_directory, generated, rate)
                if qc_enabled:
                    qc_status = generated["content_qc"]["status"]
                    if qc_status in {"passed", "rejected"}:
                        _read_qc_evidence(run_directory, manifest, unit, generated)
                        if qc_status == "passed":
                            _select(manifest, unit, generated, clock,
                                    "recovered_passed_qc_selection")
                            return "none"
                        if (retry_policy is not None and len(attempts) < retry_policy[
                                "max_total_attempts_per_unit"]):
                            return "synthesize"
                        return "none"
                    return "qc"
                _select(manifest, unit, generated, clock, "recovered_generated_attempt")
                return "none"
        latest = attempts[-1]
        path = run_directory / latest["output_path"]
        if latest["status"] == "running" and path.is_file() and rate is not None:
            try:
                audio = wav_info(path, rate)
            except (OSError, EOFError, ValueError, wave.Error):
                pass
            else:
                latest.update({
                    "status": "generated", "finished_at_utc": clock(),
                    "audio": audio, "wav_sha256": file_sha256(path),
                    "recovery": "validated_existing_wav",
                })
                if qc_enabled:
                    return "qc"
                _select(manifest, unit, latest, clock, "recovered_interrupted_attempt")
                return "none"
        if latest["status"] == "running":
            latest.update({
                "status": "failed", "finished_at_utc": clock(),
                "recovery": "interrupted_attempt_preserved",
            })
    if retry_policy is not None and len(attempts) >= retry_policy[
            "max_total_attempts_per_unit"]:
        return "none"
    return "synthesize"


def _run_attempt(run_directory, manifest_path, manifest, scene, unit,
                 backend, rate, clock):
    with profiling.span("unit.attempt", unit_id=unit["id"]) as timing:
        success = _run_attempt_impl(run_directory, manifest_path, manifest, scene,
                                    unit, backend, rate, clock)
        timing.add(success=success)
        if not success:
            timing.finish("failed")
        return success


def _run_attempt_impl(run_directory, manifest_path, manifest, scene, unit,
                      backend, rate, clock):
    state = unit["generation"]
    attempts = state["attempts"]
    next_number = (
        int(ATTEMPT_PATTERN.fullmatch(attempts[-1]["id"]).group(1)) + 1
        if attempts else 1
    )
    attempt_id = f"attempt_{next_number:03d}"
    profiling.annotate(attempt_id=attempt_id)
    output_rel = _attempt_path(scene["id"], unit["id"], attempt_id)
    output = run_directory / output_rel
    if output.parent.exists():
        raise GenerationError(f"Untracked unit attempt output already exists: {output.parent}")
    take_index = (attempts[-1]["take_index"] + 1
                  if attempts and manifest["generation"].get("content_retry") is not None
                  and attempts[-1]["status"] == "generated"
                  and attempts[-1]["content_qc"]["status"] == "rejected"
                  else attempts[-1]["take_index"] if attempts else 1)
    seed = derive_unit_seed(
        manifest["generation"]["root_seed"],
        manifest["synthesis_unit_plan"]["ordered_unit_plan_sha256"],
        unit["id"], take_index,
    )
    attempt = {
        "id": attempt_id, "take_index": take_index, "seed": seed,
        "random_state": {
            "policy": "explicit_global_seed", "seed": seed,
            "scope": ["python", "numpy", "torch_cpu", "torch_cuda"],
            "applied_after_model_initialization": True,
        },
        "normalized_text_sha256": unit["normalized_text_sha256"],
        "frontend_bypass": True, "status": "running",
        "output_path": output_rel, "started_at_utc": clock(),
    }
    title = manifest.get("title_synthesis_override")
    synthesis_text = unit["normalized_text"]
    if title is not None and unit["id"] == title["unit_id"]:
        synthesis_text = title["synthesis_text"]
        attempt["synthesis_text_override_sha256"] = title["synthesis_text_sha256"]
    qc_enabled = "content_qc" in manifest["generation"]
    if qc_enabled:
        attempt["content_qc"] = {"status": "pending"}
    attempts.append(attempt)
    save_manifest(manifest_path, manifest)
    try:
        output.parent.mkdir(parents=True)
        with profiling.span("unit.synthesis"):
            result = backend.generate_unit(synthesis_text, output, seed)
        audio = wav_info(output, rate)
        profiling.annotate(audio_seconds=audio["duration_seconds"],
                           audio_frames=audio["frames"])
        attempt.update({
            "status": "generated", "finished_at_utc": clock(),
            "audio": audio, "wav_sha256": file_sha256(output),
            "result": result,
        })
        if not qc_enabled:
            _select(manifest, unit, attempt, clock, "initial_success")
        success = True
    except Exception as error:
        attempt.update({
            "status": "failed", "finished_at_utc": clock(),
            "error": error_record(error),
        })
        success = False
    save_manifest(manifest_path, manifest)
    return success


def generate_units(run_directory, backend, root_seed=None, clock=utc_now,
                   asr_python=DEFAULT_ASR_PYTHON,
                   worker_factory=None):
    """Generate missing schema-5 units; resume uses the same persisted seeds."""
    run_directory, manifest_path, manifest = _load(run_directory)
    generation = manifest.get("generation")
    if generation is None:
        seed = secrets.randbits(32) if root_seed is None else validate_seed(root_seed)
        manifest["generation"] = {
            "status": "initializing", "started_at_utc": clock(),
            "configuration": backend.configuration(), "root_seed": seed,
            "seed_policy": UNIT_SEED_POLICY,
            "rng_semantics": "explicit_per_unit_seed_not_historical_global_stream",
            "content_qc": policy_record(asr_python),
            "content_retry": dict(CONTENT_RETRY_POLICY),
        }
        manifest["unit_execution_policy"] = RETRY_EXECUTION_POLICY
        for _, unit in _units(manifest):
            unit["generation"] = {
                "selected_attempt_id": None, "attempts": [],
                "selection_history": [],
                "retry_state": {
                    "status": "not_started", "reason": "no_attempts",
                    "attempts_used": 0,
                    "max_total_attempts": CONTENT_RETRY_POLICY["max_total_attempts_per_unit"],
                    "last_attempt_id": None,
                },
            }
        manifest["status"] = "generating"
        save_manifest(manifest_path, manifest)
        generation = manifest["generation"]
    else:
        if root_seed is not None and root_seed != generation["root_seed"]:
            raise GenerationError("Requested root seed differs from persisted root seed.")
        recorded = generation.get("backend") or generation.get("configuration")
        if not isinstance(recorded, dict) or any(
            recorded.get(key) != value for key, value in backend.configuration().items()
        ):
            raise GenerationError("Requested backend configuration differs from this unit run.")
        if "content_qc" in generation and generation["content_qc"] != policy_record(asr_python):
            raise GenerationError("Requested content-QC configuration differs from this run.")
    rate = generation.get("backend", {}).get("sample_rate_hz")
    actions = []
    for scene, unit in _units(manifest):
        action = _recover_or_target(run_directory, manifest, unit, rate, clock)
        if action != "none":
            actions.append((action, scene, unit))
    save_manifest(manifest_path, manifest)
    if not actions:
        _refresh(manifest)
        save_manifest(manifest_path, manifest)
        return manifest
    backend_ready = False

    def initialize_backend():
        nonlocal backend_ready, rate
        if backend_ready:
            return True
        try:
            metadata = backend.initialize_units()
            if metadata.get("frontend_identity_sha256") != manifest[
                "synthesis_unit_plan"
            ]["frontend_identity_sha256"]:
                raise GenerationError("Initialized frontend differs from frozen unit plan.")
            actual_rate = metadata.get("sample_rate_hz")
            if not isinstance(actual_rate, int) or actual_rate <= 0 or (
                rate is not None and actual_rate != rate
            ):
                raise GenerationError("Unit backend sample rate is invalid or changed.")
        except Exception as error:
            generation["initialization_failure"] = error_record(error)
            _refresh(manifest)
            save_manifest(manifest_path, manifest)
            return False
        generation["backend"] = metadata
        generation.pop("configuration", None)
        rate = actual_rate
        backend_ready = True
        return True

    generation["status"] = "running"
    manifest["status"] = "generating"
    save_manifest(manifest_path, manifest)
    worker = None
    try:
        for action, scene, unit in actions:
            with profiling.span("unit.cycle", unit_id=unit["id"], initial_action=action) as timing:
                while action != "none":
                    if action == "synthesize":
                        if not initialize_backend():
                            return manifest
                        success = _run_attempt(
                            run_directory, manifest_path, manifest, scene, unit,
                            backend, rate, clock,
                        )
                        if not success or "content_qc" not in generation:
                            break
                        attempt = unit["generation"]["attempts"][-1]
                    else:
                        attempt = next(
                            item for item in reversed(unit["generation"]["attempts"])
                            if item["status"] == "generated"
                        )
                    if (worker is None and attempt["audio"]["duration_seconds"]
                            <= MAX_WHOLE_WAV_SECONDS):
                        worker = (worker_factory or ASRWorkerClient)(
                            asr_python, run_directory / "content_qc_worker.log"
                        )
                    decision = _run_qc(run_directory, manifest_path, manifest, unit, attempt,
                                       worker, rate, clock)
                    if decision == "rejected" and generation.get("content_retry") is not None:
                        with profiling.span("unit.retry_decision", decision=decision,
                                            attempt_id=attempt["id"]):
                            action = _recover_or_target(run_directory, manifest, unit, rate, clock)
                    else:
                        break
                timing.add(attempts_used=len(unit["generation"]["attempts"]),
                           selected_attempt_id=unit["generation"]["selected_attempt_id"])
    finally:
        if worker is not None:
            worker.close()
    _refresh(manifest)
    generation["finished_at_utc"] = clock()
    save_manifest(manifest_path, manifest)
    return manifest


def assemble_units(run_directory, clock=utc_now):
    """Publish exact selected unit PCM in scene and unit order."""
    with profiling.span("assembly.total"):
        return _assemble_units(run_directory, clock=clock)


def _assemble_units(run_directory, clock=utc_now):
    run_directory, manifest_path, manifest = _load(run_directory)
    generation = manifest.get("generation")
    if not isinstance(generation, dict) or manifest["status"] != "generated":
        raise GenerationError("Every synthesis unit must be selected before assembly.")
    rate = generation.get("backend", {}).get("sample_rate_hz")
    if not isinstance(rate, int) or rate <= 0:
        raise GenerationError("Unit run has no valid backend sample rate.")
    clips = []
    params = None
    cursor = 0
    for scene, unit in _units(manifest):
        state = unit["generation"]
        selected_id = state["selected_attempt_id"]
        if selected_id is None:
            raise GenerationError(f"{unit['id']} has no selected attempt.")
        attempt = next(item for item in state["attempts"] if item["id"] == selected_id)
        with profiling.span("assembly.selected_validation", unit_id=unit["id"]):
            _verify_unit_artifact(run_directory, attempt, rate)
            if "content_qc" in generation:
                if attempt["content_qc"]["status"] != "passed":
                    raise GenerationError(f"{unit['id']} has no passed content QC.")
                _read_qc_evidence(run_directory, manifest, unit, attempt)
        path = (run_directory / attempt["output_path"]).resolve()
        path.relative_to(run_directory)
        with profiling.span("assembly.read_pcm", unit_id=unit["id"]) as timing:
            clip_params, frames, payload = read_wav_payload(path)
            timing.add(bytes_read=len(payload))
        if params is None:
            params = clip_params
        elif clip_params != params:
            raise GenerationError("Selected unit WAV formats are incompatible.")
        clips.append({
            "scene_id": scene["id"], "unit_id": unit["id"],
            "selected_attempt_id": selected_id,
            "artifact_path": attempt["output_path"],
            "artifact_wav_sha256": attempt["wav_sha256"],
            "start_frame": cursor, "end_frame_exclusive": cursor + frames,
            "frame_count": frames, "payload": payload,
        })
        cursor += frames
    if len(clips) != manifest["synthesis_unit_plan"]["total_units"] or len({
        clip["unit_id"] for clip in clips
    }) != len(clips):
        raise GenerationError("Assembly does not contain every planned unit exactly once.")
    final_dir = run_directory / "final"
    final_dir.mkdir(exist_ok=True)
    output = final_dir / "chapter.wav"
    partial = final_dir / "chapter.partial.wav"
    if partial.exists():
        raise GenerationError(f"Incomplete assembly output already exists: {partial}")
    try:
        with profiling.span("assembly.write_pcm", bytes_written=cursor * params["sample_width_bytes"]):
            with wave.open(str(partial), "wb") as wav:
                wav.setparams((params["channels"], params["sample_width_bytes"],
                               params["sample_rate_hz"], 0, "NONE", "not compressed"))
                for clip in clips:
                    wav.writeframesraw(clip["payload"])
        audio = wav_info(partial, rate)
        if audio["frames"] != cursor:
            raise GenerationError("Assembled unit WAV frame count is incorrect.")
        digest = file_sha256(partial)
        with profiling.span("assembly.publish"):
            partial.replace(output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    for clip in clips:
        clip.pop("payload")
    manifest["assembly"] = {
        "status": "assembled", "created_at_utc": clock(),
        "output_path": output.relative_to(run_directory).as_posix(),
        "wav_sha256": digest, "audio": audio,
        "ordered_unit_plan_sha256": manifest["synthesis_unit_plan"]["ordered_unit_plan_sha256"],
        "extra_silence_ms_between_units": 0,
        "units": clips,
    }
    save_manifest(manifest_path, manifest)
    return manifest


def select_unit_attempt(run_directory, unit_id, attempt_id, clock=utc_now):
    """Select a previously generated valid unit take and stale its assembly."""
    run_directory, manifest_path, manifest = _load(run_directory)
    generation = manifest.get("generation")
    rate = generation.get("backend", {}).get("sample_rate_hz") if generation else None
    if not isinstance(rate, int) or rate <= 0:
        raise GenerationError("Unit run has no valid backend sample rate.")
    selected_unit = next(
        (unit for _, unit in _units(manifest) if unit["id"] == unit_id), None
    )
    if selected_unit is None:
        raise GenerationError(f"Synthesis unit does not exist: {unit_id}")
    attempt = next(
        (item for item in selected_unit["generation"]["attempts"]
         if item["id"] == attempt_id), None
    )
    if attempt is None:
        raise GenerationError(f"Unit attempt does not exist: {attempt_id}")
    _verify_unit_artifact(run_directory, attempt, rate)
    if "content_qc" in generation:
        if attempt["content_qc"]["status"] != "passed":
            raise GenerationError("Only a passed-QC unit attempt may be selected.")
        _read_qc_evidence(run_directory, manifest, selected_unit, attempt)
    _select(manifest, selected_unit, attempt, clock, "explicit_selection")
    _refresh(manifest)
    save_manifest(manifest_path, manifest)
    return manifest
