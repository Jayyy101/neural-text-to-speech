"""Freeze CosyVoice frontend units and prove their source-text mapping."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from .cosyvoice import (
    PINNED_FRONTEND_SPLITTING_SETTINGS,
    TEXT_PREPROCESSING_POLICY,
    CosyVoiceFrontendAdapter,
    preprocess_synthesis_text,
)
from .pipeline import (
    read_manifest,
    save_manifest,
    validate_plan_identity,
)
from .planning import PlanningError


UNIT_PLAN_SCHEMA_VERSION = 5
UNIT_PLANNER_VERSION = 1
UNIT_PLANNING_POLICY = "pinned_cosyvoice_frontend_units_v1"
LEGACY_SOURCE_MAPPING_POLICY = "unique_verified_partition_trailing_formatting_left_v1"
SOURCE_MAPPING_POLICY = "unique_verified_partition_native_terminal_dunhao_v2"
TERMINAL_DUNHAO_EQUIVALENCE = "native_slice_terminal_u3001_to_u3002_v1"
MAX_NORMALIZED_UNIT_CHARACTERS = 200
MAPPING_SEARCH_LIMIT = 100_000
# These hashes identify the installed implementation whose digit-free path
# preserves U+4E00-U+9FFF characters. A different installation uses the
# bounded exhaustive search until its normalization behavior is reviewed.
HAN_SUFFIX_FRONTEND_IDENTITY_SHA256 = (
    "4db610ce6bfa2a03e808e31232aa150d51de09a1ef7391d9b88b92da0a35146b"
)
HAN_SUFFIX_WETEXT_PY_SHA256 = (
    "aad8edaddeaa15d51dd3cf98b25f6f85faaf9282832770f781681038f9521130"
)
MAX_UNCERTIFIED_PREFIX_CHARACTERS = 64
IMMUTABLE_UNIT_KEYS = (
    "id", "order", "source_span", "source_text", "source_text_sha256",
    "preprocessed_text", "preprocessed_text_sha256", "text_preprocessing",
    "normalized_text", "normalized_text_sha256", "mapping",
)
TRAILING_PUNCTUATION = frozenset("。！？?!；;.'”」』")
TITLE_PUNCTUATION_POLICY = "explicit_native_chapter_title_period_v1"
CHAPTER_HEADING = re.compile(
    r"(?:[^\s。！？\r\n]{1,40}\s*)?第[0-9零〇一二三四五六七八九十百千]+章[^\r\n]*"
)


def _heading_has_terminal_punctuation(heading):
    return heading.rstrip("”」』\"' ").endswith(tuple("。！？?!；;"))


class SynthesisUnitPlanningError(PlanningError):
    """Raised when an immutable synthesis-unit plan cannot be certified."""


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sha256(value):
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition, message):
    if not condition:
        raise SynthesisUnitPlanningError(message)


def _canonical_boundary(source_text, end):
    """Keep sentence-ending punctuation and following whitespace on the left."""
    return (
        end == len(source_text)
        or (not source_text[end].isspace()
            and source_text[end] not in TRAILING_PUNCTUATION)
    )


def _slice_certification(source_slice, expected_unit, observed, identity_sha,
                         mapping_policy):
    """Certify exact output, or the pinned native terminal-dunhao case only."""
    mapping = {
        "policy": mapping_policy,
        "frontend_identity_sha256": identity_sha,
        "independent_slice_reproduced_exactly": True,
    }
    if tuple(observed) == (expected_unit,):
        return mapping
    tail = source_slice.rstrip()
    if (mapping_policy == SOURCE_MAPPING_POLICY
            and identity_sha == HAN_SUFFIX_FRONTEND_IDENTITY_SHA256
            and len(tail) >= 2 and _is_han(tail[-2]) and tail[-1] == "、"
            and expected_unit.endswith("、")
            and tuple(observed) == (expected_unit[:-1] + "。",)):
        mapping.update({
            "independent_slice_reproduced_exactly": False,
            "normalization_equivalence": {
                "rule": TERMINAL_DUNHAO_EQUIVALENCE,
                "source_character_offset": len(tail) - 1,
                "independent_normalized_text": observed[0],
                "independent_normalized_text_sha256": text_sha256(observed[0]),
            },
        })
        return mapping
    return None


def _planning_policy(mapping_policy):
    _require(mapping_policy in {LEGACY_SOURCE_MAPPING_POLICY, SOURCE_MAPPING_POLICY},
             "Requested source-mapping policy is incompatible.")
    return {
        "name": UNIT_PLANNING_POLICY,
        "version": UNIT_PLANNER_VERSION,
        "text_preprocessing_policy": TEXT_PREPROCESSING_POLICY,
        "source_mapping_policy": mapping_policy,
        "max_normalized_unit_characters": MAX_NORMALIZED_UNIT_CHARACTERS,
        "oversized_unit_behavior": "reject_without_fallback_splitting",
        "splitting_settings": dict(PINNED_FRONTEND_SPLITTING_SETTINGS),
    }


def _is_han(character):
    return "\u4e00" <= character <= "\u9fff"


def _supports_han_suffix_mapping(source_text, frozen_units, frontend,
                                 frontend_identity):
    """Recognize only the pinned, character-conserving input class."""
    if (len(frozen_units) < 2
            or not isinstance(frontend, CosyVoiceFrontendAdapter)
            or canonical_sha256(frontend_identity)
            != HAN_SUFFIX_FRONTEND_IDENTITY_SHA256):
        return False
    try:
        from wetext import wetext
        if file_sha256(Path(wetext.__file__)) != HAN_SUFFIX_WETEXT_PY_SHA256:
            return False
    except (ImportError, OSError, TypeError):
        return False
    return _han_suffix_input_is_safe(source_text)


def _han_suffix_input_is_safe(source_text):
    """Exclude source characters that can invalidate the Han invariant."""
    digit_positions = [i for i, char in enumerate(source_text)
                       if "0" <= char <= "9"]
    if digit_positions and digit_positions[-1] >= MAX_UNCERTIFIED_PREFIX_CHARACTERS:
        return False
    if digit_positions:
        first_sentence_end = next(
            (i for i, char in enumerate(source_text)
             if char in "。！？?!；;."), len(source_text)
        )
        if digit_positions[-1] > first_sentence_end:
            return False
    return all(
        _is_han(char) or char.isspace()
        or unicodedata.category(char).startswith("P")
        or "0" <= char <= "9"
        for char in source_text
    )


def _han_suffix_boundaries(source_text, frozen_units, first_eligible_end):
    """Keep every canonical cut with the required conserved suffix Han count."""
    source_counts = [0] * (len(source_text) + 1)
    for index in range(len(source_text) - 1, -1, -1):
        source_counts[index] = source_counts[index + 1] + _is_han(source_text[index])
    by_count = {}
    for end in range(first_eligible_end, len(source_text)):
        if _canonical_boundary(source_text, end):
            by_count.setdefault(source_counts[end], []).append(end)

    boundaries = [()] * (len(frozen_units) + 1)
    boundaries[0] = (0,)
    boundaries[-1] = (len(source_text),)
    remaining = 0
    for index in range(len(frozen_units) - 1, 0, -1):
        remaining += sum(_is_han(char) for char in frozen_units[index])
        boundaries[index] = by_count.get(remaining, ())
    return boundaries


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _find_unique_partition(scene_id, source_text, frozen_units, frontend,
                           frontend_identity=None, diagnostics=None,
                           mapping_policy=SOURCE_MAPPING_POLICY):
    """Return one certified partition, rejecting missing or ambiguous mappings."""
    normalize_cache = {}
    normalization_calls = 0
    candidate_calls = 0
    fallback_calls = 0
    identity_sha = canonical_sha256(frontend_identity)
    guarded = _supports_han_suffix_mapping(
        source_text, frozen_units, frontend, frontend_identity
    ) if frontend_identity is not None else False

    def normalized(start, end, kind):
        nonlocal normalization_calls, candidate_calls, fallback_calls
        key = (start, end)
        if key not in normalize_cache:
            normalization_calls += 1
            if normalization_calls > MAPPING_SEARCH_LIMIT:
                raise SynthesisUnitPlanningError(
                    f"{scene_id} source mapping exceeded the deterministic search "
                    f"limit of {MAPPING_SEARCH_LIMIT} frontend calls."
                )
            clean_text, _ = preprocess_synthesis_text(source_text[start:end])
            result = frontend.normalize(clean_text)
            _require(
                isinstance(result, list)
                and all(isinstance(item, str) and item for item in result),
                f"{scene_id} frontend returned invalid normalized units.",
            )
            normalize_cache[key] = tuple(result)
            if kind == "candidate":
                candidate_calls += 1
            else:
                fallback_calls += 1
        return normalize_cache[key]

    def search(boundaries=None):
        """Count complete paths up to two, with exhaustive endpoints if needed."""
        @lru_cache(maxsize=None)
        def solve(unit_index, start):
            if unit_index == len(frozen_units):
                return ((),) if start == len(source_text) else ()

            target = frozen_units[unit_index]
            if unit_index == len(frozen_units) - 1:
                end_positions = (len(source_text),)
            elif boundaries is None:
                end_positions = range(start + 1, len(source_text))
            else:
                end_positions = boundaries[unit_index + 1]

            solutions = []
            for end in end_positions:
                if end <= start or not _canonical_boundary(source_text, end):
                    continue
                observed = normalized(
                    start, end, "candidate" if boundaries is not None else "fallback"
                )
                if _slice_certification(source_text[start:end], target, observed,
                                        identity_sha, mapping_policy) is not None:
                    for suffix in solve(unit_index + 1, end):
                        solutions.append(((start, end),) + suffix)
                        if len(solutions) >= 2:
                            return tuple(solutions)
            return tuple(solutions)

        return solve(0, 0)

    solutions = ()
    used_pruning = False
    used_fallback = False
    if guarded:
        last_digit = max(
            (i for i, char in enumerate(source_text) if "0" <= char <= "9"),
            default=-1,
        )
        # A valid partition cutting before the final digit must make one of
        # these positions its first cut. Defer to exhaustive search if any
        # such first unit is possible.
        early_match = any(
            _slice_certification(source_text[:end], frozen_units[0],
                                 normalized(0, end, "candidate"), identity_sha,
                                 mapping_policy) is not None
            for end in range(1, last_digit + 1)
            if _canonical_boundary(source_text, end)
        )
        if not early_match:
            used_pruning = True
            boundaries = _han_suffix_boundaries(
                source_text, frozen_units, last_digit + 1
            )
            solutions = search(boundaries)
    if not solutions:
        used_fallback = True
        solutions = search()

    if diagnostics is not None:
        diagnostics.update({
            "guarded_pruning_used": used_pruning,
            "fallback_exhaustive_used": used_fallback,
            "candidate_span_frontend_calls": candidate_calls,
            "fallback_exhaustive_calls": fallback_calls,
        })
    if not solutions:
        raise SynthesisUnitPlanningError(
            f"{scene_id} normalized units cannot be mapped to an exact, verified "
            "partition of the original scene text."
        )
    if len(solutions) != 1:
        raise SynthesisUnitPlanningError(
            f"{scene_id} normalized-unit source mapping is ambiguous; at least two "
            f"exact verified partitions exist: {solutions[0]} and {solutions[1]}."
        )
    return list(solutions[0]), normalization_calls


def _unit_record(scene, unit_index, local_start, local_end, normalized_text,
                 mapping):
    source_text = scene["narration_text"][local_start:local_end]
    preprocessed_text, preprocessing = preprocess_synthesis_text(source_text)
    scene_span = scene["source_span"]
    prefix = scene["narration_text"][:local_start]
    selected = scene["narration_text"][local_start:local_end]
    start_byte = scene_span["start_byte"] + len(prefix.encode("utf-8"))
    end_byte = start_byte + len(selected.encode("utf-8"))
    unit_id = f"{scene['id']}_unit_{unit_index:04d}"
    return {
        "id": unit_id,
        "order": unit_index,
        "source_span": {
            "start_character": scene_span["start_character"] + local_start,
            "end_character": scene_span["start_character"] + local_end,
            "start_byte": start_byte,
            "end_byte": end_byte,
        },
        "source_text": source_text,
        "source_text_sha256": text_sha256(source_text),
        "preprocessed_text": preprocessed_text,
        "preprocessed_text_sha256": text_sha256(preprocessed_text),
        "text_preprocessing": preprocessing,
        "normalized_text": normalized_text,
        "normalized_text_sha256": text_sha256(normalized_text),
        "mapping": mapping,
    }


def build_synthesis_unit_plan(manifest, frontend, diagnostics=None, *,
                              mapping_policy=SOURCE_MAPPING_POLICY):
    """Return a schema-5 copy after frontend normalization and source certification."""
    _require(
        manifest.get("schema_version") == 1 and manifest.get("status") == "planned",
        "Synthesis-unit preparation requires an untouched schema-version 1 run.",
    )
    policy = _planning_policy(mapping_policy)
    frontend_provenance = frontend.initialize()
    _require(
        isinstance(frontend_provenance, dict)
        and isinstance(frontend_provenance.get("identity"), dict),
        "Frontend must provide structured identity provenance.",
    )
    frontend_identity = frontend_provenance["identity"]
    frontend_identity_sha256 = canonical_sha256(frontend_identity)
    if diagnostics is not None:
        diagnostics.update({
            "authoritative_frontend_calls": 0,
            "candidate_span_frontend_calls": 0,
            "final_independent_verification_calls": 0,
            "fallback_exhaustive_calls": 0,
            "guarded_pruning_used": False,
            "fallback_exhaustive_used": False,
            "scenes": [],
        })

    prepared = json.loads(json.dumps(manifest))
    total_units = 0
    scene_records_for_hash = []
    for scene in prepared["scenes"]:
        scene_text = scene["narration_text"]
        clean_scene_text, _ = preprocess_synthesis_text(scene_text)
        frozen_units = frontend.normalize(clean_scene_text)
        if diagnostics is not None:
            diagnostics["authoritative_frontend_calls"] += 1
        _require(
            isinstance(frozen_units, list)
            and frozen_units
            and all(isinstance(item, str) and item for item in frozen_units),
            f"{scene['id']} frontend returned no valid normalized units.",
        )
        oversized = [
            (index, len(text)) for index, text in enumerate(frozen_units, 1)
            if len(text) > MAX_NORMALIZED_UNIT_CHARACTERS
        ]
        if oversized:
            details = ", ".join(
                f"unit {index}: {length}" for index, length in oversized
            )
            raise SynthesisUnitPlanningError(
                f"{scene['id']} contains unsupported oversized normalized units "
                f"({details} characters; maximum supported is "
                f"{MAX_NORMALIZED_UNIT_CHARACTERS}). No fallback splitting policy exists."
            )

        scene_diagnostics = {}
        partitions, search_calls = _find_unique_partition(
            scene["id"], scene_text, frozen_units, frontend,
            frontend_identity, scene_diagnostics, mapping_policy,
        )
        units = []
        for index, ((start, end), normalized_text) in enumerate(
                zip(partitions, frozen_units), 1):
            clean_slice, _ = preprocess_synthesis_text(scene_text[start:end])
            provenance = _slice_certification(
                scene_text[start:end], normalized_text,
                frontend.normalize(clean_slice), frontend_identity_sha256,
                mapping_policy,
            )
            if diagnostics is not None:
                diagnostics["final_independent_verification_calls"] += 1
            _require(
                provenance is not None,
                f"{scene['id']} unit {index} failed independent mapping verification.",
            )
            units.append(_unit_record(
                scene, index, start, end, normalized_text,
                provenance,
            ))

        _require(
            "".join(unit["source_text"] for unit in units) == scene_text,
            f"{scene['id']} unit source slices do not reconstruct narration text.",
        )
        scene["synthesis_units"] = units
        scene["synthesis_unit_mapping"] = {
            "policy": mapping_policy,
            "status": "verified",
            "unit_count": len(units),
            "source_reconstruction_sha256": text_sha256(scene_text),
            "frontend_verification_calls": search_calls + len(units),
        }
        if diagnostics is not None:
            diagnostics["candidate_span_frontend_calls"] += (
                scene_diagnostics["candidate_span_frontend_calls"]
            )
            diagnostics["fallback_exhaustive_calls"] += (
                scene_diagnostics["fallback_exhaustive_calls"]
            )
            diagnostics["guarded_pruning_used"] |= (
                scene_diagnostics["guarded_pruning_used"]
            )
            diagnostics["fallback_exhaustive_used"] |= (
                scene_diagnostics["fallback_exhaustive_used"]
            )
            diagnostics["scenes"].append({
                "scene_id": scene["id"], **scene_diagnostics,
            })
        total_units += len(units)
        scene_records_for_hash.append({
            "scene_id": scene["id"],
            "scene_text_sha256": scene["text_sha256"],
            "units": [{key: unit[key] for key in IMMUTABLE_UNIT_KEYS} for unit in units],
        })

    unit_plan_hash_input = {
        "source_scene_plan_hash": prepared["plan_hash"],
        "policy": policy,
        "frontend_identity_sha256": frontend_identity_sha256,
        "scenes": scene_records_for_hash,
    }
    unit_plan_hash = canonical_sha256(unit_plan_hash_input)
    prepared["schema_version"] = UNIT_PLAN_SCHEMA_VERSION
    prepared["status"] = "units_planned"
    prepared["synthesis_unit_plan"] = {
        "policy": policy,
        "frontend": frontend_provenance,
        "frontend_identity_sha256": frontend_identity_sha256,
        "source_scene_plan_hash": prepared["plan_hash"],
        "total_units": total_units,
        "ordered_unit_plan_sha256": unit_plan_hash,
    }
    return prepared


def _unit_hash_input(manifest):
    unit_plan = manifest["synthesis_unit_plan"]
    return {
        "source_scene_plan_hash": manifest["plan_hash"],
        "policy": unit_plan["policy"],
        "frontend_identity_sha256": unit_plan["frontend_identity_sha256"],
        "scenes": [{
            "scene_id": scene["id"],
            "scene_text_sha256": scene["text_sha256"],
            "units": [{key: unit[key] for key in IMMUTABLE_UNIT_KEYS}
                      for unit in scene["synthesis_units"]],
        } for scene in manifest["scenes"]],
    }


def _title_override(manifest, frontend):
    """Record one explicit period after a source heading when native text joins it."""
    scene = manifest["scenes"][0]
    heading = scene["narration_text"].splitlines()[0]
    if not CHAPTER_HEADING.fullmatch(heading) or _heading_has_terminal_punctuation(heading):
        return None
    normalized_heading = frontend.normalize_heading(heading)
    first = scene["synthesis_units"][0]
    _require(first["normalized_text"].startswith(normalized_heading),
             "Native normalized chapter title does not match the first unit.")
    synthesis_text = (normalized_heading + "。"
                      + first["normalized_text"][len(normalized_heading):])
    return {
        "policy": TITLE_PUNCTUATION_POLICY,
        "source_heading": heading,
        "source_heading_sha256": text_sha256(heading),
        "unit_id": first["id"],
        "baseline_normalized_text_sha256": first["normalized_text_sha256"],
        "synthesis_text": synthesis_text,
        "synthesis_text_sha256": text_sha256(synthesis_text),
        "inserted_character": "。",
        "insertion_index": len(normalized_heading),
        "original_source_unchanged": True,
    }


def _validate_title_override(manifest):
    override = manifest.get("title_synthesis_override")
    if override is None:
        return
    _require(isinstance(override, dict), "Title override provenance is invalid.")
    first_scene = manifest["scenes"][0]
    first = first_scene["synthesis_units"][0]
    heading = first_scene["narration_text"].splitlines()[0]
    index = override.get("insertion_index")
    baseline = first["normalized_text"]
    _require(
        CHAPTER_HEADING.fullmatch(heading) is not None
        and not _heading_has_terminal_punctuation(heading)
        and isinstance(index, int) and not isinstance(index, bool)
        and 0 < index <= len(baseline)
        and baseline[index - 1] not in "。！？?!；;"
        and override == {
            "policy": TITLE_PUNCTUATION_POLICY,
            "source_heading": heading,
            "source_heading_sha256": text_sha256(heading),
            "unit_id": first["id"],
            "baseline_normalized_text_sha256": first["normalized_text_sha256"],
            "synthesis_text": baseline[:index] + "。" + baseline[index:],
            "synthesis_text_sha256": text_sha256(baseline[:index] + "。" + baseline[index:]),
            "inserted_character": "。", "insertion_index": index,
            "original_source_unchanged": True,
        },
        "Title override provenance is invalid.",
    )


def validate_synthesis_unit_plan(run_directory, manifest,
                                 expected_frontend_identity_sha256=None):
    """Validate persisted unit identity without loading the model frontend."""
    _require(
        manifest.get("schema_version") == UNIT_PLAN_SCHEMA_VERSION
        and manifest.get("status") in {
            "units_planned", "generating", "generated", "generation_failed",
        },
        "Operation requires a schema-version 5 synthesis-unit run.",
    )
    validate_plan_identity(Path(run_directory), manifest)
    unit_plan = manifest.get("synthesis_unit_plan")
    _require(isinstance(unit_plan, dict), "Manifest has no synthesis-unit plan.")
    recorded_policy = unit_plan.get("policy", {})
    _require(isinstance(recorded_policy, dict), "Planning policy is incompatible.")
    mapping_policy = recorded_policy.get("source_mapping_policy")
    expected_policy = _planning_policy(mapping_policy)
    _require(unit_plan.get("policy") == expected_policy,
             "Requested synthesis-unit planning policy is incompatible.")
    frontend_provenance = unit_plan.get("frontend")
    _require(isinstance(frontend_provenance, dict),
             "Frontend provenance is invalid.")
    identity = frontend_provenance.get("identity")
    _require(
        isinstance(identity, dict)
        and canonical_sha256(identity) == unit_plan.get("frontend_identity_sha256"),
        "Frontend identity provenance does not match its hash.",
    )
    if expected_frontend_identity_sha256 is not None:
        _require(
            unit_plan["frontend_identity_sha256"]
            == expected_frontend_identity_sha256,
            "Requested frontend identity differs from the frozen unit plan.",
        )

    total_units = 0
    for scene in manifest["scenes"]:
        units = scene.get("synthesis_units")
        _require(isinstance(units, list) and units,
                 f"{scene['id']} has no synthesis units.")
        expected_start_character = scene["source_span"]["start_character"]
        expected_start_byte = scene["source_span"]["start_byte"]
        reconstructed = []
        for index, unit in enumerate(units, 1):
            expected_id = f"{scene['id']}_unit_{index:04d}"
            _require(unit.get("id") == expected_id and unit.get("order") == index,
                     f"{scene['id']} has invalid synthesis-unit identity or order.")
            span = unit.get("source_span", {})
            source_text = unit.get("source_text")
            _require(
                isinstance(span, dict)
                and all(isinstance(span.get(key), int) and not isinstance(span[key], bool)
                        for key in ("start_character", "end_character",
                                    "start_byte", "end_byte")),
                f"{unit['id']} source span is invalid.",
            )
            _require(
                span.get("start_character") == expected_start_character
                and span.get("start_byte") == expected_start_byte,
                f"{unit['id']} source span is not contiguous.",
            )
            _require(
                isinstance(source_text, str)
                and unit.get("source_text_sha256") == text_sha256(source_text),
                f"{unit['id']} source text does not match its hash.",
            )
            _require(
                span.get("end_character") - span.get("start_character")
                == len(source_text)
                and span.get("end_byte") - span.get("start_byte")
                == len(source_text.encode("utf-8")),
                f"{unit['id']} source span length is invalid.",
            )
            clean_text, preprocessing = preprocess_synthesis_text(source_text)
            _require(
                unit.get("preprocessed_text") == clean_text
                and unit.get("preprocessed_text_sha256") == text_sha256(clean_text)
                and unit.get("text_preprocessing") == preprocessing,
                f"{unit['id']} preprocessing provenance is invalid.",
            )
            normalized_text = unit.get("normalized_text")
            _require(
                isinstance(normalized_text, str) and normalized_text
                and unit.get("normalized_text_sha256")
                == text_sha256(normalized_text),
                f"{unit['id']} normalized text does not match its hash.",
            )
            mapping = unit.get("mapping", {})
            _require(isinstance(mapping, dict), "Mapping provenance is invalid.")
            equivalence = mapping.get("normalization_equivalence", {})
            _require(isinstance(equivalence, dict), "Mapping equivalence is invalid.")
            observed = equivalence.get("independent_normalized_text", normalized_text)
            expected_mapping = _slice_certification(
                source_text, normalized_text, (observed,),
                unit_plan["frontend_identity_sha256"], mapping_policy,
            )
            _require(
                expected_mapping is not None and mapping == expected_mapping,
                f"{unit['id']} mapping provenance is invalid.",
            )
            reconstructed.append(source_text)
            expected_start_character = span["end_character"]
            expected_start_byte = span["end_byte"]
        _require(
            expected_start_character == scene["source_span"]["end_character"]
            and expected_start_byte == scene["source_span"]["end_byte"]
            and "".join(reconstructed) == scene["narration_text"],
            f"{scene['id']} synthesis units do not reconstruct the full scene.",
        )
        mapping = scene.get("synthesis_unit_mapping", {})
        _require(
            mapping.get("status") == "verified"
            and mapping.get("policy") == mapping_policy
            and mapping.get("unit_count") == len(units)
            and mapping.get("source_reconstruction_sha256") == scene["text_sha256"],
            f"{scene['id']} source-mapping certification is invalid.",
        )
        total_units += len(units)

    _require(unit_plan.get("total_units") == total_units,
             "Synthesis-unit total is inconsistent.")
    _require(
        unit_plan.get("source_scene_plan_hash") == manifest["plan_hash"]
        and unit_plan.get("ordered_unit_plan_sha256")
        == canonical_sha256(_unit_hash_input(manifest)),
        "Ordered synthesis-unit plan identity is invalid.",
    )
    _validate_title_override(manifest)
    return manifest["scenes"]


def prepare_synthesis_unit_run(run_directory, frontend, diagnostics=None):
    """Atomically upgrade only an untouched D1 run to an immutable unit plan."""
    run_directory, manifest_path, manifest = read_manifest(run_directory)
    _require(
        manifest.get("schema_version") == 1 and manifest.get("status") == "planned",
        "Synthesis-unit preparation requires an untouched schema-version 1 run.",
    )
    validate_plan_identity(run_directory, manifest)
    prepared = build_synthesis_unit_plan(manifest, frontend, diagnostics)
    title = _title_override(prepared, frontend)
    if title is not None:
        prepared["title_synthesis_override"] = title
    validate_synthesis_unit_plan(run_directory, prepared)
    save_manifest(manifest_path, prepared)
    return prepared
