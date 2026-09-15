"""Validate a UTF-8 chapter and build a deterministic explicit-scene plan."""

import hashlib
import json
from pathlib import Path
import re


PLANNER_VERSION = 1
SCENE_MARKER = "***"
PLANNER_SETTINGS = {
    "scene_marker": SCENE_MARKER,
    "marker_rule": "trimmed_standalone_line",
}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class PlanningError(ValueError):
    """Raised when source text or planning settings cannot form a valid plan."""


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def validate_id(label, value):
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise PlanningError(
            f"{label} must start with an ASCII letter or digit and contain only "
            "ASCII letters, digits, underscores, or hyphens."
        )


def decode_source(source_bytes):
    """Decode UTF-8 source, accepting and excluding an optional BOM from narration."""
    try:
        decoded = source_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PlanningError(f"Chapter source is not valid UTF-8: {error}") from error
    if decoded.startswith("\ufeff"):
        return decoded[1:], 1, 3
    return decoded, 0, 0


def marker_spans(text):
    """Return character spans for trimmed standalone marker lines."""
    spans = []
    offset = 0
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        if content.strip() == SCENE_MARKER:
            spans.append((offset, offset + len(line)))
        offset += len(line)
    return spans


def line_number_at(text, position):
    return text.count("\n", 0, position) + 1


def scene_record(text, start, end, order, character_offset_base, byte_offset_base):
    narration_text = text[start:end]
    logical_end = end - 1 if end > start else end
    encoded = narration_text.encode("utf-8")
    return {
        "id": f"scene_{order:04d}",
        "order": order,
        "source_span": {
            "start_character": character_offset_base + start,
            "end_character": character_offset_base + end,
            "start_byte": byte_offset_base + len(text[:start].encode("utf-8")),
            "end_byte": byte_offset_base + len(text[:end].encode("utf-8")),
            "start_line": line_number_at(text, start),
            "end_line": line_number_at(text, logical_end),
        },
        "narration_text": narration_text,
        "text_sha256": sha256_bytes(encoded),
    }


def build_plan(source_bytes):
    """Return deterministic source and scene metadata without run-specific fields."""
    text, character_offset_base, byte_offset_base = decode_source(source_bytes)
    if not text.strip():
        raise PlanningError("Chapter source must not be empty or whitespace-only.")

    spans = marker_spans(text)
    boundaries = []
    start = 0
    for marker_start, marker_end in spans:
        boundaries.append((start, marker_start))
        start = marker_end
    boundaries.append((start, len(text)))

    for index, (scene_start, scene_end) in enumerate(boundaries, 1):
        if not text[scene_start:scene_end].strip():
            raise PlanningError(
                f"Scene marker placement creates an empty scene at position {index}."
            )

    scenes = [
        scene_record(
            text, scene_start, scene_end, index,
            character_offset_base, byte_offset_base,
        )
        for index, (scene_start, scene_end) in enumerate(boundaries, 1)
    ]
    source_sha256 = sha256_bytes(source_bytes)
    hash_input = {
        "planner_version": PLANNER_VERSION,
        "planner_settings": PLANNER_SETTINGS,
        "source_sha256": source_sha256,
        "scenes": scenes,
    }
    canonical = json.dumps(
        hash_input, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "source_sha256": source_sha256,
        "source_byte_length": len(source_bytes),
        "planner_version": PLANNER_VERSION,
        "planner_settings": dict(PLANNER_SETTINGS),
        "plan_hash": sha256_bytes(canonical),
        "scenes": scenes,
    }


def plan_chapter(source_path, chapter_id, run_id):
    """Read and validate one source file, returning its bytes and logical plan."""
    validate_id("chapter ID", chapter_id)
    validate_id("run ID", run_id)
    source_path = Path(source_path).expanduser().resolve()
    try:
        source_bytes = source_path.read_bytes()
    except OSError as error:
        raise PlanningError(f"Cannot read chapter source: {error}") from error
    plan = build_plan(source_bytes)
    plan["original_source_filename"] = source_path.name
    return source_bytes, plan
