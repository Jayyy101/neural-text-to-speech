"""Create and persist the versioned Milestone D1 planning manifest."""

from datetime import datetime, timezone
import json
from pathlib import Path

from .planning import PlanningError, plan_chapter


SCHEMA_VERSION = 1


def create_planning_run(source_path, chapter_id, run_id, output_root, now=None):
    """Create a new run directory containing an exact source snapshot and manifest."""
    source_bytes, plan = plan_chapter(source_path, chapter_id, run_id)
    output_root = Path(output_root).expanduser().resolve()
    chapter_dir = output_root / chapter_id
    run_dir = chapter_dir / run_id

    chapter_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir()
    except FileExistsError as error:
        raise PlanningError(f"Target run directory already exists: {run_dir}") from error

    snapshot_path = run_dir / "source.txt"
    manifest_path = run_dir / "manifest.json"
    snapshot_path.write_bytes(source_bytes)

    created = now or datetime.now(timezone.utc)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "chapter_id": chapter_id,
        "run_id": run_id,
        "created_at_utc": created.astimezone(timezone.utc).isoformat(),
        "status": "planned",
        "source": {
            "original_filename": plan["original_source_filename"],
            "snapshot_path": "source.txt",
            "sha256": plan["source_sha256"],
            "byte_length": plan["source_byte_length"],
        },
        "planner": {
            "version": plan["planner_version"],
            "settings": plan["planner_settings"],
        },
        "plan_hash": plan["plan_hash"],
        "scenes": plan["scenes"],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir, manifest
