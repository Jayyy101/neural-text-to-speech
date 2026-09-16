"""End-to-end orchestration for one planned audiobook chapter."""

from .manifest import create_planning_run
from .pipeline import generate_planned_run, utc_now
from .postprocessing import assemble_chapter


def run_chapter(
        source_path, chapter_id, run_id, output_root, backend, now=None,
        clock=utc_now):
    """Plan, generate once, and assemble only when every scene succeeds."""
    run_directory, _ = create_planning_run(
        source_path, chapter_id, run_id, output_root, now=now
    )
    manifest = generate_planned_run(run_directory, backend, clock=clock)
    if manifest["status"] != "generated":
        return run_directory, manifest
    manifest = assemble_chapter(run_directory, clock=clock)
    return run_directory, manifest
