"""End-to-end orchestration for one planned audiobook chapter."""

from .manifest import create_planning_run
from .pipeline import generate_planned_run, utc_now
from .postprocessing import assemble_chapter
from .unit_planning import prepare_synthesis_unit_run
from .unit_execution import generate_units, assemble_units
from . import profiling


def run_chapter(
        source_path, chapter_id, run_id, output_root, backend, now=None,
        clock=utc_now):
    """Plan, generate once, and assemble only when every scene succeeds."""
    with profiling.span("workflow.planning"):
        run_directory, _ = create_planning_run(
            source_path, chapter_id, run_id, output_root, now=now
        )
    with profiling.span("workflow.generation"):
        manifest = generate_planned_run(run_directory, backend, clock=clock)
    if manifest["status"] != "generated":
        return run_directory, manifest
    with profiling.span("workflow.assembly"):
        manifest = assemble_chapter(run_directory, clock=clock)
    return run_directory, manifest


def run_unit_chapter(source_path, chapter_id, run_id, output_root, frontend,
                     backend, now=None, clock=utc_now):
    """Plan, certify, generate, and assemble using the validated unit path."""
    with profiling.span("workflow.planning"):
        run_directory, _ = create_planning_run(
            source_path, chapter_id, run_id, output_root, now=now
        )
    try:
        with profiling.span("workflow.unit_preparation"):
            prepare_synthesis_unit_run(run_directory, frontend)
    finally:
        if hasattr(frontend, "release"):
            with profiling.span("workflow.frontend_release"):
                frontend.release()
    with profiling.span("workflow.generation"):
        manifest = generate_units(run_directory, backend, clock=clock)
    if manifest["status"] == "generated":
        with profiling.span("workflow.assembly"):
            manifest = assemble_units(run_directory, clock=clock)
    return run_directory, manifest
