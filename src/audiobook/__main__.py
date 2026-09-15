"""Command-line entry point for audiobook backend workflows."""

import argparse
from pathlib import Path

from .manifest import create_planning_run
from .planning import PlanningError


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/audiobooks"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Validate and plan one UTF-8 chapter.")
    plan.add_argument("source", type=Path, help="UTF-8 chapter text file.")
    plan.add_argument("--chapter-id", required=True)
    plan.add_argument("--run-id", required=True)
    plan.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)

    try:
        run_dir, manifest = create_planning_run(
            args.source, args.chapter_id, args.run_id, args.output_root
        )
    except (PlanningError, OSError) as error:
        parser.error(str(error))

    print(f"Planned {len(manifest['scenes'])} scene(s).")
    print(f"Run directory: {run_dir}")
    print(f"Manifest: {run_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
