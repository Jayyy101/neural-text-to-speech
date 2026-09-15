"""Command-line entry point for audiobook backend workflows."""

import argparse
from pathlib import Path

from .cosyvoice import CosyVoiceAdapter
from .manifest import create_planning_run
from .pipeline import GenerationError, generate_planned_run
from .planning import PlanningError


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/audiobooks"
DEFAULT_COSYVOICE_ROOT = Path.home() / "CosyVoice"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Validate and plan one UTF-8 chapter.")
    plan.add_argument("source", type=Path, help="UTF-8 chapter text file.")
    plan.add_argument("--chapter-id", required=True)
    plan.add_argument("--run-id", required=True)
    plan.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    generate = commands.add_parser(
        "generate", help="Generate attempt_001 for an existing D1 planned run."
    )
    generate.add_argument("run_directory", type=Path)
    generate.add_argument("--cosyvoice-root", type=Path, default=DEFAULT_COSYVOICE_ROOT)
    generate.add_argument("--model-dir", type=Path)
    generate.add_argument("--prompt-wav", type=Path)
    generate.add_argument("--prompt-text-file", type=Path)
    args = parser.parse_args(argv)

    if args.command == "plan":
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

    cosyvoice_root = args.cosyvoice_root.expanduser()
    adapter = CosyVoiceAdapter(
        cosyvoice_root=cosyvoice_root,
        model_dir=args.model_dir or cosyvoice_root / "pretrained_models/Fun-CosyVoice3-0.5B",
        prompt_wav=args.prompt_wav or cosyvoice_root / "reference_audio/xiaoxiao_narrator_short.wav",
        prompt_text_file=(
            args.prompt_text_file
            or cosyvoice_root / "reference_audio/xiaoxiao_narrator_short.txt"
        ),
    )
    try:
        manifest = generate_planned_run(args.run_directory, adapter)
    except (GenerationError, OSError) as error:
        parser.error(str(error))
    summary = manifest["generation"].get("summary")
    if summary:
        print(
            f"Generated {summary['generated_scenes']} of {summary['total_scenes']} "
            f"scene(s); failures: {summary['failed_scenes']}."
        )
    else:
        print("CosyVoice initialization failed; no scenes were generated.")
    print(f"Manifest: {Path(args.run_directory).expanduser().resolve() / 'manifest.json'}")
    return int(manifest["status"] != "generated")


if __name__ == "__main__":
    raise SystemExit(main())
