"""Command-line entry point for audiobook backend workflows."""

import argparse
from pathlib import Path

from .cosyvoice import CosyVoiceAdapter
from .manifest import create_planning_run
from .pipeline import (
    GenerationError,
    generate_planned_run,
    regenerate_scene,
    resume_generation,
    validate_seed,
)
from .postprocessing import assemble_chapter, repair_scene
from .planning import PlanningError
from .workflow import run_chapter


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/audiobooks"
DEFAULT_COSYVOICE_ROOT = Path.home() / "CosyVoice"


def cli_seed(value):
    if isinstance(value, bool):
        raise argparse.ArgumentTypeError(
            "seed must be an integer from 0 through 4294967295"
        )
    try:
        seed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "seed must be an integer from 0 through 4294967295"
        ) from error
    try:
        return validate_seed(seed)
    except GenerationError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def add_adapter_options(command):
    command.add_argument("--cosyvoice-root", type=Path, default=DEFAULT_COSYVOICE_ROOT)
    command.add_argument("--model-dir", type=Path)
    command.add_argument("--prompt-wav", type=Path)
    command.add_argument("--prompt-text-file", type=Path)


def add_backend_options(command):
    command.add_argument("run_directory", type=Path)
    add_adapter_options(command)


def create_adapter(args):
    cosyvoice_root = args.cosyvoice_root.expanduser()
    return CosyVoiceAdapter(
        cosyvoice_root=cosyvoice_root,
        model_dir=args.model_dir or cosyvoice_root / "pretrained_models/Fun-CosyVoice3-0.5B",
        prompt_wav=args.prompt_wav or cosyvoice_root / "reference_audio/xiaoxiao_narrator_short.wav",
        prompt_text_file=(
            args.prompt_text_file
            or cosyvoice_root / "reference_audio/xiaoxiao_narrator_short.txt"
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Validate and plan one UTF-8 chapter.")
    plan.add_argument("source", type=Path, help="UTF-8 chapter text file.")
    plan.add_argument("--chapter-id", required=True)
    plan.add_argument("--run-id", required=True)
    plan.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run = commands.add_parser(
        "run", help="Plan, generate, validate, and assemble one chapter."
    )
    run.add_argument("source", type=Path, help="UTF-8 chapter text file.")
    run.add_argument("--chapter-id", required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    add_adapter_options(run)
    generate = commands.add_parser(
        "generate", help="Generate attempt_001 for an existing D1 planned run."
    )
    add_backend_options(generate)
    resume = commands.add_parser(
        "resume", help="Generate one new attempt for each incomplete scene."
    )
    add_backend_options(resume)
    regenerate = commands.add_parser(
        "regenerate", help="Generate one new attempt for one scene."
    )
    add_backend_options(regenerate)
    regenerate.add_argument("--scene-id", required=True)
    regenerate.add_argument("--seed", type=cli_seed)
    repair = commands.add_parser(
        "repair", help="Apply a source-bound manual pause plan to one scene."
    )
    repair.add_argument("run_directory", type=Path)
    repair.add_argument("--scene-id", required=True)
    repair.add_argument("--plan", type=Path, required=True)
    assemble = commands.add_parser(
        "assemble", help="Assemble selected scene artifacts into one chapter WAV."
    )
    assemble.add_argument("run_directory", type=Path)
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

    if args.command == "run":
        adapter = create_adapter(args)
        try:
            run_directory, manifest = run_chapter(
                args.source, args.chapter_id, args.run_id,
                args.output_root, adapter,
            )
        except (PlanningError, GenerationError, OSError, ValueError) as error:
            parser.error(str(error))
        summary = manifest["generation"].get("summary")
        if summary:
            print(
                f"Generated {summary['generated_scenes']} of "
                f"{summary['total_scenes']} scene(s); "
                f"failures: {summary['failed_scenes']}."
            )
        if manifest["status"] != "generated":
            print("Generation incomplete; chapter assembly was not attempted.")
            print(f"Run directory: {run_directory}")
            print(f"Manifest: {run_directory / 'manifest.json'}")
            return 1
        assembly = manifest["assembly"]
        print(
            f"Assembled {len(assembly['scenes'])} scene(s), "
            f"{assembly['audio']['frames']} frames."
        )
        print(f"Chapter: {run_directory / assembly['output_path']}")
        print(f"Manifest: {run_directory / 'manifest.json'}")
        return 0

    if args.command in {"repair", "assemble"}:
        try:
            if args.command == "repair":
                manifest = repair_scene(
                    args.run_directory, args.scene_id, args.plan
                )
                repair_state = next(
                    scene["repair"] for scene in manifest["scenes"]
                    if scene["id"] == args.scene_id
                )
                print(
                    f"Selected {repair_state['selected_repair_id']} for {args.scene_id}."
                )
            else:
                manifest = assemble_chapter(args.run_directory)
                assembly = manifest["assembly"]
                print(
                    f"Assembled {len(assembly['scenes'])} scene(s), "
                    f"{assembly['audio']['frames']} frames."
                )
        except (GenerationError, OSError, ValueError) as error:
            parser.error(str(error))
        print(
            f"Manifest: "
            f"{Path(args.run_directory).expanduser().resolve() / 'manifest.json'}"
        )
        return 0

    adapter = create_adapter(args)
    try:
        if args.command == "generate":
            manifest = generate_planned_run(args.run_directory, adapter)
        elif args.command == "resume":
            manifest = resume_generation(args.run_directory, adapter)
        else:
            manifest = regenerate_scene(
                args.run_directory, args.scene_id, adapter, seed=args.seed
            )
    except (GenerationError, OSError) as error:
        parser.error(str(error))
    generation_state = manifest["generation"]
    summary = generation_state.get("summary")
    if summary:
        print(
            f"Generated {summary['generated_scenes']} of {summary['total_scenes']} "
            f"scene(s); failures: {summary['failed_scenes']}."
        )
    else:
        print("CosyVoice initialization failed; no scenes were generated.")
    operation = generation_state.get("last_operation")
    if operation:
        print(
            f"{operation['type']}: {operation['status']}; "
            f"attempted {operation['attempted_scenes']} scene(s)."
        )
        if operation["status"] == "duplicate":
            print(
                f"Requested seed produced duplicate take "
                f"{operation['duplicate_attempt_id']} matching "
                f"{operation['duplicate_of_attempt_id']}; previous selection retained."
            )
    print(f"Manifest: {Path(args.run_directory).expanduser().resolve() / 'manifest.json'}")
    return int(
        manifest["status"] != "generated"
        or (operation is not None and operation["status"] in {"failed", "duplicate"})
    )


if __name__ == "__main__":
    raise SystemExit(main())
