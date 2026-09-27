"""Command-line entry point for audiobook backend workflows."""

import argparse
from pathlib import Path

from .cosyvoice import (
    CosyVoiceAdapter, CosyVoiceFrontendAdapter, ensure_rl_model_view,
    file_sha256,
)
from .manifest import create_planning_run
from .pipeline import (
    GenerationError,
    generate_planned_run,
    regenerate_scene,
    resume_generation,
    read_manifest,
    validate_seed,
)
from .postprocessing import assemble_chapter, repair_scene
from .planning import PlanningError
from .unit_planning import prepare_synthesis_unit_run
from .unit_execution import assemble_units, generate_units
from .content_qc import DEFAULT_ASR_PYTHON
from .workflow import run_chapter, run_unit_chapter


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs/audiobooks"
DEFAULT_COSYVOICE_ROOT = Path.home() / "CosyVoice"
DEFAULT_RL_VIEW = REPOSITORY_ROOT / "outputs/model_views/cosyvoice3_rl"
VALIDATED_XIAOXIAO_WAV_SHA256 = "d00856f65e90b7449c1286af2c4ad61e656927d5d35ce818ee4e0d25a3e8544e"
VALIDATED_XIAOXIAO_TRANSCRIPT_FILE_SHA256 = "1c3c8da9cef04bf1537adaa86f089ae119a068b3a38b5ceb0b47ecb3dc1f9b83"


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


def add_frontend_options(command):
    command.add_argument("--cosyvoice-root", type=Path, default=DEFAULT_COSYVOICE_ROOT)
    command.add_argument("--model-dir", type=Path)
    command.add_argument("--wetext-asset-root", type=Path)


def add_backend_options(command):
    command.add_argument("run_directory", type=Path)
    add_adapter_options(command)


def create_adapter(args, *, model_dir=None):
    cosyvoice_root = args.cosyvoice_root.expanduser()
    selected_model = model_dir or args.model_dir or cosyvoice_root / "pretrained_models/Fun-CosyVoice3-0.5B"
    default_rl = Path(selected_model).expanduser().resolve() == DEFAULT_RL_VIEW.resolve()
    if default_rl and args.prompt_wav is None and args.prompt_text_file is None:
        reference = cosyvoice_root / "reference_audio"
        if (file_sha256(reference / "xiaoxiao_narrator_short.wav")
                != VALIDATED_XIAOXIAO_WAV_SHA256
                or file_sha256(reference / "xiaoxiao_narrator_short.txt")
                != VALIDATED_XIAOXIAO_TRANSCRIPT_FILE_SHA256):
            raise RuntimeError("Default Xiaoxiao reference differs from validated RL runs.")
    return CosyVoiceAdapter(
        cosyvoice_root=cosyvoice_root,
        model_dir=selected_model,
        prompt_wav=args.prompt_wav or cosyvoice_root / "reference_audio/xiaoxiao_narrator_short.wav",
        prompt_text_file=(
            args.prompt_text_file
            or cosyvoice_root / "reference_audio/xiaoxiao_narrator_short.txt"
        ),
        offline_cached_wetext=default_rl,
    )


def unit_model_dir(args):
    return args.model_dir or ensure_rl_model_view(
        args.cosyvoice_root, DEFAULT_RL_VIEW
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Validate and plan one UTF-8 chapter.")
    plan.add_argument("source", type=Path, help="UTF-8 chapter text file.")
    plan.add_argument("--chapter-id", required=True)
    plan.add_argument("--run-id", required=True)
    plan.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    prepare_units = commands.add_parser(
        "prepare-units",
        help="Freeze and verify CosyVoice frontend units for a planned run.",
    )
    prepare_units.add_argument("run_directory", type=Path)
    add_frontend_options(prepare_units)
    run = commands.add_parser(
        "run", help="Plan, certify, generate, validate, and assemble one chapter with CosyVoice3 RL."
    )
    run.add_argument("source", type=Path, help="UTF-8 chapter text file.")
    run.add_argument("--chapter-id", required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    add_adapter_options(run)
    run.add_argument("--legacy-scenes", action="store_true",
                     help="Use the historical scene workflow for compatibility.")
    generate = commands.add_parser(
        "generate", help="Generate attempt_001 for an existing D1 planned run."
    )
    add_backend_options(generate)
    generate.add_argument(
        "--root-seed", type=cli_seed,
        help="Optional root seed for a new unit-planned run; otherwise generated and persisted.",
    )
    generate.add_argument("--asr-python", default=DEFAULT_ASR_PYTHON,
                          help="Pinned persistent Mandarin ASR worker interpreter.")
    resume = commands.add_parser(
        "resume", help="Generate one new attempt for each incomplete scene."
    )
    add_backend_options(resume)
    resume.add_argument("--asr-python", default=DEFAULT_ASR_PYTHON,
                        help="Pinned persistent Mandarin ASR worker interpreter.")
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

    if args.command == "prepare-units":
        cosyvoice_root = args.cosyvoice_root.expanduser()
        try:
            frontend = CosyVoiceFrontendAdapter(
                cosyvoice_root=cosyvoice_root,
                model_dir=unit_model_dir(args),
                wetext_asset_root=args.wetext_asset_root,
                offline_cached_wetext=args.model_dir is None,
            )
            manifest = prepare_synthesis_unit_run(
                args.run_directory, frontend
            )
        except (PlanningError, GenerationError, OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))
        unit_plan = manifest["synthesis_unit_plan"]
        print(
            f"Prepared {unit_plan['total_units']} synthesis unit(s) across "
            f"{len(manifest['scenes'])} scene(s)."
        )
        print(
            f"Unit plan SHA-256: {unit_plan['ordered_unit_plan_sha256']}"
        )
        print(
            f"Manifest: "
            f"{Path(args.run_directory).expanduser().resolve() / 'manifest.json'}"
        )
        return 0

    if args.command == "run":
        try:
            if args.legacy_scenes:
                adapter = create_adapter(args)
                run_directory, manifest = run_chapter(
                    args.source, args.chapter_id, args.run_id,
                    args.output_root, adapter,
                )
            else:
                model_dir = unit_model_dir(args)
                adapter = create_adapter(args, model_dir=model_dir)
                frontend = CosyVoiceFrontendAdapter(
                    args.cosyvoice_root, model_dir,
                    offline_cached_wetext=args.model_dir is None,
                )
                run_directory, manifest = run_unit_chapter(
                    args.source, args.chapter_id, args.run_id,
                    args.output_root, frontend, adapter,
                )
        except (PlanningError, GenerationError, OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))
        summary = manifest["generation"].get("summary")
        if summary:
            if args.legacy_scenes:
                print(f"Generated {summary['generated_scenes']} of "
                      f"{summary['total_scenes']} scene(s); "
                      f"failures: {summary['failed_scenes']}.")
            else:
                print(f"Accepted {summary['generated_units']} of "
                      f"{summary['total_units']} unit(s); "
                      f"unresolved: {summary['failed_units']}.")
        if manifest["status"] != "generated":
            print("Generation incomplete; chapter assembly was not attempted.")
            print(f"Run directory: {run_directory}")
            print(f"Manifest: {run_directory / 'manifest.json'}")
            return 1
        assembly = manifest["assembly"]
        label = "scenes" if args.legacy_scenes else "units"
        print(f"Assembled {len(assembly[label])} {label}, "
              f"{assembly['audio']['frames']} frames.")
        print(f"Chapter: {run_directory / assembly['output_path']}")
        print(f"Manifest: {run_directory / 'manifest.json'}")
        return 0

    if args.command in {"repair", "assemble"}:
        try:
            if args.command == "repair":
                if read_manifest(args.run_directory)[2].get("schema_version") == 5:
                    raise GenerationError(
                        "Manual scene repairs are unsupported for unit-planned runs."
                    )
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
                schema = read_manifest(args.run_directory)[2].get("schema_version")
                manifest = (
                    assemble_units(args.run_directory) if schema == 5
                    else assemble_chapter(args.run_directory)
                )
                assembly = manifest["assembly"]
                print(
                    f"Assembled {len(assembly.get('units', assembly.get('scenes')))} "
                    f"{'unit(s)' if schema == 5 else 'scene(s)'}, "
                    f"{assembly['audio']['frames']} frames."
                )
        except (GenerationError, OSError, ValueError) as error:
            parser.error(str(error))
        print(
            f"Manifest: "
            f"{Path(args.run_directory).expanduser().resolve() / 'manifest.json'}"
        )
        return 0

    try:
        existing = read_manifest(args.run_directory)[2]
        schema = existing.get("schema_version")
        model_dir = args.model_dir
        if schema == 5 and model_dir is None:
            model_dir = existing["synthesis_unit_plan"]["frontend"].get(
                "locations", {}
            ).get("model_dir")
        adapter = create_adapter(args, model_dir=model_dir)
        if args.command == "generate" and schema != 5 and args.root_seed is not None:
            raise GenerationError("--root-seed applies only to unit-planned runs.")
        if schema == 5 and args.command in {"generate", "resume"}:
            manifest = generate_units(
                args.run_directory, adapter,
                root_seed=args.root_seed if args.command == "generate" else None,
                asr_python=args.asr_python,
            )
        elif args.command == "generate":
            manifest = generate_planned_run(args.run_directory, adapter)
        elif args.command == "resume":
            manifest = resume_generation(args.run_directory, adapter)
        else:
            manifest = regenerate_scene(
                args.run_directory, args.scene_id, adapter, seed=args.seed
            )
    except (GenerationError, OSError, RuntimeError) as error:
        parser.error(str(error))
    generation_state = manifest["generation"]
    summary = generation_state.get("summary")
    if summary:
        if schema == 5:
            print(
                f"Accepted {summary['generated_units']} of {summary['total_units']} "
                f"unit(s); unresolved: {summary['failed_units']}."
            )
        else:
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
