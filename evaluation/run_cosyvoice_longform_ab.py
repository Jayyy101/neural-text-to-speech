"""Prepare and run the evaluation-only giant-versus-moderate chapter benchmark.

Without --execute this validates and prints the fixed scene plan without importing
or running CosyVoice. Each explicit execution runs exactly one condition/seed pair.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.run_cosyvoice_prompt_conditioning_ab import (
    configure_local_wetext_frontend,
)
from src.audiobook.cosyvoice import (
    CosyVoiceAdapter,
    file_sha256,
    set_cosyvoice_random_seed,
)
from src.audiobook.manifest import create_planning_run
from src.audiobook.pipeline import generate_planned_run
from src.audiobook.planning import (
    SCENE_MARKER,
    build_plan,
    decode_source,
    line_number_at,
)
from src.audiobook.postprocessing import assemble_chapter


PLAN_PATH = ROOT / "evaluation/inputs/cosyvoice_longform_ab_plan.json"
DEFAULT_OUTPUT_PARENT = ROOT / "outputs/evaluation/cosyvoice_longform_ab"
DEFAULT_COSYVOICE_ROOT = Path.home() / "CosyVoice"


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def text_sha256(value):
    return sha256_bytes(value.encode("utf-8"))


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def short_preview(text, width=18):
    compact = " ".join(text.strip().split())
    if len(compact) <= width:
        return compact
    return compact[:width] + "…"


def _package_version(distribution):
    try:
        return package_metadata.version(distribution)
    except package_metadata.PackageNotFoundError:
        return None


def load_plan(path=PLAN_PATH):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1:
        raise ValueError("Long-form benchmark plan must use schema version 1.")
    if plan.get("scene_marker") != SCENE_MARKER:
        raise ValueError("Benchmark marker differs from the Milestone D marker.")

    seeds = plan.get("seeds")
    if seeds != [2026091702, 2026091703]:
        raise ValueError("Benchmark seeds differ from the approved fixed seeds.")
    order = plan.get("execution_order")
    expected_order = [
        {"order": 1, "condition": "giant", "seed": 2026091702},
        {"order": 2, "condition": "moderate", "seed": 2026091702},
        {"order": 3, "condition": "moderate", "seed": 2026091703},
        {"order": 4, "condition": "giant", "seed": 2026091703},
    ]
    if order != expected_order:
        raise ValueError("Benchmark execution order differs from the approved order.")
    return plan


def _original_byte_offset(text, position, bom_bytes):
    return bom_bytes + len(text[:position].encode("utf-8"))


def _scene_records(original_text, boundaries, bom_bytes):
    positions = [0] + [item["start_character"] for item in boundaries] + [
        len(original_text)
    ]
    records = []
    for index, (start, end) in enumerate(zip(positions, positions[1:]), 1):
        scene_text = original_text[start:end]
        before = (
            "Chapter start; title remains with the coherent opening passage."
            if index == 1 else boundaries[index - 2]["reason"]
        )
        after = boundaries[index - 1]["reason"] if index <= len(boundaries) else None
        records.append({
            "scene_id": f"scene_{index:04d}",
            "order": index,
            "original_source_span": {
                "start_character": start,
                "end_character": end,
                "start_byte": _original_byte_offset(original_text, start, bom_bytes),
                "end_byte": _original_byte_offset(original_text, end, bom_bytes),
                "start_line": line_number_at(original_text, start),
                "end_line": line_number_at(
                    original_text, end - 1 if end > start else end
                ),
            },
            "source_character_count": end - start,
            "text_sha256": text_sha256(scene_text),
            "begin_preview": short_preview(scene_text),
            "end_preview": short_preview(scene_text[::-1])[::-1],
            "boundary_reason_before": before,
            "boundary_reason_after": after,
        })
    return records


def prepare_design(plan, source_path=None):
    source_path = (
        Path(source_path).expanduser().resolve()
        if source_path is not None
        else (ROOT / plan["source_path"]).resolve()
    )
    original_bytes = source_path.read_bytes()
    if sha256_bytes(original_bytes) != plan["source_sha256"]:
        raise ValueError("Original chapter does not match the approved SHA-256.")
    original_text, character_base, byte_base = decode_source(original_bytes)
    if len(original_text) != plan["source_character_count"]:
        raise ValueError("Original chapter character count differs from the plan.")
    if character_base not in {0, 1} or byte_base not in {0, 3}:
        raise ValueError("Unexpected UTF-8 source prefix.")

    boundaries = plan.get("moderate_boundaries")
    if not isinstance(boundaries, list) or len(boundaries) != 6:
        raise ValueError("Moderate condition must contain exactly six boundaries.")
    positions = [item.get("start_character") for item in boundaries]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in positions):
        raise ValueError("Every moderate boundary must have an integer character offset.")
    if positions != sorted(set(positions)):
        raise ValueError("Moderate boundary offsets must be unique and increasing.")

    for boundary in boundaries:
        position = boundary["start_character"]
        if position <= 0 or position >= len(original_text):
            raise ValueError("Moderate boundary lies outside the chapter.")
        if position > 0 and original_text[position - 1] != "\n":
            raise ValueError(f"Boundary {position} is not at the start of a line.")
        if not original_text.startswith(boundary["starts_with"], position):
            raise ValueError(f"Boundary anchor does not match at character {position}.")
        if not isinstance(boundary.get("reason"), str) or not boundary["reason"].strip():
            raise ValueError(f"Boundary {position} has no reason.")

    newline = "\r\n" if "\r\n" in original_text else "\n"
    prepared_text = original_text
    marker_line = SCENE_MARKER + newline
    for position in reversed(positions):
        prepared_text = prepared_text[:position] + marker_line + prepared_text[position:]
    bom = b"\xef\xbb\xbf" if byte_base else b""
    prepared_bytes = bom + prepared_text.encode("utf-8")
    if sha256_bytes(prepared_bytes) != plan["prepared_moderate_sha256"]:
        raise ValueError("Prepared moderate source SHA-256 differs from the plan.")

    giant_plan = build_plan(original_bytes)
    moderate_plan = build_plan(prepared_bytes)
    if len(giant_plan["scenes"]) != 1:
        raise ValueError("Giant condition must remain exactly one Milestone D scene.")
    if len(moderate_plan["scenes"]) != 7:
        raise ValueError("Moderate condition must produce exactly seven scenes.")

    reconstructed = "".join(
        scene["narration_text"] for scene in moderate_plan["scenes"]
    )
    if reconstructed != original_text:
        raise ValueError("Prepared scenes omit, duplicate, or alter original narration text.")

    scene_records = _scene_records(original_text, boundaries, byte_base)
    for record, scene in zip(scene_records, moderate_plan["scenes"]):
        span = record["original_source_span"]
        expected = original_text[span["start_character"]:span["end_character"]]
        if scene["narration_text"] != expected:
            raise ValueError(f"{record['scene_id']} does not match its original span.")

    giant_scene = _scene_records(original_text, [], byte_base)
    return {
        "schema_version": 1,
        "experiment_id": plan["experiment_id"],
        "source": {
            "path": str(source_path),
            "sha256": sha256_bytes(original_bytes),
            "byte_length": len(original_bytes),
            "character_count": len(original_text),
        },
        "prepared_moderate_source": {
            "sha256": sha256_bytes(prepared_bytes),
            "byte_length": len(prepared_bytes),
            "character_count_with_markers": len(prepared_text),
            "inserted_marker_count": len(boundaries),
            "marker_line_ending": "CRLF" if newline == "\r\n" else "LF",
        },
        "title_opening_policy": plan["title_opening_policy"],
        "seeds": list(plan["seeds"]),
        "execution_order": list(plan["execution_order"]),
        "conditions": {
            "giant": {
                "scene_count": 1,
                "milestone_d_plan_hash": giant_plan["plan_hash"],
                "scenes": giant_scene,
            },
            "moderate": {
                "scene_count": 7,
                "milestone_d_plan_hash": moderate_plan["plan_hash"],
                "scenes": scene_records,
                "boundaries": boundaries,
            },
        },
        "_source_path": source_path,
        "_original_bytes": original_bytes,
        "_prepared_bytes": prepared_bytes,
    }


def public_design(design):
    return {key: value for key, value in design.items() if not key.startswith("_")}


def schedule_entry(plan, condition, seed):
    for entry in plan["execution_order"]:
        if entry["condition"] == condition and entry["seed"] == seed:
            return dict(entry)
    raise ValueError("Condition/seed pair is outside the approved benchmark schedule.")


def bundle_name(entry):
    return f"{entry['order']:02d}_{entry['condition']}_seed_{entry['seed']}"


def create_bundle_directory(output_parent, entry):
    output_parent = Path(output_parent).expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    bundle = output_parent / bundle_name(entry)
    try:
        bundle.mkdir()
    except FileExistsError as error:
        raise FileExistsError(f"Benchmark output already exists: {bundle}") from error
    return bundle


def show_plan(design):
    print("CosyVoice long-form GIANT versus MODERATE benchmark plan")
    print("Original source SHA-256:", design["source"]["sha256"])
    print("Prepared source SHA-256:", design["prepared_moderate_source"]["sha256"])
    print("Title/opening:", design["title_opening_policy"])
    print("GIANT scenes: 1")
    print("MODERATE scenes:")
    for scene in design["conditions"]["moderate"]["scenes"]:
        print(
            f"  {scene['scene_id']}: {scene['source_character_count']} chars; "
            f"{scene['begin_preview']} / …{scene['end_preview']}"
        )
        if scene["boundary_reason_after"]:
            print("    next boundary:", scene["boundary_reason_after"])
    print("Execution order:")
    for entry in design["execution_order"]:
        print(f"  {entry['order']}. {entry['condition']} seed={entry['seed']}")
    print("No synthesis runs without the explicit --execute flag.")


class ObservedSeededBackend:
    """Evaluation wrapper that seeds once and observes the real frontend calls."""

    def __init__(self, adapter, seed, scene_ids):
        self.adapter = adapter
        self.seed = seed
        self.scene_ids = list(scene_ids)
        self.initialize_seconds = None
        self.frontend_metadata = None
        self.rng_resets = 0
        self.scene_wall_seconds = {}
        self.normalized_units = {}
        self._scene_index = 0
        self._current_scene_id = None
        self._current_text = None

    def configuration(self):
        return self.adapter.configuration()

    def initialize(self):
        started = time.perf_counter()
        metadata = self.adapter.initialize()
        frontend = self.adapter._model.frontend
        self.frontend_metadata = configure_local_wetext_frontend(frontend)
        self.frontend_metadata["package_distribution"] = "wetext"
        self.frontend_metadata["package_version"] = _package_version("wetext")

        original_normalize = frontend.text_normalize

        def observed_normalize(text, split=True, text_frontend=True):
            result = original_normalize(
                text, split=split, text_frontend=text_frontend
            )
            if (
                self._current_scene_id is not None
                and text == self._current_text
                and split is True
                and text_frontend is True
            ):
                self.normalized_units[self._current_scene_id] = list(result)
            return result

        frontend.text_normalize = observed_normalize
        set_cosyvoice_random_seed(self.seed)
        self.rng_resets += 1
        self.initialize_seconds = time.perf_counter() - started
        return {
            **metadata,
            "evaluation_frontend": self.frontend_metadata,
            "benchmark_random_state": {
                "policy": "single_global_seed_before_full_condition",
                "seed": self.seed,
                "scope": ["python", "numpy", "torch_cpu", "torch_cuda"],
                "rng_resets": self.rng_resets,
                "reseeded_between_scenes": False,
                "applied_after_model_initialization": True,
            },
        }

    def generate_scene(self, text, output_path):
        if self._scene_index >= len(self.scene_ids):
            raise RuntimeError("Benchmark generated more scenes than planned.")
        scene_id = self.scene_ids[self._scene_index]
        self._current_scene_id = scene_id
        self._current_text = text
        started = time.perf_counter()
        try:
            return self.adapter.generate_scene(text, output_path)
        finally:
            self.scene_wall_seconds[scene_id] = time.perf_counter() - started
            self._scene_index += 1
            self._current_scene_id = None
            self._current_text = None


def _attempt_for_scene(scene):
    attempts = scene.get("generation", {}).get("attempts", [])
    return attempts[0] if attempts else None


def _scene_results(design, condition, manifest, backend):
    planned = {
        item["scene_id"]: item
        for item in design["conditions"][condition]["scenes"]
    }
    results = []
    for scene in manifest["scenes"]:
        scene_id = scene["id"]
        attempt = _attempt_for_scene(scene)
        units = backend.normalized_units.get(scene_id, [])
        record = {
            **planned[scene_id],
            "normalized_unit_count": len(units),
            "normalized_units": [
                {"text": text, "text_sha256": text_sha256(text)} for text in units
            ],
            "scene_wall_seconds": backend.scene_wall_seconds.get(scene_id),
            "attempt_status": attempt.get("status") if attempt else "not_run",
        }
        if attempt:
            for key in (
                "output_path", "wav_sha256", "audio", "cosyvoice_chunks",
                "inference_seconds", "rtf", "peak_torch_cuda_allocated_gib",
            ):
                if key in attempt:
                    record[key] = attempt[key]
        results.append(record)
    return results


def execute_condition(plan, design, condition, seed, output_parent, adapter):
    entry = schedule_entry(plan, condition, seed)
    bundle = create_bundle_directory(output_parent, entry)
    benchmark_path = bundle / "benchmark_manifest.json"
    condition_design = design["conditions"][condition]
    benchmark = {
        "schema_version": 1,
        "experiment_id": plan["experiment_id"],
        "status": "preparing",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "condition": condition,
        "seed": seed,
        "execution_order": entry["order"],
        "planned_execution_order": design["execution_order"],
        "seed_policy": {
            "policy": "single_global_seed_before_full_condition",
            "reseeded_between_scenes": False,
            "note": (
                "The evaluation sidecar is authoritative for the condition-wide seed; "
                "the unchanged Milestone D attempt schema records no per-attempt seed."
            ),
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__)),
        },
        "plan": {
            "path": str(PLAN_PATH.resolve()),
            "sha256": file_sha256(PLAN_PATH),
        },
        "design": public_design(design),
    }
    save_json(benchmark_path, benchmark)

    try:
        if condition == "moderate":
            source_path = bundle / "prepared_moderate_source.txt"
            source_path.write_bytes(design["_prepared_bytes"])
        else:
            source_path = design["_source_path"]

        chapter_started = time.perf_counter()
        planning_started = time.perf_counter()
        run_id = f"{condition}_seed_{seed}"
        run_directory, planning_manifest = create_planning_run(
            source_path,
            plan["chapter_id"],
            run_id,
            bundle / "milestone_d_runs",
        )
        planning_seconds = time.perf_counter() - planning_started
        if len(planning_manifest["scenes"]) != condition_design["scene_count"]:
            raise RuntimeError("Milestone D scene count differs from benchmark design.")

        backend = ObservedSeededBackend(
            adapter,
            seed,
            [scene["scene_id"] for scene in condition_design["scenes"]],
        )
        generation_started = time.perf_counter()
        manifest = generate_planned_run(run_directory, backend)
        generation_wall_seconds = time.perf_counter() - generation_started

        benchmark.update({
            "status": "generation_failed" if manifest["status"] != "generated" else "assembling",
            "milestone_d_run_directory": str(run_directory),
            "timing": {
                "planning_seconds": planning_seconds,
                "backend_initialize_seconds": backend.initialize_seconds,
                "generation_wall_seconds": generation_wall_seconds,
            },
            "rng_resets": backend.rng_resets,
            "frontend": backend.frontend_metadata,
            "backend_provenance": manifest.get("generation", {}).get("backend"),
            "scenes": _scene_results(design, condition, manifest, backend),
        })
        save_json(benchmark_path, benchmark)
        if manifest["status"] != "generated":
            benchmark["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            benchmark["timing"]["total_chapter_wall_seconds"] = (
                time.perf_counter() - chapter_started
            )
            save_json(benchmark_path, benchmark)
            return bundle, benchmark, 1

        assembly_started = time.perf_counter()
        manifest = assemble_chapter(run_directory)
        assembly_seconds = time.perf_counter() - assembly_started
        total_chapter_wall_seconds = time.perf_counter() - chapter_started
        scenes = _scene_results(design, condition, manifest, backend)
        inference_seconds = sum(
            item.get("inference_seconds", 0.0) for item in scenes
        )
        synthesis_wall_seconds = sum(
            item.get("scene_wall_seconds", 0.0) for item in scenes
        )
        final_audio = manifest["assembly"]["audio"]
        source_chars = design["source"]["character_count"]
        benchmark.update({
            "status": "completed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "scenes": scenes,
            "timing": {
                **benchmark["timing"],
                "scene_synthesis_wall_seconds": synthesis_wall_seconds,
                "total_inference_seconds": inference_seconds,
                "assembly_seconds": assembly_seconds,
                "total_chapter_wall_seconds": total_chapter_wall_seconds,
            },
            "summary": {
                "scene_count": len(scenes),
                "normalized_unit_count": sum(
                    scene["normalized_unit_count"] for scene in scenes
                ),
                "source_character_count": source_chars,
                "final_audio_duration_seconds": final_audio["duration_seconds"],
                "aggregate_rtf": inference_seconds / final_audio["duration_seconds"],
                "chars_per_second": source_chars / generation_wall_seconds,
                "chars_per_inference_second": source_chars / inference_seconds,
                "final_wav_sha256": manifest["assembly"]["wav_sha256"],
                "final_output_path": str(
                    run_directory / manifest["assembly"]["output_path"]
                ),
            },
            "assembly": {
                "policy": "Milestone D exact PCM concatenation",
                "extra_silence_ms_between_scenes": manifest["assembly"][
                    "extra_silence_ms_between_scenes"
                ],
                "wav_sha256": manifest["assembly"]["wav_sha256"],
                "audio": final_audio,
                "scenes": manifest["assembly"]["scenes"],
            },
        })
        save_json(benchmark_path, benchmark)
        return bundle, benchmark, 0
    except Exception as error:
        benchmark.update({
            "status": "failed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        })
        save_json(benchmark_path, benchmark)
        raise


def build_adapter(args, plan):
    cosyvoice_root = args.cosyvoice_root.expanduser().resolve()
    return CosyVoiceAdapter(
        cosyvoice_root=cosyvoice_root,
        model_dir=(
            args.model_dir
            or cosyvoice_root / "pretrained_models/Fun-CosyVoice3-0.5B"
        ),
        prompt_wav=args.prompt_wav or Path(plan["prompt_wav"]),
        prompt_text_file=(
            args.prompt_text_file or Path(plan["prompt_transcript_file"])
        ),
    )


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--condition", choices=("giant", "moderate"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--cosyvoice-root", type=Path, default=DEFAULT_COSYVOICE_ROOT)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--prompt-wav", type=Path)
    parser.add_argument("--prompt-text-file", type=Path)
    args = parser.parse_args(argv)

    plan = load_plan()
    design = prepare_design(plan)
    show_plan(design)
    if not args.execute:
        return 0
    if args.condition is None or args.seed is None:
        parser.error("--execute requires --condition and --seed")
    try:
        schedule_entry(plan, args.condition, args.seed)
    except ValueError as error:
        parser.error(str(error))

    adapter = build_adapter(args, plan)
    bundle, benchmark, return_code = execute_condition(
        plan, design, args.condition, args.seed, args.output_parent, adapter
    )
    print("Benchmark output:", bundle)
    print("Benchmark manifest:", bundle / "benchmark_manifest.json")
    if benchmark.get("summary"):
        print("Final chapter:", benchmark["summary"]["final_output_path"])
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
