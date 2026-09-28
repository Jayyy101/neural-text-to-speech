"""Audit historical CRs and synthesize only the approved clean 12-unit excerpt."""

import argparse
from collections import Counter
import contextlib
import csv
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.run_cosyvoice_cr_format_ab import GIANT, SWEEP, SEED, pcm, require
from evaluation.run_cosyvoice_rng_sweep import (
    COSYVOICE_ROOT, MODEL_DIR, configure_local_wetext_frontend,
    save_manifest, text_sha256, validate_speech,
)
from src.audiobook.cosyvoice import (
    PROMPT_PREFIX, SETTLED_SETTINGS, create_cosyvoice_model, file_sha256,
    git_head, load_cosyvoice_runtime, normalize_prompt_transcript,
    set_cosyvoice_random_seed, wav_info, write_pcm16_wav,
)

SOURCE = ROOT / "test-shentongzhe/神通者01.txt"
MODERATE = GIANT.parent.parent / "02_moderate_seed_2026091702/benchmark_manifest.json"
SOURCE_HASH = "146051b562c7dc5c350a4ba8f7c532f7bf347b71437b664f6a45e8ba1c0b4ee3"
BOUNDARIES = (0, 92, 191, 268, 356, 441, 527, 604, 675, 805, 884, 1007, 1083)
SIZES = (78, 91, 73, 76, 71, 78, 69, 63, 122, 71, 115, 76)


def invisible_counts(text):
    counts = Counter(c for c in text if unicodedata.category(c) in ("Cc", "Cf", "Zl", "Zp")
                     or (c.isspace() and c != " "))
    return {f"U+{ord(c):04X}": count for c, count in sorted(counts.items())}


def audit_cr(raw, manifests):
    audit = {"source_sha256": file_sha256(SOURCE), "source_characters": len(raw),
             "source_crlf_count": raw.count("\r\n"), "source_invisibles": invisible_counts(raw),
             "runs": {}}
    for name, (path, manifest) in manifests.items():
        units = [u["text"] for s in manifest["scenes"] for u in s["normalized_units"]]
        result = {"manifest": str(path), "manifest_sha256": file_sha256(path),
                  "normalized_units": len(units), "units_with_cr": sum("\r" in u for u in units),
                  "cr_count": sum(u.count("\r") for u in units),
                  "invisibles": invisible_counts("".join(units)), "scenes": []}
        mapped = 0
        for scene in manifest["scenes"]:
            span = scene["original_source_span"]
            source = raw[span["start_character"]:span["end_character"]]
            require(text_sha256(source) == scene["text_sha256"], "Source scene hash differs")
            text = "".join(u["text"] for u in scene["normalized_units"])
            for unit in scene["normalized_units"]:
                require(text_sha256(unit["text"]) == unit["text_sha256"], "Observation text hash differs")
            source_positions = [span["start_character"] + i for i, c in enumerate(source) if c != "\n"]
            no_lf = source.replace("\n", "")
            correspondences, changes = [], []
            for tag, a, b, c, d in difflib.SequenceMatcher(None, no_lf, text, autojunk=False).get_opcodes():
                if tag == "equal":
                    for offset, char in enumerate(text[c:d]):
                        if char == "\r":
                            source_pos = source_positions[a + offset]
                            require(raw[source_pos:source_pos + 2] == "\r\n", "CR does not originate in CRLF")
                            correspondences.append({"normalized_offset": c + offset, "source_offset": source_pos})
                elif "\r" in no_lf[a:b] or "\r" in text[c:d]:
                    changes.append({"operation": tag, "source_fragment": no_lf[a:b],
                                    "normalized_fragment": text[c:d], "at_scene_end": b == len(no_lf)})
            require(len(correspondences) == text.count("\r"), "Unmapped normalized CR")
            mapped += len(correspondences)
            result["scenes"].append({"scene_id": scene["scene_id"], "cr_source_mapping": correspondences,
                                     "cr_changes": changes})
        result["all_crs_match_original_crlf_positions"] = mapped == result["cr_count"]
        audit["runs"][name] = result
    audit["total_observed_crs_across_runs"] = sum(r["cr_count"] for r in audit["runs"].values())
    return audit


def prepare():
    require(file_sha256(SOURCE) == SOURCE_HASH, "Chapter source changed; STOP before synthesis")
    raw = SOURCE.read_bytes().decode("utf-8-sig")
    sweep = json.loads(SWEEP.read_text(encoding="utf-8"))
    giant = json.loads(GIANT.read_text(encoding="utf-8"))
    moderate = json.loads(MODERATE.read_text(encoding="utf-8"))
    require(sweep["status"] == giant["status"] == moderate["status"] == "completed", "Incomplete history")
    require(giant["seed"] == moderate["seed"] == SEED, "Incorrect benchmark seeds")
    audit = audit_cr(raw, {"giant_1702": (GIANT, giant), "moderate_1702": (MODERATE, moderate)})
    texts = [u["text"].replace("\r", "") for u in giant["scenes"][0]["normalized_units"][:12]]
    require(len(texts) == 12 and tuple(map(len, texts)) == SIZES, "12-unit count/sizes changed")
    require(not any(invisible_counts(t) for t in texts), "Control/invisible characters in clean units")
    history = sorted((r for r in sweep["records"] if r["global_seed"] == SEED), key=lambda r: r["unit_index"])
    require(len(history) == 4, "Expected four historical successful units")
    for i in range(4):
        require(texts[i] == sweep["units"][i]["normalized_text"], "Historical clean string mismatch")
        require(text_sha256(texts[i]) == sweep["units"][i]["normalized_text_sha256"]
                == history[i]["normalized_text_sha256"], "Historical clean hash mismatch")
        require(file_sha256(SWEEP.parent / history[i]["output_path"]) == history[i]["wav_sha256"],
                "Historical unit WAV changed")
    excerpt = raw[:BOUNDARIES[-1]]
    pieces = [raw[a:b] for a, b in zip(BOUNDARIES, BOUNDARIES[1:])]
    require("".join(pieces) == excerpt and len(excerpt) == 1083, "Source excerpt coverage mismatch")
    require(excerpt.endswith("也有着殒命的风险。"), "Excerpt end changed")
    # Explicitly verify the only non-whitespace normalization changes in this excerpt.
    compact_source = excerpt.replace("\r", "").replace("\n", "").replace(" ", "")
    expected = compact_source.replace("第1章", "第一章", 1).replace("?!", "?")
    require(expected == "".join(texts), "Unexpected linguistic text/order change; STOP before synthesis")
    plan = {"source": str(SOURCE), "source_sha256": SOURCE_HASH,
            "source_excerpt": excerpt, "source_excerpt_sha256": text_sha256(excerpt),
            "source_character_count": 1083, "normalized_character_count": sum(SIZES),
            "normalization_changes": "Remove source whitespace; 第1章 -> 第一章; two ?! -> ? sequences, matching saved frontend observations",
            "seed": SEED, "unit_count": 12, "combined_input_sha256": text_sha256("".join(texts)),
            "checks": {"exactly_12_units": True, "zero_cr_lf_or_other_controls": True,
                       "first_four_strings_and_hashes_match_sweep": True,
                       "source_spans_reconstruct_exact_excerpt": True,
                       "normalized_text_preserves_order_with_only_documented_changes": True},
            "units": [{"unit_index": i + 1, "source_start": BOUNDARIES[i], "source_end": BOUNDARIES[i + 1],
                       "source_text": pieces[i], "source_sha256": text_sha256(pieces[i]),
                       "normalized_text": text, "text_sha256": text_sha256(text), "characters": len(text)}
                      for i, text in enumerate(texts)]}
    return plan, audit, sweep, history


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.stream.write(value)
        self.log.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def execute(prepared, run):
    plan, audit, sweep, history = prepared
    texts = [u["normalized_text"] for u in plan["units"]]
    combined = "".join(texts)
    started = time.perf_counter()
    manifest = {"status": "initializing", "experiment_id": "cosyvoice_clean12", "seed": SEED,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "runner_sha256": file_sha256(Path(__file__)), "settings": dict(SETTLED_SETTINGS),
                "plan_sha256": file_sha256(run / "plan.json"), "audit_sha256": file_sha256(run / "cr_audit.json"),
                "historical_sweep": str(SWEEP), "historical_sweep_sha256": file_sha256(SWEEP),
                "seed_policy": "Once after initialization before unit 01; RNG advances sequentially; no per-unit reseeding",
                "model_lifetime": "One initialization, resident for all 12 units; no additional warmup synthesis",
                "records": [], "rng_resets": [], "actual_synthesis_texts": []}
    path = run / "manifest.json"
    save_manifest(path, manifest)
    try:
        prompt = sweep["prompt"]
        wav = Path(prompt["wav"])
        transcript = normalize_prompt_transcript(Path(prompt["transcript_file"]).read_text(encoding="utf-8-sig"))
        require(transcript == prompt["transcript"] and file_sha256(wav) == prompt["wav_sha256"], "Narrator changed")
        prompt_text = PROMPT_PREFIX + transcript
        require("\r" not in prompt_text and "\r" not in combined, "CR in model inputs")
        require(dict(SETTLED_SETTINGS) == sweep["settings"], "Model settings changed")
        manifest["prompt"] = prompt
        manifest["cosyvoice_git_head"] = git_head(COSYVOICE_ROOT)
        manifest["model_config_sha256"] = file_sha256(MODEL_DIR / "cosyvoice3.yaml")
        for key in ("cosyvoice_git_head", "model_config_sha256"):
            require(manifest[key] == sweep[key], f"Historical model provenance differs: {key}")
        imports_started = time.perf_counter()
        torch, torchaudio, auto_model = load_cosyvoice_runtime(COSYVOICE_ROOT)
        manifest["import_seconds"] = time.perf_counter() - imports_started
        require(torch.cuda.is_available(), "CUDA unavailable")
        manifest["runtime"] = {"python": platform.python_version(), "executable": sys.executable,
                               "torch": torch.__version__, "torchaudio": torchaudio.__version__,
                               "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
        load_started = time.perf_counter()
        model = create_cosyvoice_model(auto_model, MODEL_DIR, SETTLED_SETTINGS)
        manifest["model_load_seconds"] = time.perf_counter() - load_started
        manifest["frontend"] = configure_local_wetext_frontend(model.frontend)
        require(manifest["frontend"]["assets"] == sweep["frontend"]["assets"], "Frontend assets changed")
        # Both checks occur before opening any synthesis generator; no raw CR is passed to the frontend.
        from_source = model.frontend.text_normalize(plan["source_excerpt"].replace("\r", ""), split=True)
        from_frozen = model.frontend.text_normalize(combined, split=True)
        manifest["frontend_preflight"] = {"from_clean_source": from_source, "from_frozen_units": from_frozen}
        require(from_source == from_frozen == texts, "Frontend text/boundary check FAILED; STOP before synthesis")
        manifest["all_pre_synthesis_checks_passed"] = True
        save_manifest(path, manifest)
        print("All pre-synthesis checks passed: 12 units, zero CR, exact first four, exact excerpt coverage.", flush=True)
        original = model.frontend.frontend_zero_shot

        def observed(*args, **kwargs):
            text = args[0] if args else kwargs["tts_text"]
            index = len(manifest["actual_synthesis_texts"])
            require(index < 12 and text == texts[index] and "\r" not in text, "Actual synthesis input mismatch")
            actual_prompt = args[1] if len(args) > 1 else kwargs["prompt_text"]
            require("\r" not in actual_prompt, "CR in actual prompt")
            manifest["actual_synthesis_texts"].append(text)
            return original(*args, **kwargs)

        model.frontend.frontend_zero_shot = observed
        torch.cuda.synchronize()
        set_cosyvoice_random_seed(SEED)
        manifest["rng_resets"].append({"seed": SEED, "before_unit": 1})
        generator = iter(model.inference_zero_shot(combined, prompt_text, str(wav), stream=False, text_frontend=True))
        sequence_started = previous = time.perf_counter()
        speeches, timings = [], []
        manifest["status"] = "generating"
        for i in range(12):
            output = next(generator)
            now = time.perf_counter()
            speeches.append(validate_speech(output))
            timings.append(now - previous)
            previous = now
            print(f"Unit {i + 1:02d}/12: {timings[-1]:.3f}s", flush=True)
        try:
            next(generator)
        except StopIteration:
            pass
        else:
            raise RuntimeError("Unexpected thirteenth synthesis yield")
        torch.cuda.synchronize()
        manifest["generation_seconds"] = time.perf_counter() - sequence_started
        model.frontend.frontend_zero_shot = original
        require(manifest["actual_synthesis_texts"] == texts, "Incomplete synthesis observation")
        for i, (speech, seconds, unit) in enumerate(zip(speeches, timings, plan["units"]), 1):
            output_path = run / f"unit_{i:02d}.wav"
            write_pcm16_wav(torchaudio, output_path, speech, model.sample_rate)
            record = {**unit, "output_path": output_path.name, "generation_seconds": seconds,
                      "audio": wav_info(output_path, model.sample_rate), "wav_sha256": file_sha256(output_path)}
            if i <= 4:
                record["historical_wav_sha256"] = history[i - 1]["wav_sha256"]
                record["historical_wav_byte_identical"] = record["wav_sha256"] == history[i - 1]["wav_sha256"]
            manifest["records"].append(record)
        output_path = run / "concatenated.wav"
        write_pcm16_wav(torchaudio, output_path, torch.cat(speeches, dim=1), model.sample_rate)
        audio = wav_info(output_path, model.sample_rate)
        require(audio["frames"] == sum(r["audio"]["frames"] for r in manifest["records"]), "Assembly frame mismatch")
        require(pcm(output_path) == b"".join(pcm(run / r["output_path"]) for r in manifest["records"]), "Assembly PCM mismatch")
        manifest["concatenation"] = {"output_path": output_path.name, "audio": audio,
                                     "wav_sha256": file_sha256(output_path), "added_silence_ms": 0,
                                     "sample_exact_ordered_concatenation": True}
        manifest["actual_counts"] = {"model_initializations": 1, "public_calls": 1,
                                     "synthesis_jobs": len(speeches), "rng_resets": len(manifest["rng_resets"])}
        require(file_sha256(SOURCE) == SOURCE_HASH, "Source changed during execution")
        manifest.update(status="completed", total_wall_seconds=time.perf_counter() - started,
                        finished_at_utc=datetime.now(timezone.utc).isoformat())
        save_manifest(path, manifest)
        with (run / "per_unit.csv").open("x", encoding="utf-8-sig", newline="") as handle:
            fields = ["unit", "characters", "duration_seconds", "generation_seconds", "text_sha256", "wav_sha256", "output_path"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for r in manifest["records"]:
                writer.writerow(dict(unit=r["unit_index"], characters=r["characters"], duration_seconds=r["audio"]["duration_seconds"],
                                     generation_seconds=r["generation_seconds"], text_sha256=r["text_sha256"],
                                     wav_sha256=r["wav_sha256"], output_path=str(run / r["output_path"])))
        print("Completed:", run, flush=True)
    except Exception as error:
        manifest.update(status="failed", error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        save_manifest(path, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    prepared = prepare()
    plan, audit, _, _ = prepared
    print(json.dumps({"audit": {name: {k: v for k, v in r.items() if k != "scenes"} for name, r in audit["runs"].items()},
                      "preflight": plan["checks"], "sizes": SIZES, "source_characters": 1083,
                      "normalized_characters": 983}, ensure_ascii=True, indent=2))
    if not args.execute:
        return 0
    run = ROOT / "outputs/evaluation" / ("cosyvoice_clean12_" + datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f"))
    run.mkdir()
    save_manifest(run / "plan.json", plan)
    save_manifest(run / "cr_audit.json", audit)
    (run / "source_excerpt.txt").write_bytes(plan["source_excerpt"].encode("utf-8"))
    with (run / "console.log").open("x", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print("Output:", run, flush=True)
            execute(prepared, run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
