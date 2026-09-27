"""Model-free checks for schema-5 unit execution and recovery."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave

from src.audiobook.cosyvoice import CosyVoiceAdapter, file_sha256, wav_info
from src.audiobook.__main__ import main as cli_main
from src.audiobook.manifest import create_planning_run
from src.audiobook.pipeline import GenerationError, save_manifest
from src.audiobook.unit_execution import (
    assemble_units, derive_unit_seed, generate_units, select_unit_attempt,
)
from src.audiobook.unit_planning import prepare_synthesis_unit_run
from src.audiobook.content_qc import MODEL_ID, MODEL_REVISION
from tests.test_audiobook_unit_planning import FixtureFrontend
from src.audiobook.unit_planning import canonical_sha256


def write_wav(path, value=1, frames=4):
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        wav.writeframes(bytes((value, 0)) * frames)


class FakeUnitBackend:
    def __init__(self, frontend_hash, fail_text=None):
        self.frontend_hash = frontend_hash
        self.fail_text = fail_text
        self.calls = []
        self.initialize_calls = 0

    def configuration(self):
        return {"backend": "fake_unit_backend", "settings": {"stream": False}}

    def initialize_units(self):
        self.initialize_calls += 1
        return {
            **self.configuration(), "sample_rate_hz": 24000,
            "frontend_identity_sha256": self.frontend_hash,
        }

    def generate_unit(self, text, output_path, seed):
        self.calls.append((text, Path(output_path), seed))
        if text == self.fail_text:
            raise RuntimeError("fake interrupted synthesis")
        write_wav(output_path, len(self.calls))
        return {"frontend_bypass": True, "cosyvoice_chunks": 1}

    def generate_scene(self, *_args, **_kwargs):
        raise AssertionError("Historical scene generator must not run")


class FakeASRWorker:
    instances = []

    def __init__(self, _asr_python, _log_path):
        self.requests = []
        self.model = {"resolved_revision": MODEL_REVISION, "model_id": MODEL_ID}
        self.__class__.instances.append(self)

    def recognize(self, request):
        assert set(request) == {"type", "request_id", "audio_path", "wav_sha256"}
        self.requests.append(request)
        token = {"comparison_token": "甲", "start_seconds": 0.0,
                 "end_seconds": 0.1}
        return {"type": "recognized", "request_id": request["request_id"],
                "wav_sha256": request["wav_sha256"],
                "raw_transcript": "甲", "comparison_tokens": [token],
                "comparison_text": "甲", "raw_emitted_tokens": [],
                "ignored_tokens": []}

    def close(self):
        pass


class UnitExecutionTests(unittest.TestCase):
    def setUp(self):
        FakeASRWorker.instances.clear()
        worker_patch = patch("src.audiobook.unit_execution.ASRWorkerClient", FakeASRWorker)
        worker_patch.start()
        self.addCleanup(worker_patch.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = self.root / "chapter.txt"
        source.write_text("甲。乙。\n***\n丙。", encoding="utf-8")
        self.run_dir, _ = create_planning_run(
            source, "chapter", "run", self.root / "outputs"
        )
        self.prepared = prepare_synthesis_unit_run(self.run_dir, FixtureFrontend())
        self.frontend_hash = self.prepared["synthesis_unit_plan"]["frontend_identity_sha256"]

    def backend(self, fail_text=None):
        return FakeUnitBackend(self.frontend_hash, fail_text)

    def manifest(self):
        return json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8"))

    def units(self, manifest):
        return [unit for scene in manifest["scenes"] for unit in scene["synthesis_units"]]

    def test_initial_generation_uses_frozen_text_and_stable_explicit_seeds(self):
        backend = self.backend()
        result = generate_units(self.run_dir, backend, root_seed=17)
        units = self.units(result)
        self.assertEqual(backend.initialize_calls, 1)
        self.assertEqual(len(backend.calls), len(units))
        self.assertEqual([call[0] for call in backend.calls],
                         [unit["normalized_text"] for unit in units])
        self.assertEqual(result["status"], "generated")
        self.assertEqual(result["generation"]["root_seed"], 17)
        plan_hash = result["synthesis_unit_plan"]["ordered_unit_plan_sha256"]
        for unit, call in zip(units, backend.calls):
            state = unit["generation"]
            attempt = state["attempts"][0]
            self.assertEqual(state["selected_attempt_id"], "attempt_001")
            self.assertEqual(len(state["selection_history"]), 1)
            self.assertEqual(attempt["take_index"], 1)
            self.assertEqual(call[2], derive_unit_seed(17, plan_hash, unit["id"], 1))
            self.assertEqual(attempt["seed"], call[2])
            self.assertEqual(attempt["normalized_text_sha256"],
                             unit["normalized_text_sha256"])
            self.assertTrue((self.run_dir / attempt["output_path"]).is_file())
        self.assertTrue(all("attempts" not in scene["generation"]
                            for scene in result["scenes"]))
        self.assertTrue(all(scene["generation"]["status"] == "generated"
                            for scene in result["scenes"]))

    def test_resume_skips_selected_units_and_does_not_initialize_backend(self):
        first = generate_units(self.run_dir, self.backend(), root_seed=29)
        previous = [(unit["generation"]["attempts"][0]["seed"],
                     unit["generation"]["attempts"][0]["wav_sha256"])
                    for unit in self.units(first)]
        backend = self.backend()
        second = generate_units(self.run_dir, backend)
        self.assertEqual(backend.calls, [])
        self.assertEqual(backend.initialize_calls, 0)
        self.assertEqual(previous, [
            (unit["generation"]["attempts"][0]["seed"],
             unit["generation"]["attempts"][0]["wav_sha256"])
            for unit in self.units(second)
        ])

    def test_failed_unit_resume_only_retries_that_unit_with_same_take_seed(self):
        failed_text = self.units(self.prepared)[1]["normalized_text"]
        first = generate_units(self.run_dir, self.backend(failed_text), root_seed=51)
        self.assertEqual(first["status"], "generation_failed")
        self.assertEqual(first["scenes"][0]["generation"]["status"], "incomplete")
        old = self.units(first)[1]["generation"]["attempts"][0]
        backend = self.backend()
        resumed = generate_units(self.run_dir, backend)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0][0], failed_text)
        attempts = self.units(resumed)[1]["generation"]["attempts"]
        self.assertEqual([item["status"] for item in attempts],
                         ["failed", "generated"])
        self.assertEqual(attempts[1]["take_index"], 1)
        self.assertEqual(attempts[1]["seed"], old["seed"])
        self.assertEqual(resumed["status"], "generated")

    def test_interrupted_valid_wav_and_unpublished_selection_recover_without_synthesis(self):
        first = generate_units(self.run_dir, self.backend(), root_seed=77)
        units = self.units(first)
        units[0]["generation"]["selected_attempt_id"] = None
        units[0]["generation"]["attempts"][0]["status"] = "running"
        units[1]["generation"]["selected_attempt_id"] = None
        self.run_dir.joinpath("manifest.json").write_text(
            json.dumps(first, ensure_ascii=False), encoding="utf-8"
        )
        backend = self.backend()
        result = generate_units(self.run_dir, backend)
        self.assertEqual(backend.calls, [])
        self.assertEqual(backend.initialize_calls, 0)
        self.assertEqual(result["status"], "generated")
        self.assertEqual(self.units(result)[0]["generation"]["attempts"][0]["recovery"],
                         "validated_existing_wav")
        self.assertEqual(len(self.units(result)[0]["generation"]["attempts"]), 1)

    def test_corrupt_selected_wav_fails_closed(self):
        result = generate_units(self.run_dir, self.backend(), root_seed=9)
        artifact = self.run_dir / self.units(result)[0]["generation"]["attempts"][0]["output_path"]
        artifact.write_bytes(b"corrupt")
        before = (self.run_dir / "manifest.json").read_bytes()
        backend = self.backend()
        with self.assertRaisesRegex(GenerationError, "Selected unit WAV is invalid"):
            generate_units(self.run_dir, backend)
        with self.assertRaisesRegex(GenerationError, "Selected unit WAV is invalid"):
            assemble_units(self.run_dir)
        self.assertEqual(backend.calls, [])
        self.assertEqual((self.run_dir / "manifest.json").read_bytes(), before)

    def test_missing_selected_wav_fails_closed(self):
        result = generate_units(self.run_dir, self.backend(), root_seed=10)
        artifact = self.run_dir / self.units(result)[-1]["generation"]["attempts"][0]["output_path"]
        artifact.unlink()
        with self.assertRaisesRegex(GenerationError, "Selected unit WAV is invalid"):
            generate_units(self.run_dir, self.backend())
        with self.assertRaisesRegex(GenerationError, "Selected unit WAV is invalid"):
            assemble_units(self.run_dir)

    def test_interrupted_bad_wav_is_preserved_and_retried_with_same_seed(self):
        failed_text = self.units(self.prepared)[1]["normalized_text"]
        first = generate_units(self.run_dir, self.backend(failed_text), root_seed=81)
        unit = self.units(first)[1]
        old = unit["generation"]["attempts"][0]
        old["status"] = "running"
        old_path = self.run_dir / old["output_path"]
        old_path.write_bytes(b"incomplete wav")
        (self.run_dir / "manifest.json").write_text(
            json.dumps(first, ensure_ascii=False), encoding="utf-8"
        )
        resumed = generate_units(self.run_dir, self.backend())
        attempts = self.units(resumed)[1]["generation"]["attempts"]
        self.assertEqual(old_path.read_bytes(), b"incomplete wav")
        self.assertEqual([item["status"] for item in attempts], ["failed", "generated"])
        self.assertEqual(attempts[0]["seed"], attempts[1]["seed"])
        self.assertEqual(attempts[0]["recovery"], "interrupted_attempt_preserved")

    def test_cli_routes_schema_five_without_historical_scene_generation(self):
        backend = self.backend()
        with (patch("src.audiobook.__main__.create_adapter", return_value=backend),
              patch("src.audiobook.__main__.generate_planned_run") as historical,
              patch("src.audiobook.__main__.assemble_chapter") as old_assembly):
            self.assertEqual(cli_main(["generate", str(self.run_dir),
                                       "--root-seed", "21"]), 0)
            self.assertEqual(cli_main(["assemble", str(self.run_dir)]), 0)
            historical.assert_not_called()
            old_assembly.assert_not_called()
        self.assertEqual(self.manifest()["assembly"]["status"], "assembled")

    def test_default_run_cli_uses_certified_unit_workflow(self):
        source = self.root / "normal_run.txt"
        source.write_text("甲。乙。", encoding="utf-8")
        frontend = FixtureFrontend()
        backend = FakeUnitBackend(canonical_sha256(frontend.initialize()["identity"]))
        with (patch("src.audiobook.__main__.unit_model_dir", return_value=self.root),
              patch("src.audiobook.__main__.CosyVoiceFrontendAdapter", return_value=frontend),
              patch("src.audiobook.__main__.create_adapter", return_value=backend)):
            result = cli_main([
                "run", str(source), "--chapter-id", "normal_run",
                "--run-id", "run", "--output-root", str(self.root / "outputs"),
            ])
        run = self.root / "outputs/normal_run/run"
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(result, 0)
        self.assertEqual(manifest["schema_version"], 5)
        self.assertEqual(manifest["assembly"]["status"], "assembled")
        self.assertEqual(len(backend.calls), 2)

    def test_title_punctuation_is_sent_only_to_recorded_first_unit(self):
        class HeadingFrontend(FixtureFrontend):
            def normalize_heading(self, heading):
                return "第3章"

        source = self.root / "titled.txt"
        source.write_text("第3章\n甲。乙。", encoding="utf-8")
        run, _ = create_planning_run(source, "titled", "run", self.root / "outputs")
        frontend = HeadingFrontend()
        prepared = prepare_synthesis_unit_run(run, frontend)
        backend = FakeUnitBackend(prepared["synthesis_unit_plan"]["frontend_identity_sha256"])
        generated = generate_units(run, backend, root_seed=31)
        self.assertEqual(generated["status"], "generated")
        self.assertEqual([call[0] for call in backend.calls], ["第3章。甲。", "乙。"])
        first = generated["scenes"][0]["synthesis_units"][0]
        self.assertEqual(first["normalized_text"], "第3章甲。")
        self.assertEqual(first["generation"]["attempts"][0]["synthesis_text_override_sha256"],
                         prepared["title_synthesis_override"]["synthesis_text_sha256"])
        assemble_units(run)
        resumed = generate_units(run, backend)
        self.assertEqual(resumed["status"], "generated")
        self.assertEqual(len(backend.calls), 2)
        manifest_path = run / "manifest.json"
        tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
        tampered["scenes"][0]["synthesis_units"][0]["generation"]["attempts"][0][
            "synthesis_text_override_sha256"
        ] = "0" * 64
        manifest_path.write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")
        with self.assertRaisesRegex(GenerationError, "synthesis text override differs"):
            generate_units(run, backend)

    def test_assembly_is_exact_ordered_pcm_with_zero_extra_frames(self):
        result = generate_units(self.run_dir, self.backend(), root_seed=37)
        assembly = assemble_units(self.run_dir)["assembly"]
        self.assertEqual(assembly["extra_silence_ms_between_units"], 0)
        self.assertEqual(len(assembly["units"]), len(self.units(result)))
        self.assertEqual([clip["unit_id"] for clip in assembly["units"]],
                         [unit["id"] for unit in self.units(result)])
        payloads = []
        for clip in assembly["units"]:
            with wave.open(str(self.run_dir / clip["artifact_path"]), "rb") as wav:
                payloads.append(wav.readframes(wav.getnframes()))
        with wave.open(str(self.run_dir / assembly["output_path"]), "rb") as wav:
            self.assertEqual(wav.readframes(wav.getnframes()), b"".join(payloads))
            self.assertEqual(wav.getnframes(), sum(clip["frame_count"] for clip in assembly["units"]))
        self.assertEqual(assembly["wav_sha256"],
                         file_sha256(self.run_dir / assembly["output_path"]))

    def test_valid_selection_change_marks_assembly_stale(self):
        result = generate_units(self.run_dir, self.backend(), root_seed=11)
        assemble_units(self.run_dir)
        unit = self.units(result)[0]
        scene_id = result["scenes"][0]["id"]
        attempt_id = "attempt_002"
        relative = f"units/{scene_id}/{unit['id']}/{attempt_id}/generated.wav"
        output = self.run_dir / relative
        output.parent.mkdir(parents=True)
        write_wav(output, 99)
        attempt = {
            "id": attempt_id, "take_index": 2,
            "seed": derive_unit_seed(11, result["synthesis_unit_plan"]["ordered_unit_plan_sha256"],
                                     unit["id"], 2),
            "status": "generated", "output_path": relative,
            "normalized_text_sha256": unit["normalized_text_sha256"],
            "frontend_bypass": True,
            "audio": wav_info(output, 24000), "wav_sha256": file_sha256(output),
        }
        current = self.manifest()
        # A schema-5 run created before content QC remains readable under its
        # historical unit selection behavior.
        current.pop("unit_execution_policy")
        current["generation"].pop("content_qc")
        current["generation"].pop("content_retry")
        for old_unit in self.units(current):
            old_unit["generation"].pop("retry_state")
            for old_attempt in old_unit["generation"]["attempts"]:
                old_attempt.pop("content_qc")
        self.units(current)[0]["generation"]["attempts"].append(attempt)
        (self.run_dir / "manifest.json").write_text(
            json.dumps(current, ensure_ascii=False), encoding="utf-8"
        )
        changed = select_unit_attempt(self.run_dir, unit["id"], attempt_id)
        self.assertEqual(changed["assembly"]["status"], "stale")
        self.assertEqual(self.units(changed)[0]["generation"]["selected_attempt_id"],
                         attempt_id)
        self.assertEqual(len(self.units(changed)[0]["generation"]["selection_history"]), 2)

    def test_frontend_identity_mismatch_fails_before_unit_synthesis(self):
        backend = FakeUnitBackend("0" * 64)
        result = generate_units(self.run_dir, backend, root_seed=5)
        self.assertEqual(result["status"], "generation_failed")
        self.assertEqual(backend.calls, [])
        self.assertIn("differs", result["generation"]["initialization_failure"]["message"])


class AdapterUnitBypassTests(unittest.TestCase):
    def test_unit_method_passes_frozen_text_once_with_frontend_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = CosyVoiceAdapter(root, root, root / "prompt.wav", root / "prompt.txt")
            model = Mock(sample_rate=24000)
            model.inference_zero_shot.return_value = [{"tts_speech": Mock()}]
            torch = Mock()
            torch.cuda.is_available.return_value = False
            torch.cuda.max_memory_allocated.return_value = 0
            adapter._model = model
            adapter._torch = torch
            adapter._torchaudio = Mock()
            adapter._prompt_text = "prepared prompt"
            adapter._unit_frontend_identity = {"test": True}
            output = root / "generated.wav"
            with (patch("src.audiobook.cosyvoice.set_cosyvoice_random_seed"),
                  patch("src.audiobook.cosyvoice.write_pcm16_wav",
                        side_effect=lambda _ta, path, _speech, _rate: write_wav(path))):
                adapter.generate_unit("甲。", output, 123)
            model.inference_zero_shot.assert_called_once_with(
                "甲。", "prepared prompt", str(adapter.prompt_wav),
                stream=False, text_frontend=False,
            )

    def test_explicit_native_breath_token_passes_through_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = CosyVoiceAdapter(root, root, root / "prompt.wav", root / "prompt.txt")
            model = Mock(sample_rate=24000)
            model.inference_zero_shot.return_value = [{"tts_speech": Mock()}]
            torch = Mock()
            torch.cuda.is_available.return_value = False
            torch.cuda.max_memory_allocated.return_value = 0
            adapter._model, adapter._torch = model, torch
            adapter._torchaudio = Mock()
            adapter._prompt_text = "prepared prompt"
            adapter._unit_frontend_identity = {"test": True}
            with (patch("src.audiobook.cosyvoice.set_cosyvoice_random_seed"),
                  patch("src.audiobook.cosyvoice.write_pcm16_wav",
                        side_effect=lambda _ta, path, _speech, _rate: write_wav(path))):
                adapter.generate_unit("甲。[breath]乙。", root / "breath.wav", 123)
            self.assertEqual(model.inference_zero_shot.call_args.args[0],
                             "甲。[breath]乙。")
            self.assertFalse(model.inference_zero_shot.call_args.kwargs["text_frontend"])

class AtomicManifestTests(unittest.TestCase):
    def test_one_transient_destination_lock_preserves_atomic_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text('{"old": true}', encoding="utf-8")
            original_replace = Path.replace
            count = 0

            def replace(source, destination):
                nonlocal count
                count += 1
                if count == 1:
                    raise PermissionError("transient OneDrive lock")
                return original_replace(source, destination)

            with (patch("src.audiobook.pipeline.Path.replace", replace),
                  patch("src.audiobook.pipeline.time.sleep")):
                save_manifest(path, {"complete": True})
            self.assertEqual(count, 2)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")),
                             {"complete": True})
            self.assertEqual(list(Path(temporary).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
