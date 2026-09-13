"""Characterization tests: no TTS packages, network, GPU, or model downloads."""

import importlib.util
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from datetime import datetime
import io
import json
import os
from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest.mock import Mock, patch
import warnings
import wave


ROOT = Path(__file__).resolve().parents[1]


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MeloBaselineTests(unittest.TestCase):
    def setUp(self):
        self.model = types.SimpleNamespace(
            hps=types.SimpleNamespace(data=types.SimpleNamespace(spk2id={"ZH": 7, "EN-Default": 0})),
            tts_to_file=Mock(),
        )
        self.factory = Mock(return_value=self.model)
        fake_api = types.ModuleType("melo.api")
        fake_api.TTS = self.factory
        fake_melo = types.ModuleType("melo")
        fake_melo.api = fake_api
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=Mock(return_value=True))
        with patch.dict("sys.modules", {"melo": fake_melo, "melo.api": fake_api, "torch": fake_torch}), patch.dict(os.environ), warnings.catch_warnings():
            self.backend = load_source("baseline_under_test", ROOT / "src/generate_melo.py")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        previous_cwd = Path.cwd()
        self.addCleanup(os.chdir, previous_cwd)
        os.chdir(self.temporary.name)

    def test_language_aliases_and_existing_fallback(self):
        for alias in ("zh", " ZH-CN ", "Chinese", "mandarin"):
            self.assertEqual(self.backend.normalize_language(alias), "ZH")
        self.assertEqual(self.backend.normalize_language("unsupported"), "EN")

    def test_speed_defaults_boundaries_and_invalid_input(self):
        self.assertEqual(self.backend.parse_speed(" "), 1.0)
        for speed in (0.8, "0.9", 1.0, 1.2):
            self.assertEqual(self.backend.parse_speed(speed), float(speed))
        for invalid in (0.7, 1.3, "not a number"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.backend.parse_speed(invalid)

    def test_model_loading_is_lazy(self):
        self.factory.assert_not_called()

    def test_cuda_cache_reused_per_language(self):
        self.backend.load_model("ZH")
        self.backend.load_model("ZH")
        self.factory.assert_called_once_with(language="ZH", device="cuda:0")
        self.backend.load_model("EN")
        self.assertEqual(self.factory.call_count, 2)

    def test_cpu_fallback(self):
        self.backend.torch.cuda.is_available.return_value = False
        self.backend.load_model("ZH")
        self.factory.assert_called_once_with(language="ZH", device="cpu")

    def test_text_settings_and_result_contract(self):
        text = "  你好，GPU ready。\n下一段。  "
        result = self.backend.synthesize_melo(text, "zh-cn", "ZH", "0.9")
        self.model.tts_to_file.assert_called_once_with(text, 7, result["output_path"], speed=0.9)
        self.assertEqual(set(result), {"language", "speaker", "speed", "output_path", "inference_time"})
        self.assertEqual((result["language"], result["speaker"], result["speed"]), ("ZH", "ZH", 0.9))
        self.assertRegex(result["output_path"], r"^outputs/melo_zh_\d{8}_\d{6}\.wav$")
        self.assertTrue(Path("outputs").is_dir())

    def test_blank_text_and_speaker_use_existing_defaults(self):
        result = self.backend.synthesize_melo(" \n", "ZH", " ")
        self.model.tts_to_file.assert_called_once_with(self.backend.DEFAULT_TEXTS["ZH"], 7, result["output_path"], speed=1.0)

    def test_invalid_speaker_does_not_synthesize(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            self.backend.synthesize_melo("你好", "ZH", "missing")
        self.model.tts_to_file.assert_not_called()

    def test_model_error_propagates(self):
        self.model.tts_to_file.side_effect = RuntimeError("synthetic inference failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic inference failure"):
            self.backend.synthesize_melo("你好", "ZH")


class EvaluationTests(unittest.TestCase):
    def test_corpus_has_all_seven_cases_and_baseline_settings(self):
        corpus = json.loads((ROOT / "evaluation/inputs/mandarin_diagnostics.json").read_text(encoding="utf-8"))
        self.assertEqual((corpus["language"], corpus["speaker"], corpus["speed"]), ("ZH", "ZH", 1.0))
        cases = corpus["cases"]
        self.assertEqual({c["id"] for c in cases}, {"narration", "dialogue", "punctuation", "names_vocabulary", "numbers", "mixed_text", "long_passage"})
        self.assertEqual(len(cases), 7)
        for case in cases:
            self.assertTrue(case["text"].strip())
            self.assertTrue(case["listen_for"])
        passage = next(c["text"] for c in cases if c["id"] == "long_passage")
        self.assertGreater(len(passage), 600)
        self.assertIn("\n\n", passage)

    def test_smoke_trials_keep_audio_separate_and_restore_directory(self):
        runner = load_source("smoke_under_test", ROOT / "evaluation/run_melo_smoke.py")

        def fake_synthesize(text, **settings):
            Path("outputs").mkdir()
            with wave.open("outputs/fixed.wav", "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(b"\x00\x00" * 160)
            return {"output_path": "outputs/fixed.wav", "inference_time": 0.01}

        original = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            for number in (1, 2):
                trial = Path(temporary) / str(number)
                output = Path(temporary) / f"narration_{number:02d}.wav"
                result = runner.run_case(fake_synthesize, {"text": "你好"}, {}, trial, output)
                self.assertEqual(result["output_filename"], output.name)
                self.assertTrue(output.is_file())
                self.assertFalse((trial / "outputs/fixed.wav").exists())
                self.assertEqual(result["audio"]["frames"], 160)
                self.assertEqual(Path.cwd(), original)

    def test_smoke_failure_restores_directory(self):
        runner = load_source("smoke_failure_under_test", ROOT / "evaluation/run_melo_smoke.py")
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "failed"):
                runner.run_case(Mock(side_effect=RuntimeError("failed")), {"text": "你好"}, {}, Path(temporary) / "trial", Path(temporary) / "failed_01.wav")
        self.assertEqual(Path.cwd(), original)


class EvaluationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.runner = load_source("workflow_under_test", ROOT / "evaluation/run_melo_smoke.py")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    @staticmethod
    def fake_synthesize(text, **settings):
        if "_" in text:
            raise AssertionError()  # Reproduce an empty-message model failure.
        Path("outputs").mkdir()
        with wave.open("outputs/fixed.wav", "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(b"\x00\x00" * 160)
        return {"output_path": "outputs/fixed.wav", "inference_time": 0.01}

    def run_main(self, args, cuda=True):
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: cuda, get_device_name=lambda _: "test GPU")
        fake_torch.version = types.SimpleNamespace(cuda="test")
        fake_backend = types.ModuleType("generate_melo")
        fake_backend.synthesize_melo = self.fake_synthesize
        out = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.runner, "ROOT", self.root))
            stack.enter_context(patch.object(self.runner, "file_hash", return_value="test-hash"))
            stack.enter_context(patch.object(self.runner.metadata, "version", return_value="test-version"))
            stack.enter_context(patch.object(self.runner.metadata, "distribution", return_value=types.SimpleNamespace(read_text=lambda _: None)))
            stack.enter_context(patch.dict("sys.modules", {"torch": fake_torch, "generate_melo": fake_backend}))
            stack.enter_context(patch.object(sys, "path", list(sys.path)))
            stack.enter_context(patch.dict(os.environ))
            opener = stack.enter_context(patch.object(self.runner, "open_run_folder", return_value=None))
            stack.enter_context(redirect_stdout(out))
            stack.enter_context(redirect_stderr(out))
            exit_code = self.runner.main(args)
        manifest_path = next((self.root / "outputs/evaluation").glob("*/manifest.json"))
        return exit_code, json.loads(manifest_path.read_text(encoding="utf-8")), manifest_path.parent.resolve(), out.getvalue(), opener

    def test_full_run_continues_after_both_mixed_failures_and_opens_when_requested(self):
        original = Path.cwd()
        code, manifest, folder, output, opener = self.run_main(["--all", "--repeat", "2", "--open-output"])
        self.assertEqual(code, 1)
        self.assertEqual(manifest["status"], "completed_with_failures")
        self.assertEqual(manifest["summary"]["passed_trials"], 12)
        self.assertEqual(manifest["summary"]["failed_trials"], 2)
        self.assertEqual(manifest["summary"]["failed_cases"], ["mixed_text"])
        self.assertIn("12 passed, 2 failed, 0 not completed", output)
        self.assertEqual(len(manifest["records"]), 14)
        self.assertEqual(len(list(folder.glob("*.wav"))), 12)
        for record in manifest["records"]:
            self.assertEqual(record["settings"], manifest["settings"])
            self.assertGreaterEqual(record["trial_elapsed_seconds"], 0)
            self.assertTrue(record["text"])
            if record["case_id"] == "mixed_text":
                self.assertIsNone(record["output_filename"])
                self.assertEqual(record["expected_output_filename"], f"mixed_language_{record['repeat']:02d}.wav")
                self.assertEqual(record["error"]["type"], "AssertionError")
                self.assertEqual(record["error"]["message"], "")
                self.assertIn("AssertionError", record["error"]["traceback"])
            else:
                self.assertTrue((folder / record["output_filename"]).exists())
        self.assertTrue((folder / "long_passage_02.wav").is_file())
        opener.assert_called_once_with(folder)
        self.assertEqual(Path.cwd(), original)

    def test_all_success_exits_zero_and_does_not_open_by_default(self):
        code, manifest, folder, output, opener = self.run_main(["--case", "narration"])
        self.assertEqual(code, 0)
        self.assertEqual(manifest["status"], "passed_wav_checks_listening_pending")
        self.assertEqual(manifest["summary"]["passed_cases"], ["narration"])
        self.assertEqual({p.name for p in folder.glob("*.wav")}, {"narration_01.wav", "narration_02.wav"})
        opener.assert_not_called()

    def test_startup_failure_records_all_trials_as_not_run(self):
        code, manifest, _, _, _ = self.run_main(["--all"], cuda=False)
        self.assertEqual(code, 1)
        self.assertEqual(manifest["status"], "aborted")
        self.assertEqual(manifest["summary"]["not_completed_trials"], 14)
        self.assertTrue(all(r["status"] == "not_run" for r in manifest["records"]))
        self.assertIn("requires CUDA", manifest["error"]["traceback"])

    def test_timestamp_collision_uses_readable_suffix_and_preserves_previous_run(self):
        instant = datetime(2026, 9, 13, 15, 4, 5)
        first = self.runner.create_run_directory(self.root, instant)
        (first / "keep.txt").write_text("keep", encoding="utf-8")
        second = self.runner.create_run_directory(self.root, instant)
        self.assertEqual(first.name, "melo_baseline_2026-09-13_15-04-05")
        self.assertEqual(second.name, first.name + "_02")
        self.assertEqual((first / "keep.txt").read_text(), "keep")

    def test_failed_only_case_has_nonzero_summary(self):
        code, manifest, folder, _, _ = self.run_main(["--case", "mixed_text"])
        self.assertEqual(code, 1)
        self.assertEqual(manifest["summary"]["failed_trials"], 2)
        self.assertEqual(list(folder.glob("*.wav")), [])

    def test_existing_named_audio_is_not_overwritten(self):
        destination = self.root / "narration_01.wav"
        destination.write_bytes(b"existing audio")
        with self.assertRaises(FileExistsError):
            self.runner.run_case(self.fake_synthesize, {"text": "你好"}, {}, self.root / "trial", destination)
        self.assertEqual(destination.read_bytes(), b"existing audio")

    def test_explorer_failure_is_a_warning(self):
        with patch.object(self.runner.os, "name", "nt"), patch.object(self.runner.os, "startfile", side_effect=OSError("no association"), create=True):
            warning = self.runner.open_run_folder(self.root)
        self.assertIn("Could not open run folder", warning)


if __name__ == "__main__":
    unittest.main()
