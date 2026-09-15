"""Model-free tests for Milestone D2 scene generation orchestration."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave

from src.audiobook.cosyvoice import CosyVoiceAdapter, PROMPT_PREFIX, wav_info
from src.audiobook.manifest import create_planning_run
from src.audiobook.pipeline import GenerationError, generate_planned_run


ROOT = Path(__file__).resolve().parents[1]


def write_wav(path, rate=24000, channels=1, width=2, frames=240):
    with wave.open(str(path), "wb") as audio:
        audio.setparams((channels, width, rate, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x01\x00" * frames * channels)


class FakeBackend:
    def __init__(self, failing_text=None, invalid_text=None):
        self.failing_text = failing_text
        self.invalid_text = invalid_text
        self.initialize_calls = 0
        self.calls = []

    def configuration(self):
        return {"backend": "fake_cosyvoice", "settings": {"stream": False}}

    def initialize(self):
        self.initialize_calls += 1
        return {
            **self.configuration(),
            "sample_rate_hz": 24000,
            "prompt_wav_sha256": "fake-reference-hash",
        }

    def generate_scene(self, text, output_path):
        self.calls.append((text, Path(output_path)))
        if text == self.failing_text:
            raise RuntimeError("synthetic generation failure")
        if text == self.invalid_text:
            Path(output_path).write_bytes(b"not a wav")
        else:
            write_wav(output_path)
        return {"cosyvoice_chunks": 2, "inference_seconds": 0.25, "rtf": 0.5}


class AudiobookGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = self.root / "chapter.txt"
        source.write_bytes("场景甲。\n***\n场景乙？\n***\n场景丙！".encode("utf-8"))
        self.run_dir, _ = create_planning_run(
            source, "chapter_0001", "run_001", self.root / "outputs",
            now=datetime(2026, 9, 15, tzinfo=timezone.utc),
        )
        self.times = iter(f"2026-09-15T00:00:{second:02d}+00:00" for second in range(30))

    def generate(self, backend):
        return generate_planned_run(self.run_dir, backend, clock=lambda: next(self.times))

    def test_order_exact_text_single_initialization_paths_and_metadata(self):
        backend = FakeBackend()
        manifest = self.generate(backend)
        self.assertEqual(backend.initialize_calls, 1)
        self.assertEqual([call[0] for call in backend.calls], ["场景甲。\n", "场景乙？\n", "场景丙！"])
        expected = [
            self.run_dir / "scenes" / f"scene_{index:04d}" / "attempt_001" / "generated.wav"
            for index in range(1, 4)
        ]
        self.assertEqual([call[1] for call in backend.calls], expected)
        self.assertTrue(all(path.is_file() for path in expected))
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["status"], "generated")
        self.assertEqual(manifest["generation"]["summary"], {
            "generated_scenes": 3, "failed_scenes": 0, "total_scenes": 3,
        })
        for index, scene in enumerate(manifest["scenes"], 1):
            generation = scene["generation"]
            attempt = generation["attempt"]
            self.assertEqual(generation["status"], "generated")
            self.assertEqual(attempt["id"], "attempt_001")
            self.assertEqual(
                attempt["output_path"],
                f"scenes/scene_{index:04d}/attempt_001/generated.wav",
            )
            self.assertEqual(attempt["cosyvoice_chunks"], 2)
            self.assertEqual(attempt["audio"]["sample_rate_hz"], 24000)
            self.assertEqual(attempt["audio"]["channels"], 1)
            self.assertEqual(len(attempt["wav_sha256"]), 64)
        saved = json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, manifest)

    def test_one_failure_is_recorded_others_continue_and_no_retry_occurs(self):
        backend = FakeBackend(failing_text="场景乙？\n")
        manifest = self.generate(backend)
        self.assertEqual(len(backend.calls), 3)
        self.assertEqual(backend.initialize_calls, 1)
        self.assertEqual(manifest["status"], "generation_failed")
        self.assertEqual(manifest["generation"]["summary"]["failed_scenes"], 1)
        failed = manifest["scenes"][1]["generation"]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["attempt"]["id"], "attempt_001")
        self.assertEqual(failed["attempt"]["error"]["type"], "RuntimeError")
        self.assertIn("synthetic generation failure", failed["attempt"]["error"]["message"])
        serialized = json.dumps(manifest)
        for forbidden in ("attempt_002", "retry", "repair", "assembly", "listening"):
            self.assertNotIn(forbidden, serialized)

    def test_structurally_invalid_wav_fails_scene(self):
        manifest = self.generate(FakeBackend(invalid_text="场景甲。\n"))
        first = manifest["scenes"][0]["generation"]
        self.assertEqual(first["status"], "failed")
        self.assertIn(first["attempt"]["error"]["type"], {"Error", "EOFError"})
        self.assertEqual(manifest["generation"]["summary"]["failed_scenes"], 1)

    def test_wav_validation_rejects_empty_wrong_rate_channels_and_width(self):
        cases = ((24000, 1, 2, 0), (16000, 1, 2, 2), (24000, 2, 2, 2), (24000, 1, 1, 2))
        for index, parameters in enumerate(cases):
            path = self.root / f"invalid_{index}.wav"
            write_wav(path, *parameters)
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                wav_info(path, 24000)

    def test_initialization_failure_is_persisted_and_no_scene_runs(self):
        backend = FakeBackend()
        backend.initialize = Mock(side_effect=RuntimeError("CUDA unavailable"))
        manifest = self.generate(backend)
        self.assertEqual(manifest["status"], "generation_failed")
        self.assertEqual(manifest["generation"]["error"]["type"], "RuntimeError")
        self.assertTrue(all(s["generation"]["status"] == "not_run" for s in manifest["scenes"]))
        self.assertEqual(backend.calls, [])

    def test_only_untouched_d1_run_can_generate(self):
        backend = FakeBackend()
        self.generate(backend)
        with self.assertRaisesRegex(GenerationError, "schema-version 1 planned"):
            self.generate(backend)
        self.assertEqual(backend.initialize_calls, 1)

    def test_tampered_d1_plan_is_rejected_before_backend_initialization(self):
        manifest_path = self.run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["plan_hash"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        backend = FakeBackend()
        with self.assertRaisesRegex(GenerationError, "deterministic source plan"):
            self.generate(backend)
        self.assertEqual(backend.initialize_calls, 0)

    def test_cli_help_is_model_free_and_exposes_only_plan_and_generate(self):
        result = subprocess.run(
            [sys.executable, "-B", "-m", "src.audiobook", "--help"],
            cwd=ROOT, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("{plan,generate}", result.stdout)
        for command in ("regenerate", "repair", "assemble", "resume"):
            self.assertNotIn(command, result.stdout)


class CosyVoiceAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cosyvoice_root = self.root / "CosyVoice"
        self.model_dir = self.cosyvoice_root / "pretrained_models/model"
        self.model_dir.mkdir(parents=True)
        (self.model_dir / "cosyvoice3.yaml").write_text("model: fake", encoding="utf-8")
        self.prompt_wav = self.root / "reference.wav"
        self.prompt_wav.write_bytes(b"fake reference")
        self.prompt_text = self.root / "reference.txt"
        self.prompt_text.write_text("参考声音。", encoding="utf-8")

    def test_adapter_loads_once_and_forwards_exact_zero_shot_contract(self):
        speech = Mock()
        speech.cpu.return_value = "cpu speech"
        model = Mock(sample_rate=24000)
        model.inference_zero_shot.return_value = [{"tts_speech": "chunk"}]
        torch = Mock(__version__="test-torch")
        torch.version.cuda = "test-cuda"
        torch.cuda.is_available.return_value = True
        torch.cuda.get_device_name.return_value = "test GPU"
        torch.cuda.max_memory_allocated.return_value = 1024**3
        torch.cat.return_value = speech
        torchaudio = Mock(__version__="test-audio")

        def save(path, value, rate, **settings):
            self.assertEqual(value, "cpu speech")
            self.assertEqual(settings, {"encoding": "PCM_S", "bits_per_sample": 16})
            write_wav(path, rate=rate)

        torchaudio.save.side_effect = save
        factory = Mock(return_value=model)
        adapter = CosyVoiceAdapter(
            self.cosyvoice_root, self.model_dir, self.prompt_wav, self.prompt_text
        )
        with patch("src.audiobook.cosyvoice.load_cosyvoice_runtime", return_value=(torch, torchaudio, factory)):
            first = adapter.initialize()
            second = adapter.initialize()
            result = adapter.generate_scene("精确场景文本。\n", self.root / "generated.wav")

        factory.assert_called_once_with(
            model_dir=str(self.model_dir), load_trt=False, load_vllm=False, fp16=False
        )
        model.inference_zero_shot.assert_called_once_with(
            "精确场景文本。\n", PROMPT_PREFIX + "参考声音。", str(self.prompt_wav), stream=False
        )
        self.assertEqual(first, second)
        self.assertEqual(result["cosyvoice_chunks"], 1)
        self.assertEqual(wav_info(self.root / "generated.wav", 24000)["frames"], 240)


if __name__ == "__main__":
    unittest.main()
