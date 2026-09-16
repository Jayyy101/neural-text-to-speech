"""Model-free acceptance tests for the end-to-end chapter workflow."""

from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

from src.audiobook.__main__ import main
from src.audiobook.cosyvoice import wav_info
from src.audiobook.planning import PlanningError
from src.audiobook.workflow import run_chapter


ROOT = Path(__file__).resolve().parents[1]


def read_payload(path):
    with wave.open(str(path), "rb") as audio:
        return audio.readframes(audio.getnframes())


class FakeBackend:
    def __init__(self, failing_call=None):
        self.failing_call = failing_call
        self.initialize_calls = 0
        self.calls = []
        self.payloads = []

    def configuration(self):
        return {"backend": "fake_cosyvoice", "settings": {"stream": False}}

    def initialize(self):
        self.initialize_calls += 1
        return {**self.configuration(), "sample_rate_hz": 24000}

    def generate_scene(self, text, output_path):
        call_number = len(self.calls) + 1
        self.calls.append((text, Path(output_path)))
        if call_number == self.failing_call:
            raise RuntimeError("synthetic workflow generation failure")
        frames = 100 + call_number
        payload = struct.pack(f"<{frames}h", *([call_number * 100] * frames))
        self.payloads.append(payload)
        with wave.open(str(output_path), "wb") as audio:
            audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            audio.writeframes(payload)
        return {"cosyvoice_chunks": 1}


class AudiobookWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "chapter.txt"
        self.source.write_bytes(
            "第一幕，渡口入夜。\n***\n第二幕，旧信出现。\n***\n第三幕，船离开岸边。".encode(
                "utf-8"
            )
        )
        self.output_root = self.root / "outputs"
        self.tick = 0

    def clock(self):
        value = f"2026-09-15T02:00:{self.tick:02d}+00:00"
        self.tick += 1
        return value

    def execute_workflow(self, backend, run_id="run_001"):
        return run_chapter(
            self.source, "chapter_0001", run_id, self.output_root, backend,
            now=datetime(2026, 9, 15, tzinfo=timezone.utc), clock=self.clock,
        )

    def test_end_to_end_plan_generate_validate_and_assemble_exact_pcm(self):
        backend = FakeBackend()
        run_directory, manifest = self.execute_workflow(backend)
        self.assertEqual(backend.initialize_calls, 1)
        self.assertEqual(
            [call[0] for call in backend.calls],
            ["第一幕，渡口入夜。\n", "第二幕，旧信出现。\n", "第三幕，船离开岸边。"],
        )
        self.assertEqual(manifest["chapter_id"], "chapter_0001")
        self.assertEqual(manifest["run_id"], "run_001")
        self.assertEqual(manifest["status"], "generated")
        self.assertEqual(manifest["generation"]["summary"], {
            "generated_scenes": 3, "failed_scenes": 0, "total_scenes": 3,
        })
        self.assertTrue(all(
            scene["generation"]["selected_attempt_id"] == "attempt_001"
            for scene in manifest["scenes"]
        ))
        self.assertTrue(all(len(scene["generation"]["attempts"]) == 1
                            for scene in manifest["scenes"]))
        self.assertTrue(all("repair" not in scene for scene in manifest["scenes"]))

        assembly = manifest["assembly"]
        final_path = run_directory / assembly["output_path"]
        self.assertEqual(assembly["status"], "assembled")
        self.assertEqual(assembly["extra_silence_ms_between_scenes"], 0)
        self.assertEqual(
            [item["scene_id"] for item in assembly["scenes"]],
            ["scene_0001", "scene_0002", "scene_0003"],
        )
        self.assertEqual(read_payload(final_path), b"".join(backend.payloads))
        self.assertEqual(wav_info(final_path, 24000)["frames"], 306)
        saved = json.loads(
            (run_directory / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(saved, manifest)

    def test_generation_failure_records_once_and_prevents_assembly(self):
        backend = FakeBackend(failing_call=2)
        run_directory, manifest = self.execute_workflow(backend)
        self.assertEqual(backend.initialize_calls, 1)
        self.assertEqual(len(backend.calls), 3)
        self.assertEqual(manifest["status"], "generation_failed")
        self.assertEqual(manifest["generation"]["summary"]["failed_scenes"], 1)
        self.assertNotIn("assembly", manifest)
        self.assertFalse((run_directory / "final/chapter.wav").exists())
        self.assertTrue(all(len(scene["generation"]["attempts"]) == 1
                            for scene in manifest["scenes"]))
        serialized = json.dumps(manifest)
        self.assertNotIn("attempt_002", serialized)
        self.assertNotIn('"repair"', serialized)

    def test_existing_run_collision_precedes_backend_initialization(self):
        first_backend = FakeBackend()
        run_directory, _ = self.execute_workflow(first_backend)
        manifest_before = (run_directory / "manifest.json").read_bytes()
        second_backend = FakeBackend()
        with self.assertRaisesRegex(PlanningError, "already exists"):
            self.execute_workflow(second_backend)
        self.assertEqual(second_backend.initialize_calls, 0)
        self.assertEqual(second_backend.calls, [])
        self.assertEqual((run_directory / "manifest.json").read_bytes(), manifest_before)

    def test_run_cli_uses_injected_backend_and_reports_completed_chapter(self):
        backend = FakeBackend()
        with patch("src.audiobook.__main__.create_adapter", return_value=backend):
            result = main([
                "run", str(self.source),
                "--chapter-id", "chapter_cli",
                "--run-id", "run_cli",
                "--output-root", str(self.output_root),
            ])
        self.assertEqual(result, 0)
        self.assertEqual(backend.initialize_calls, 1)
        self.assertTrue(
            (self.output_root / "chapter_cli/run_cli/final/chapter.wav").is_file()
        )

    def test_run_cli_returns_nonzero_and_does_not_assemble_after_failure(self):
        backend = FakeBackend(failing_call=2)
        with patch("src.audiobook.__main__.create_adapter", return_value=backend):
            result = main([
                "run", str(self.source),
                "--chapter-id", "chapter_failed_cli",
                "--run-id", "run_failed_cli",
                "--output-root", str(self.output_root),
            ])
        run_directory = self.output_root / "chapter_failed_cli/run_failed_cli"
        self.assertEqual(result, 1)
        self.assertEqual(backend.initialize_calls, 1)
        self.assertFalse((run_directory / "final/chapter.wav").exists())
        manifest = json.loads(
            (run_directory / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["status"], "generation_failed")
        self.assertNotIn("assembly", manifest)


if __name__ == "__main__":
    unittest.main()
