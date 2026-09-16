"""Model-free tests for Milestone D3 recovery and targeted regeneration."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import wave

from src.audiobook.manifest import create_planning_run
from src.audiobook.pipeline import (
    GenerationError,
    generate_planned_run,
    regenerate_scene,
    resume_generation,
)


ROOT = Path(__file__).resolve().parents[1]


def write_wav(path, frames=240, rate=24000):
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x01\x00" * frames)


class FakeBackend:
    def __init__(self, failing_text=None):
        self.failing_text = failing_text
        self.initialize_calls = 0
        self.calls = []

    def configuration(self):
        return {"backend": "fake_cosyvoice", "settings": {"stream": False}}

    def initialize(self):
        self.initialize_calls += 1
        return {**self.configuration(), "sample_rate_hz": 24000}

    def generate_scene(self, text, output_path):
        self.calls.append((text, Path(output_path)))
        if text == self.failing_text:
            raise RuntimeError("synthetic generation failure")
        write_wav(output_path, frames=240 + len(self.calls))
        return {"cosyvoice_chunks": 1, "inference_seconds": 0.01, "rtf": 1.0}


class AudiobookRecoveryTests(unittest.TestCase):
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
        self.tick = 0

    def clock(self):
        value = f"2026-09-15T00:00:{self.tick:02d}+00:00"
        self.tick += 1
        return value

    def generate(self, backend=None):
        backend = backend or FakeBackend()
        return generate_planned_run(self.run_dir, backend, clock=self.clock)

    def read_manifest(self):
        return json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8"))

    def test_resume_skips_valid_scenes_and_retries_failed_scene_once(self):
        self.generate(FakeBackend(failing_text="场景乙？\n"))
        backend = FakeBackend()
        manifest = resume_generation(self.run_dir, backend, clock=self.clock)
        self.assertEqual(backend.initialize_calls, 1)
        self.assertEqual([call[0] for call in backend.calls], ["场景乙？\n"])
        self.assertEqual(manifest["status"], "generated")
        scene = manifest["scenes"][1]["generation"]
        self.assertEqual([a["id"] for a in scene["attempts"]], ["attempt_001", "attempt_002"])
        self.assertEqual([a["status"] for a in scene["attempts"]], ["failed", "generated"])
        self.assertEqual(scene["selected_attempt_id"], "attempt_002")
        self.assertEqual(manifest["generation"]["last_operation"]["attempted_scenes"], 1)

    def test_resume_after_initialization_failure_generates_never_attempted_scenes(self):
        backend = FakeBackend()
        backend.initialize = Mock(side_effect=RuntimeError("no CUDA"))
        self.generate(backend)
        recovery = FakeBackend()
        manifest = resume_generation(self.run_dir, recovery, clock=self.clock)
        self.assertEqual(recovery.initialize_calls, 1)
        self.assertEqual(len(recovery.calls), 3)
        self.assertEqual(manifest["status"], "generated")
        self.assertEqual(
            manifest["generation"]["initialization_failure"]["error"]["message"],
            "no CUDA",
        )
        for scene in manifest["scenes"]:
            self.assertEqual([a["id"] for a in scene["generation"]["attempts"]], ["attempt_001"])

    def test_resume_with_all_valid_scenes_does_no_work_or_model_load(self):
        self.generate()
        backend = FakeBackend()
        manifest = resume_generation(self.run_dir, backend, clock=self.clock)
        self.assertEqual(backend.initialize_calls, 0)
        self.assertEqual(backend.calls, [])
        self.assertEqual(manifest["generation"]["last_operation"]["status"], "no_work")

    def test_targeted_success_changes_only_requested_scene_and_increments_ids(self):
        original = self.generate()
        untouched_before = [deepcopy(original["scenes"][0]), deepcopy(original["scenes"][2])]
        first_backend = FakeBackend()
        first = regenerate_scene(
            self.run_dir, "scene_0002", first_backend, clock=self.clock
        )
        scene = first["scenes"][1]["generation"]
        self.assertEqual([call[0] for call in first_backend.calls], ["场景乙？\n"])
        self.assertEqual([a["id"] for a in scene["attempts"]], ["attempt_001", "attempt_002"])
        self.assertEqual(scene["selected_attempt_id"], "attempt_002")
        self.assertEqual(first["scenes"][0], untouched_before[0])
        self.assertEqual(first["scenes"][2], untouched_before[1])

        second_backend = FakeBackend()
        second = regenerate_scene(
            self.run_dir, "scene_0002", second_backend, clock=self.clock
        )
        scene = second["scenes"][1]["generation"]
        self.assertEqual([a["id"] for a in scene["attempts"]], [
            "attempt_001", "attempt_002", "attempt_003",
        ])
        self.assertEqual(scene["selected_attempt_id"], "attempt_003")

    def test_failed_regeneration_preserves_previous_valid_selection_and_history(self):
        self.generate()
        backend = FakeBackend(failing_text="场景乙？\n")
        manifest = regenerate_scene(
            self.run_dir, "scene_0002", backend, clock=self.clock
        )
        generation = manifest["scenes"][1]["generation"]
        self.assertEqual(generation["selected_attempt_id"], "attempt_001")
        self.assertEqual([a["status"] for a in generation["attempts"]], ["generated", "failed"])
        self.assertEqual(manifest["status"], "generated")
        self.assertEqual(manifest["generation"]["last_operation"]["status"], "failed")
        self.assertEqual(len(backend.calls), 1)

    def test_corrupt_selected_wav_is_invalidated_and_replaced_on_resume(self):
        self.generate()
        selected_path = (
            self.run_dir / "scenes/scene_0002/attempt_001/generated.wav"
        )
        selected_path.write_bytes(b"corrupt partial wav")
        backend = FakeBackend()
        manifest = resume_generation(self.run_dir, backend, clock=self.clock)
        generation = manifest["scenes"][1]["generation"]
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(generation["attempts"][0]["status"], "generated")
        self.assertEqual(generation["attempts"][0]["artifact_status"], "invalid")
        self.assertTrue(generation["attempts"][0]["artifact_error"])
        self.assertEqual(generation["attempts"][1]["status"], "generated")
        self.assertEqual(generation["selected_attempt_id"], "attempt_002")

    def test_failed_resume_after_corruption_leaves_no_invalid_selection_or_retry_loop(self):
        self.generate()
        (self.run_dir / "scenes/scene_0002/attempt_001/generated.wav").unlink()
        backend = FakeBackend(failing_text="场景乙？\n")
        manifest = resume_generation(self.run_dir, backend, clock=self.clock)
        generation = manifest["scenes"][1]["generation"]
        self.assertEqual(len(backend.calls), 1)
        self.assertIsNone(generation["selected_attempt_id"])
        self.assertEqual([a["id"] for a in generation["attempts"]], ["attempt_001", "attempt_002"])
        self.assertEqual(generation["attempts"][1]["status"], "failed")
        self.assertEqual(manifest["status"], "generation_failed")

    def test_changed_source_prevents_recovery_before_backend_initialization(self):
        self.generate()
        (self.run_dir / "source.txt").write_text("被修改的正文。", encoding="utf-8")
        backend = FakeBackend()
        with self.assertRaisesRegex(GenerationError, "Source snapshot"):
            resume_generation(self.run_dir, backend, clock=self.clock)
        self.assertEqual(backend.initialize_calls, 0)

    def test_unknown_scene_and_invalid_unrelated_selection_stop_targeted_operation(self):
        self.generate()
        backend = FakeBackend()
        with self.assertRaisesRegex(GenerationError, "does not exist"):
            regenerate_scene(self.run_dir, "scene_9999", backend, clock=self.clock)
        self.assertEqual(backend.initialize_calls, 0)

        (self.run_dir / "scenes/scene_0001/attempt_001/generated.wav").unlink()
        with self.assertRaisesRegex(GenerationError, "run resume first"):
            regenerate_scene(self.run_dir, "scene_0002", backend, clock=self.clock)
        self.assertEqual(backend.initialize_calls, 0)

    def test_untracked_next_attempt_directory_stops_before_model_load_or_manifest_change(self):
        self.generate()
        untracked = self.run_dir / "scenes/scene_0002/attempt_002"
        untracked.mkdir()
        manifest_before = (self.run_dir / "manifest.json").read_bytes()
        backend = FakeBackend()
        with self.assertRaisesRegex(GenerationError, "Untracked next-attempt"):
            regenerate_scene(self.run_dir, "scene_0002", backend, clock=self.clock)
        self.assertEqual(backend.initialize_calls, 0)
        self.assertEqual((self.run_dir / "manifest.json").read_bytes(), manifest_before)

    def test_existing_d2_manifest_migrates_without_regenerating_valid_audio(self):
        manifest = self.generate()
        manifest["schema_version"] = 2
        for scene in manifest["scenes"]:
            generation = scene["generation"]
            attempt = generation.pop("attempts")[0]
            attempt.pop("status")
            generation.pop("selected_attempt_id")
            generation["attempt"] = attempt
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        backend = FakeBackend()
        migrated = resume_generation(self.run_dir, backend, clock=self.clock)
        self.assertEqual(migrated["schema_version"], 4)
        self.assertEqual(backend.initialize_calls, 0)
        self.assertTrue(all(
            scene["generation"]["selected_attempt_id"] == "attempt_001"
            for scene in migrated["scenes"]
        ))


if __name__ == "__main__":
    unittest.main()
