"""Model-free tests for the Milestone E1 read-only run inspector."""

from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import Mock
import wave

from src.audiobook.manifest import create_planning_run
from src.audiobook.pipeline import GenerationError, generate_planned_run, regenerate_scene
from src.audiobook.postprocessing import assemble_chapter, repair_scene
from src.audiobook_application import inspect_run, open_audio_file


ROOT = Path(__file__).resolve().parents[1]


def write_wav(path, value, frames=1000, rate=1000):
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        audio.writeframes(struct.pack(f"<{frames}h", *([value] * frames)))


class FakeBackend:
    def __init__(self):
        self.calls = 0

    def configuration(self):
        return {"backend": "fake_cosyvoice", "settings": {"stream": False}}

    def initialize(self):
        return {**self.configuration(), "sample_rate_hz": 1000}

    def generate_scene(self, text, output_path, seed=None):
        self.calls += 1
        write_wav(output_path, self.calls + (seed or 0))
        return {"cosyvoice_chunks": 1}


class FailingBackend(FakeBackend):
    def generate_scene(self, text, output_path, seed=None):
        if text.startswith("第二"):
            raise RuntimeError("synthetic scene failure")
        return super().generate_scene(text, output_path, seed)


class AudiobookApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = self.root / "chapter.txt"
        source.write_text("第一场。\n***\n第二场。", encoding="utf-8")
        self.run_dir, _ = create_planning_run(
            source, "chapter_0001", "run_001", self.root / "outputs",
            now=datetime(2026, 9, 15, tzinfo=timezone.utc),
        )
        self.tick = 0

    def clock(self):
        value = f"2026-09-15T12:00:{self.tick:02d}+00:00"
        self.tick += 1
        return value

    @property
    def manifest_path(self):
        return self.run_dir / "manifest.json"

    def read_manifest(self):
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def write_manifest(self, manifest):
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def generate(self):
        return generate_planned_run(self.run_dir, FakeBackend(), clock=self.clock)

    def write_repair_plan(self, scene_index=0):
        manifest = self.read_manifest()
        scene = manifest["scenes"][scene_index]
        attempt = scene["generation"]["attempts"][0]
        plan = self.root / "pause-plan.json"
        plan.write_text(json.dumps({
            "source": {
                "scene_id": scene["id"],
                "attempt_id": attempt["id"],
                "wav_sha256": attempt["wav_sha256"],
            },
            "pauses": [{"around": 0.5, "label": "period"}],
        }), encoding="utf-8")
        return plan

    def test_planned_schema_is_inspected_without_generation_or_assembly(self):
        result = inspect_run(self.run_dir)
        self.assertEqual(result.schema_version, 1)
        self.assertEqual(result.generation_status, "not_started")
        self.assertEqual(result.assembly.status, "not_assembled")
        self.assertFalse(result.assembly.playable)
        self.assertEqual([scene.id for scene in result.scenes], ["scene_0001", "scene_0002"])
        self.assertTrue(all(not scene.attempts for scene in result.scenes))

    def test_generation_and_assembly_are_reported_separately(self):
        self.generate()
        generated = inspect_run(self.run_dir)
        self.assertEqual(generated.generation_status, "generated")
        self.assertEqual(generated.assembly.status, "not_assembled")
        self.assertFalse(generated.assembly.playable)

        assemble_chapter(self.run_dir, clock=self.clock)
        assembled = inspect_run(self.run_dir)
        self.assertEqual(assembled.generation_status, "generated")
        self.assertEqual(assembled.assembly.status, "assembled")
        self.assertTrue(assembled.assembly.playable)
        self.assertTrue(assembled.assembly.audio_path.is_file())

        manifest = self.read_manifest()
        manifest["assembly"]["status"] = "stale"
        manifest["assembly"]["stale_reason"] = "test selection changed"
        self.write_manifest(manifest)
        stale = inspect_run(self.run_dir)
        self.assertEqual(stale.generation_status, "generated")
        self.assertEqual(stale.assembly.status, "stale")
        self.assertFalse(stale.assembly.playable)
        self.assertEqual(stale.assembly.stale_reason, "test selection changed")

    def test_attempts_explicit_seed_and_missing_historical_seed_are_distinct(self):
        self.generate()
        regenerate_scene(
            self.run_dir, "scene_0001", FakeBackend(), clock=self.clock, seed=17
        )
        manifest = self.read_manifest()
        manifest["scenes"][0]["generation"]["attempts"][0].pop("random_state")
        self.write_manifest(manifest)

        scene = inspect_run(self.run_dir).scenes[0]
        self.assertEqual(scene.selected_attempt_id, "attempt_002")
        first, second = scene.attempts
        self.assertFalse(first.seed_recorded)
        self.assertIsNone(first.seed)
        self.assertTrue(second.seed_recorded)
        self.assertEqual(second.seed, 17)
        self.assertEqual(second.random_policy, "explicit_global_seed")

    def test_schema_two_and_three_generation_manifests_remain_inspectable(self):
        self.generate()
        for schema in (2, 3):
            with self.subTest(schema=schema):
                manifest = self.read_manifest()
                manifest["schema_version"] = schema
                self.write_manifest(manifest)
                before = self.manifest_path.read_bytes()
                result = inspect_run(self.run_dir)
                self.assertEqual(result.schema_version, schema)
                self.assertEqual(result.generation_status, "generated")
                self.assertEqual(self.manifest_path.read_bytes(), before)

    def test_selected_repair_resolves_and_invalid_repair_never_falls_back(self):
        self.generate()
        repaired = repair_scene(
            self.run_dir, "scene_0001", self.write_repair_plan(), clock=self.clock
        )
        repair = repaired["scenes"][0]["repair"]["repairs"][0]
        valid = inspect_run(self.run_dir).scenes[0]
        self.assertEqual(valid.resolved_artifact_type, "repair")
        self.assertEqual(valid.resolved_artifact_id, "repair_001")
        self.assertEqual(valid.resolved_audio_path, self.run_dir / repair["output_path"])

        (self.run_dir / repair["output_path"]).unlink()
        invalid = inspect_run(self.run_dir).scenes[0]
        self.assertEqual(invalid.selected_repair_id, "repair_001")
        self.assertIsNone(invalid.resolved_audio_path)
        self.assertIsNotNone(invalid.resolution_error)
        self.assertIn("selected repair is invalid", invalid.resolution_error.lower())

    def test_missing_selected_wav_is_visible_and_not_playable(self):
        manifest = self.generate()
        attempt = manifest["scenes"][1]["generation"]["attempts"][0]
        (self.run_dir / attempt["output_path"]).unlink()
        scene = inspect_run(self.run_dir).scenes[1]
        self.assertIsNone(scene.resolved_audio_path)
        self.assertIn("selected attempt is invalid", scene.resolution_error.lower())
        self.assertFalse(scene.attempts[0].artifact_valid)

    def test_generation_errors_are_preserved_for_inspection(self):
        generate_planned_run(self.run_dir, FailingBackend(), clock=self.clock)
        result = inspect_run(self.run_dir)
        failed = result.scenes[1]
        self.assertEqual(result.generation_status, "generation_failed")
        self.assertEqual(failed.generation_status, "failed")
        self.assertEqual(failed.attempts[0].error["type"], "RuntimeError")
        self.assertEqual(failed.attempts[0].error["message"], "synthetic scene failure")
        self.assertIsNone(failed.resolved_audio_path)

    def test_missing_final_wav_disables_chapter_playback(self):
        self.generate()
        manifest = assemble_chapter(self.run_dir, clock=self.clock)
        (self.run_dir / manifest["assembly"]["output_path"]).unlink()
        assembly = inspect_run(self.run_dir).assembly
        self.assertEqual(assembly.status, "assembled")
        self.assertFalse(assembly.playable)
        self.assertIsNone(assembly.audio_path)
        self.assertIsNotNone(assembly.error)

    def test_malformed_and_unsupported_manifests_are_rejected(self):
        self.manifest_path.write_text("{not json", encoding="utf-8")
        with self.assertRaisesRegex(GenerationError, "Cannot read audiobook manifest"):
            inspect_run(self.run_dir)

        self.manifest_path.write_text(
            json.dumps({"schema_version": 99}), encoding="utf-8"
        )
        with self.assertRaisesRegex(GenerationError, "Unsupported audiobook schema"):
            inspect_run(self.run_dir)

    def test_inspection_preserves_all_run_files_byte_for_byte(self):
        self.generate()
        assemble_chapter(self.run_dir, clock=self.clock)
        before = {
            path.relative_to(self.run_dir): path.read_bytes()
            for path in self.run_dir.rglob("*") if path.is_file()
        }
        inspect_run(self.run_dir)
        after = {
            path.relative_to(self.run_dir): path.read_bytes()
            for path in self.run_dir.rglob("*") if path.is_file()
        }
        self.assertEqual(after, before)

    def test_system_player_helper_accepts_injected_opener_and_rejects_missing_file(self):
        audio_path = self.root / "clip.wav"
        write_wav(audio_path, 1, frames=10)
        opener = Mock()
        open_audio_file(audio_path, opener=opener)
        opener.assert_called_once_with(audio_path.resolve())
        with self.assertRaises(FileNotFoundError):
            open_audio_file(self.root / "missing.wav", opener=opener)


if __name__ == "__main__":
    unittest.main()
