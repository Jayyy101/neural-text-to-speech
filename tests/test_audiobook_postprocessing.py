"""Model-free tests for Milestone D4 repair and chapter assembly."""

from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

from evaluation.apply_pause_plan import apply_pause_plan as validated_apply_pause_plan
from src.audiobook.cosyvoice import file_sha256
from src.audiobook.manifest import create_planning_run
from src.audiobook.pipeline import GenerationError, generate_planned_run, regenerate_scene
from src.audiobook.postprocessing import assemble_chapter, repair_scene


ROOT = Path(__file__).resolve().parents[1]


def write_wav(path, samples, rate=1000, channels=1):
    with wave.open(str(path), "wb") as audio:
        audio.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        audio.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def read_payload(path):
    with wave.open(str(path), "rb") as audio:
        return audio.getparams(), audio.readframes(audio.getnframes())


class FakeBackend:
    def __init__(self):
        self.calls = 0

    def configuration(self):
        return {"backend": "fake_cosyvoice", "settings": {"stream": False}}

    def initialize(self):
        return {**self.configuration(), "sample_rate_hz": 1000}

    def generate_scene(self, text, output_path):
        self.calls += 1
        amplitude = 500 + ord(text[0]) % 1000
        samples = [amplitude] * 1000
        samples[450:500] = [0] * 50
        write_wav(output_path, samples)
        return {"cosyvoice_chunks": 1}


class AudiobookPostprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = self.root / "chapter.txt"
        source.write_text("甲场景。\n***\n乙场景。\n***\n丙场景。", encoding="utf-8")
        self.run_dir, _ = create_planning_run(
            source, "chapter_0001", "run_001", self.root / "outputs",
            now=datetime(2026, 9, 15, tzinfo=timezone.utc),
        )
        self.tick = 0
        generate_planned_run(self.run_dir, FakeBackend(), clock=self.clock)

    def clock(self):
        value = f"2026-09-15T01:00:{self.tick:02d}+00:00"
        self.tick += 1
        return value

    def read_manifest(self):
        return json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8"))

    def write_plan(self, scene_id="scene_0002", **source_overrides):
        manifest = self.read_manifest()
        scene = next(item for item in manifest["scenes"] if item["id"] == scene_id)
        attempt = scene["generation"]["attempts"][0]
        source = {
            "scene_id": scene_id,
            "attempt_id": attempt["id"],
            "wav_sha256": attempt["wav_sha256"],
        }
        source.update(source_overrides)
        path = self.root / f"{scene_id}-pause-plan.json"
        path.write_text(json.dumps({
            "source": source,
            "pauses": [{"around": 0.6, "label": "period"}],
        }), encoding="utf-8")
        return path

    def test_repair_reuses_validated_logic_preserves_source_and_selects_separate_artifact(self):
        plan = self.write_plan()
        source_path = self.run_dir / "scenes/scene_0002/attempt_001/generated.wav"
        source_before = source_path.read_bytes()
        with patch(
            "src.audiobook.postprocessing.apply_pause_plan",
            wraps=validated_apply_pause_plan,
        ) as apply:
            manifest = repair_scene(
                self.run_dir, "scene_0002", plan, clock=self.clock
            )
        apply.assert_called_once()
        self.assertEqual(source_path.read_bytes(), source_before)
        state = manifest["scenes"][1]["repair"]
        self.assertEqual(state["selected_repair_id"], "repair_001")
        repair = state["repairs"][0]
        self.assertEqual(repair["source"]["attempt_id"], "attempt_001")
        self.assertEqual(repair["result"]["total_actual_add_ms"], 140)
        self.assertNotEqual(repair["output_path"], "scenes/scene_0002/attempt_001/generated.wav")
        self.assertTrue((self.run_dir / repair["output_path"]).is_file())
        self.assertEqual(repair["audio"]["frames"], 1140)

    def test_stale_plan_identity_is_rejected_without_artifacts_or_manifest_change(self):
        plan = self.write_plan(wav_sha256="0" * 64)
        manifest_before = (self.run_dir / "manifest.json").read_bytes()
        with self.assertRaisesRegex(GenerationError, "stale"):
            repair_scene(self.run_dir, "scene_0002", plan, clock=self.clock)
        self.assertEqual((self.run_dir / "manifest.json").read_bytes(), manifest_before)
        self.assertFalse((self.run_dir / "scenes/scene_0002/repairs").exists())

    def test_plan_for_previous_selected_attempt_is_rejected_after_regeneration(self):
        old_plan = self.write_plan()
        regenerate_scene(
            self.run_dir, "scene_0002", FakeBackend(), clock=self.clock
        )
        with self.assertRaisesRegex(GenerationError, "stale"):
            repair_scene(self.run_dir, "scene_0002", old_plan, clock=self.clock)
        self.assertFalse((self.run_dir / "scenes/scene_0002/repairs").exists())

    def test_regeneration_deselects_old_repair_and_marks_assembly_stale(self):
        repair_scene(
            self.run_dir, "scene_0002", self.write_plan(), clock=self.clock
        )
        assemble_chapter(self.run_dir, clock=self.clock)
        manifest = regenerate_scene(
            self.run_dir, "scene_0002", FakeBackend(), clock=self.clock
        )
        scene = manifest["scenes"][1]
        self.assertEqual(scene["generation"]["selected_attempt_id"], "attempt_002")
        self.assertIsNone(scene["repair"]["selected_repair_id"])
        self.assertEqual(len(scene["repair"]["repairs"]), 1)
        self.assertEqual(manifest["assembly"]["status"], "stale")

    def test_assembly_uses_plan_order_selected_repair_and_exact_pcm_with_zero_extra(self):
        repaired = repair_scene(
            self.run_dir, "scene_0002", self.write_plan(), clock=self.clock
        )
        selected_paths = []
        expected_payload = bytearray()
        for scene in repaired["scenes"]:
            if scene["id"] == "scene_0002":
                record = scene["repair"]["repairs"][0]
            else:
                record = scene["generation"]["attempts"][0]
            path = self.run_dir / record["output_path"]
            selected_paths.append(record["output_path"])
            expected_payload.extend(read_payload(path)[1])

        manifest = assemble_chapter(self.run_dir, clock=self.clock)
        assembly = manifest["assembly"]
        output = self.run_dir / assembly["output_path"]
        params, payload = read_payload(output)
        self.assertEqual(payload, bytes(expected_payload))
        self.assertEqual(params.nframes, 3140)
        self.assertEqual(assembly["extra_silence_ms_between_scenes"], 0)
        self.assertEqual(
            [item["scene_id"] for item in assembly["scenes"]],
            ["scene_0001", "scene_0002", "scene_0003"],
        )
        self.assertEqual(
            [item["artifact_type"] for item in assembly["scenes"]],
            ["generation", "repair", "generation"],
        )
        self.assertEqual(
            [item["artifact_path"] for item in assembly["scenes"]], selected_paths
        )
        self.assertEqual(
            [(item["start_frame"], item["end_frame_exclusive"])
             for item in assembly["scenes"]],
            [(0, 1000), (1000, 2140), (2140, 3140)],
        )
        self.assertFalse((self.run_dir / "final/chapter.partial.wav").exists())

    def test_missing_or_corrupt_selected_artifact_prevents_assembly(self):
        selected = self.run_dir / "scenes/scene_0001/attempt_001/generated.wav"
        original = selected.read_bytes()
        for replacement in (None, b"corrupt"):
            with self.subTest(replacement=replacement):
                if replacement is None:
                    selected.unlink()
                else:
                    selected.write_bytes(replacement)
                with self.assertRaisesRegex(GenerationError, "invalid"):
                    assemble_chapter(self.run_dir, clock=self.clock)
                self.assertFalse((self.run_dir / "final/chapter.wav").exists())
                selected.write_bytes(original)

    def test_incompatible_wav_formats_are_rejected(self):
        first = self.root / "first.wav"
        second = self.root / "second.wav"
        third = self.root / "third.wav"
        write_wav(first, [1] * 20)
        write_wav(second, [2] * 40, channels=2)
        write_wav(third, [3] * 20)
        artifacts = [
            (str(path.relative_to(self.run_dir)) if path.is_relative_to(self.run_dir)
             else path.name, "generation", f"attempt_{index:03d}",
             {"wav_sha256": file_sha256(path)})
            for index, path in enumerate((first, second, third), 1)
        ]
        # Put fixtures under the run so path containment remains part of the test.
        for path in (first, second, third):
            target = self.run_dir / path.name
            path.replace(target)
        artifacts = [
            (name, kind, artifact_id,
             {"wav_sha256": file_sha256(self.run_dir / name)})
            for name, kind, artifact_id, _ in artifacts
        ]
        with patch(
            "src.audiobook.postprocessing.resolve_scene_artifact",
            side_effect=artifacts,
        ):
            with self.assertRaisesRegex(GenerationError, "incompatible"):
                assemble_chapter(self.run_dir, clock=self.clock)

    def test_selecting_repair_after_assembly_marks_previous_chapter_stale(self):
        assemble_chapter(self.run_dir, clock=self.clock)
        manifest = repair_scene(
            self.run_dir, "scene_0002", self.write_plan(), clock=self.clock
        )
        self.assertEqual(manifest["assembly"]["status"], "stale")
        self.assertIn("selected repair changed", manifest["assembly"]["stale_reason"])


if __name__ == "__main__":
    unittest.main()
