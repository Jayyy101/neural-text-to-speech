"""Read-only schema-5 progress and final artifact checks with fake audio."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.audiobook.manifest import create_planning_run
from src.audiobook.unit_execution import assemble_units, generate_units
from src.audiobook.unit_planning import prepare_synthesis_unit_run
from src.audiobook_application import inspect_run
from tests.test_audiobook_unit_execution import FakeASRWorker, FakeUnitBackend
from tests.test_audiobook_unit_planning import FixtureFrontend


class SchemaFiveApplicationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source = root / "chapter.txt"
        source.write_text("甲。乙。\n***\n丙。", encoding="utf-8")
        self.run_dir, _ = create_planning_run(
            source, "chapter", "run", root / "outputs"
        )
        prepared = prepare_synthesis_unit_run(self.run_dir, FixtureFrontend())
        self.frontend_hash = prepared["synthesis_unit_plan"]["frontend_identity_sha256"]
        worker = patch("src.audiobook.unit_execution.ASRWorkerClient", FakeASRWorker)
        worker.start()
        self.addCleanup(worker.stop)

    def manifest(self):
        return json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8"))

    def save(self, manifest):
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def generate(self, fail_text=None):
        return generate_units(
            self.run_dir, FakeUnitBackend(self.frontend_hash, fail_text), root_seed=17
        )

    def test_planning_and_partial_generation_progress(self):
        planned = inspect_run(self.run_dir)
        self.assertEqual(planned.schema_version, 5)
        self.assertEqual(planned.total_units, len(planned.units))
        self.assertEqual(planned.selected_units, 0)
        self.assertEqual(planned.generation_status, "not_started")
        self.assertFalse(planned.assembly.playable)

        units = [unit for scene in self.manifest()["scenes"]
                 for unit in scene["synthesis_units"]]
        self.generate(fail_text=units[-1]["normalized_text"])
        partial = inspect_run(self.run_dir)
        self.assertEqual(partial.total_units, len(units))
        self.assertEqual(partial.selected_units, len(units) - 1)
        self.assertEqual(partial.generation_summary["generated_units"], len(units) - 1)
        self.assertEqual(partial.generation_status, "generation_failed")
        self.assertEqual(partial.units[-1].status, "failed")
        self.assertEqual(partial.units[-1].latest_error["type"], "RuntimeError")

    def test_live_progress_reads_persisted_unit_selections(self):
        observations = []

        class ProbeBackend(FakeUnitBackend):
            def generate_unit(self, text, output_path, seed):
                observations.append(inspect_run(self_run_dir))
                return super().generate_unit(text, output_path, seed)

        self_run_dir = self.run_dir
        generate_units(self.run_dir, ProbeBackend(self.frontend_hash), root_seed=17)
        self.assertEqual(observations[0].selected_units, 0)
        self.assertEqual(observations[0].units[0].status, "synthesizing")
        self.assertEqual(observations[1].selected_units, 1)
        self.assertEqual(observations[1].generation_summary["generated_units"], 1)
        self.assertEqual(observations[1].generation_status, "running")

    def test_completed_wav_and_invalid_final_artifacts(self):
        self.generate()
        generated = inspect_run(self.run_dir)
        self.assertEqual(generated.selected_units, generated.total_units)
        self.assertEqual(generated.assembly.status, "not_assembled")
        self.assertFalse(generated.assembly.playable)

        assembled = assemble_units(self.run_dir)
        before = (self.run_dir / "manifest.json").read_bytes()
        complete = inspect_run(self.run_dir)
        self.assertTrue(complete.assembly.playable)
        self.assertEqual(complete.assembly.audio_path,
                         self.run_dir / "final" / "chapter.wav")
        self.assertEqual(before, (self.run_dir / "manifest.json").read_bytes())

        manifest = self.manifest()
        manifest["assembly"]["units"][0]["selected_attempt_id"] = "attempt_999"
        self.save(manifest)
        self.assertFalse(inspect_run(self.run_dir).assembly.playable)
        self.save(assembled)
        final_path = self.run_dir / "final" / "chapter.wav"
        original_final = final_path.read_bytes()
        final_path.write_bytes(original_final[:-1] + bytes([original_final[-1] ^ 1]))
        self.assertFalse(inspect_run(self.run_dir).assembly.playable)
        final_path.write_bytes(original_final)
        selected = assembled["scenes"][0]["synthesis_units"][0]["generation"]["attempts"][0]
        selected_path = self.run_dir / selected["output_path"]
        original = selected_path.read_bytes()
        selected_path.unlink()
        self.assertFalse(inspect_run(self.run_dir).assembly.playable)
        selected_path.write_bytes(original)
        self.assertTrue(inspect_run(self.run_dir).assembly.playable)
        (self.run_dir / "final" / "chapter.wav").unlink()
        missing = inspect_run(self.run_dir).assembly
        self.assertEqual(missing.status, "assembled")
        self.assertFalse(missing.playable)
        self.assertIsNotNone(missing.error)


if __name__ == "__main__":
    unittest.main()
