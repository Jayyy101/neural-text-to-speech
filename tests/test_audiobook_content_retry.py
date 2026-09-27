"""Model-free bounded content-QC retry behavior for new schema-5 runs."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.audiobook import unit_execution
from src.audiobook.content_qc import MODEL_ID, MODEL_REVISION, han_tokens
from src.audiobook.manifest import create_planning_run
from src.audiobook.pipeline import GenerationError
from src.audiobook.unit_execution import (
    CONTENT_RETRY_POLICY, QC_EXECUTION_POLICY, RETRY_EXECUTION_POLICY,
    assemble_units, derive_unit_seed, generate_units,
)
from src.audiobook.unit_planning import prepare_synthesis_unit_run
from tests.test_audiobook_content_qc import FakeBackend, recognition
from tests.test_audiobook_unit_planning import FixtureFrontend


class ScriptedRetryWorker:
    def __init__(self, transcripts, failures=()):
        self.transcripts = transcripts
        self.failures = set(failures)
        self.requests = []
        self.model = {"model_id": MODEL_ID, "resolved_revision": MODEL_REVISION}
        self.close_calls = 0

    def recognize(self, request):
        self.requests.append(request)
        assert set(request) == {"type", "request_id", "audio_path", "wav_sha256"}
        unit_id, attempt_id = request["request_id"].split(":")
        if (unit_id, attempt_id) in self.failures:
            raise RuntimeError("scripted ASR infrastructure failure")
        text = self.transcripts.get((unit_id, attempt_id), self.transcripts[unit_id])
        return recognition(text, request["request_id"], request["wav_sha256"])

    def close(self):
        self.close_calls += 1


class RetryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source = root / "source.txt"
        source.write_text("甲乙丙丁戊己。庚辛壬癸。", encoding="utf-8")
        self.run_dir, _ = create_planning_run(source, "chapter", "run", root / "out")
        prepared = prepare_synthesis_unit_run(self.run_dir, FixtureFrontend())
        self.frontend_hash = prepared["synthesis_unit_plan"]["frontend_identity_sha256"]
        self.ids = [unit["id"] for unit in self.units(prepared)]
        self.good = {unit["id"]: "".join(han_tokens(unit["normalized_text"]))
                     for unit in self.units(prepared)}
        self.bad = self.good[self.ids[0]][0] + self.good[self.ids[0]][-1]
        self.workers = []

    @staticmethod
    def units(manifest):
        return [unit for scene in manifest["scenes"] for unit in scene["synthesis_units"]]

    def backend(self):
        return FakeBackend(self.frontend_hash)

    def factory(self, transcripts=None, failures=()):
        def create(_asr_python, _log_path):
            worker = ScriptedRetryWorker(transcripts or self.good, failures)
            self.workers.append(worker)
            return worker
        return create

    def execute(self, transcripts=None, failures=(), backend=None):
        backend = backend or self.backend()
        manifest = generate_units(
            self.run_dir, backend, root_seed=19,
            worker_factory=self.factory(transcripts, failures),
        )
        return manifest, backend

    def scripted(self, rejected_attempts):
        return {**self.good, **{
            (self.ids[0], attempt_id): self.bad for attempt_id in rejected_attempts
        }}

    def save(self, manifest):
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )

    def test_first_attempt_passes_without_retry_and_persists_policy(self):
        manifest, backend = self.execute()
        self.assertEqual(manifest["generation"]["content_retry"], CONTENT_RETRY_POLICY)
        self.assertEqual(manifest["unit_execution_policy"], RETRY_EXECUTION_POLICY)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(len(self.workers), 1)
        self.assertEqual(len(self.workers[0].requests), 2)
        self.assertTrue(all(len(unit["generation"]["attempts"]) == 1
                            for unit in self.units(manifest)))
        self.assertTrue(all(unit["generation"]["retry_state"]["status"] == "selected"
                            for unit in self.units(manifest)))

    def test_first_reject_second_pass_selects_and_preserves_history(self):
        manifest, backend = self.execute(self.scripted(["attempt_001"]))
        unit, other = self.units(manifest)
        state = unit["generation"]
        attempts = state["attempts"]
        self.assertEqual(len(backend.calls), 3)
        self.assertEqual(backend.initialize_calls, 1)
        self.assertEqual(len(self.workers), 1)
        self.assertEqual(len(self.workers[0].requests), 3)
        self.assertEqual([item["take_index"] for item in attempts], [1, 2])
        self.assertEqual([item["content_qc"]["status"] for item in attempts],
                         ["rejected", "passed"])
        self.assertEqual(state["selected_attempt_id"], "attempt_002")
        self.assertEqual([entry["selected_attempt_id"]
                          for entry in state["selection_history"]], ["attempt_002"])
        self.assertEqual(len(other["generation"]["attempts"]), 1)
        self.assertEqual(other["generation"]["selected_attempt_id"], "attempt_001")
        self.assertEqual(state["retry_state"]["status"], "selected")
        plan_hash = manifest["synthesis_unit_plan"]["ordered_unit_plan_sha256"]
        for index, attempt in enumerate(attempts, 1):
            self.assertEqual(attempt["seed"], derive_unit_seed(19, plan_hash, unit["id"], index))
            self.assertEqual(attempt["seed"], backend.calls[index - 1][2])
            wav = self.run_dir / attempt["output_path"]
            sidecar = self.run_dir / attempt["content_qc"]["evidence_path"]
            self.assertEqual(hashlib.sha256(wav.read_bytes()).hexdigest(), attempt["wav_sha256"])
            self.assertEqual(hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                             attempt["content_qc"]["evidence_sha256"])
        self.assertNotEqual(attempts[0]["seed"], attempts[1]["seed"])
        self.assertEqual(assemble_units(self.run_dir)["assembly"]["status"], "assembled")
        before = [(item["id"], item["seed"], item["wav_sha256"],
                   item["content_qc"]["evidence_sha256"]) for item in attempts]
        resume_backend = self.backend()
        resumed = generate_units(self.run_dir, resume_backend,
                                 worker_factory=self.factory())
        self.assertEqual(resume_backend.calls, [])
        self.assertEqual(len(self.workers), 1)
        self.assertEqual(before, [(item["id"], item["seed"], item["wav_sha256"],
                                   item["content_qc"]["evidence_sha256"])
                                  for item in self.units(resumed)[0]["generation"]["attempts"]])

    def test_two_reject_then_third_pass_has_distinct_stable_seeds(self):
        manifest, backend = self.execute(self.scripted(["attempt_001", "attempt_002"]))
        unit, other = self.units(manifest)
        attempts = unit["generation"]["attempts"]
        self.assertEqual([item["id"] for item in attempts],
                         ["attempt_001", "attempt_002", "attempt_003"])
        self.assertEqual([item["take_index"] for item in attempts], [1, 2, 3])
        self.assertEqual([item["content_qc"]["status"] for item in attempts],
                         ["rejected", "rejected", "passed"])
        self.assertEqual(unit["generation"]["selected_attempt_id"], "attempt_003")
        self.assertEqual(len(other["generation"]["attempts"]), 1)
        self.assertEqual(len(backend.calls), 4)
        self.assertEqual(len(set(item["seed"] for item in attempts)), 3)
        self.assertEqual(assemble_units(self.run_dir)["assembly"]["status"], "assembled")
        previous_seeds = [item["seed"] for item in attempts]
        resumed = generate_units(self.run_dir, self.backend(),
                                 worker_factory=self.factory())
        self.assertEqual([item["seed"] for item in self.units(resumed)[0][
            "generation"]["attempts"]], previous_seeds)

    def test_three_rejections_exhaust_without_fourth_attempt(self):
        transcripts = self.scripted(["attempt_001", "attempt_002", "attempt_003"])
        manifest, backend = self.execute(transcripts)
        state = self.units(manifest)[0]["generation"]
        self.assertEqual(len(backend.calls), 4)
        self.assertEqual(len(state["attempts"]), 3)
        self.assertEqual(state["retry_state"]["status"], "exhausted")
        self.assertEqual(state["retry_state"]["reason"],
                         "all_allowed_attempts_rejected_for_content")
        self.assertIsNone(state["selected_attempt_id"])
        self.assertEqual(state["selection_history"], [])
        self.assertTrue(all((self.run_dir / item["output_path"]).is_file()
                            for item in state["attempts"]))
        with self.assertRaisesRegex(GenerationError, "Every synthesis unit"):
            assemble_units(self.run_dir)
        resume_backend = self.backend()
        resumed = generate_units(self.run_dir, resume_backend,
                                 worker_factory=self.factory())
        self.assertEqual(resume_backend.calls, [])
        self.assertEqual(len(self.workers), 1)
        self.assertEqual(len(self.units(resumed)[0]["generation"]["attempts"]), 3)

    def test_asr_error_uses_same_wav_on_resume_before_retry(self):
        failure = {(self.ids[0], "attempt_001")}
        first, backend = self.execute(failures=failure)
        state = self.units(first)[0]["generation"]
        self.assertEqual(state["retry_state"]["status"], "qc_error")
        self.assertEqual(len(state["attempts"]), 1)
        old_hash = state["attempts"][0]["wav_sha256"]
        next_backend = self.backend()
        resumed = generate_units(self.run_dir, next_backend,
                                 worker_factory=self.factory())
        self.assertEqual(next_backend.calls, [])
        self.assertEqual(next_backend.initialize_calls, 0)
        current = self.units(resumed)[0]["generation"]
        self.assertEqual(len(current["attempts"]), 1)
        self.assertEqual(current["attempts"][0]["wav_sha256"], old_hash)
        self.assertEqual(current["retry_state"]["status"], "selected")

    def test_short_deletion_and_substitution_noise_do_not_retry(self):
        # Three missing Han characters are below the fixed four-character gate.
        short = self.good[self.ids[0]][0] + self.good[self.ids[0]][4:]
        manifest, backend = self.execute({**self.good, self.ids[0]: short})
        self.assertEqual(len(backend.calls), 2)
        unit = self.units(manifest)[0]
        self.assertEqual(len(unit["generation"]["attempts"]), 1)
        self.assertEqual(unit["generation"]["selected_attempt_id"], "attempt_001")

    def test_substitution_insertion_and_cer_alone_do_not_retry(self):
        expected = self.good[self.ids[0]]
        noisy = "囧" + expected[1:] + "囧"
        manifest, backend = self.execute({**self.good, self.ids[0]: noisy})
        self.assertEqual(len(backend.calls), 2)
        unit = self.units(manifest)[0]
        attempt = unit["generation"]["attempts"][0]
        evidence = json.loads((self.run_dir / attempt["content_qc"]["evidence_path"])
                              .read_text(encoding="utf-8"))
        self.assertGreater(evidence["comparison"]["counts"]["substitution"], 0)
        self.assertGreater(evidence["comparison"]["counts"]["insertion"], 0)
        self.assertGreater(evidence["comparison"]["cer"], 0)
        self.assertEqual(evidence["comparison"]["flagged_deletion_groups"], [])
        self.assertEqual(unit["generation"]["selected_attempt_id"], "attempt_001")

    def test_resume_after_first_rejection_creates_next_logical_take(self):
        original = unit_execution._run_attempt
        calls = []

        def interrupted(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise KeyboardInterrupt("simulated stop after persisted QC rejection")
            return original(*args, **kwargs)

        with patch.object(unit_execution, "_run_attempt", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.execute(self.scripted(["attempt_001"]))
        persisted = json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8"))
        first = self.units(persisted)[0]["generation"]
        self.assertEqual(len(first["attempts"]), 1)
        self.assertEqual(first["attempts"][0]["content_qc"]["status"], "rejected")
        backend = self.backend()
        resumed = generate_units(self.run_dir, backend,
                                 worker_factory=self.factory(self.scripted(["attempt_001"])))
        current = self.units(resumed)[0]["generation"]
        self.assertEqual([item["take_index"] for item in current["attempts"]], [1, 2])
        self.assertEqual(current["selected_attempt_id"], "attempt_002")
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(len(self.workers), 2)

    def test_failed_physical_retry_reuses_same_logical_take_seed(self):
        class FailSecondSynthesis(FakeBackend):
            def generate_unit(self, text, path, seed):
                if len(self.calls) == 1:
                    self.calls.append((text, path, seed))
                    raise RuntimeError("scripted synthesis interruption")
                return super().generate_unit(text, path, seed)

        first_backend = FailSecondSynthesis(self.frontend_hash)
        manifest, _ = self.execute(self.scripted(["attempt_001"]), backend=first_backend)
        attempts = self.units(manifest)[0]["generation"]["attempts"]
        self.assertEqual([item["status"] for item in attempts], ["generated", "failed"])
        self.assertEqual([item["take_index"] for item in attempts], [1, 2])
        resume_backend = self.backend()
        resumed = generate_units(self.run_dir, resume_backend,
                                 worker_factory=self.factory(self.scripted(["attempt_001"])))
        current = self.units(resumed)[0]["generation"]
        self.assertEqual([item["take_index"] for item in current["attempts"]], [1, 2, 2])
        self.assertEqual(current["attempts"][1]["seed"], current["attempts"][2]["seed"])
        self.assertEqual(current["selected_attempt_id"], "attempt_003")

    def test_existing_qc_run_without_retry_policy_keeps_step_three_behavior(self):
        first, _ = self.execute(failures={(self.ids[0], "attempt_001")})
        first["generation"].pop("content_retry")
        first["unit_execution_policy"] = QC_EXECUTION_POLICY
        for unit in self.units(first):
            unit["generation"].pop("retry_state")
        self.save(first)
        backend = self.backend()
        resumed = generate_units(self.run_dir, backend,
                                 worker_factory=self.factory(self.scripted(["attempt_001"])))
        state = self.units(resumed)[0]["generation"]
        self.assertEqual(backend.calls, [])
        self.assertEqual(len(state["attempts"]), 1)
        self.assertEqual(state["attempts"][0]["content_qc"]["status"], "rejected")
        self.assertIsNone(state["selected_attempt_id"])

    def test_retry_policy_mismatch_and_rejected_evidence_corruption_fail_closed(self):
        manifest, _ = self.execute(self.scripted(["attempt_001", "attempt_002",
                                               "attempt_003"]))
        manifest["generation"]["content_retry"]["max_total_attempts_per_unit"] = 4
        self.save(manifest)
        with self.assertRaisesRegex(GenerationError, "content-retry policy differs"):
            generate_units(self.run_dir, self.backend(), worker_factory=self.factory())
        manifest["generation"].pop("content_retry")
        self.save(manifest)
        with self.assertRaisesRegex(GenerationError, "content-retry policy differs"):
            generate_units(self.run_dir, self.backend(), worker_factory=self.factory())
        manifest["generation"]["content_retry"] = dict(CONTENT_RETRY_POLICY)
        second = self.units(manifest)[0]["generation"]["attempts"][1]
        second["id"] = "attempt_004"
        self.save(manifest)
        with self.assertRaisesRegex(GenerationError, "gap in bounded attempt history"):
            generate_units(self.run_dir, self.backend(), worker_factory=self.factory())
        second["id"] = "attempt_002"
        first = self.units(manifest)[0]["generation"]["attempts"][0]
        (self.run_dir / first["content_qc"]["evidence_path"]).write_bytes(b"corrupt")
        self.save(manifest)
        with self.assertRaisesRegex(GenerationError, "evidence SHA-256 differs"):
            generate_units(self.run_dir, self.backend(), worker_factory=self.factory())


if __name__ == "__main__":
    unittest.main()
