"""Model-free production Mandarin content-QC and worker protocol tests."""

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave

from evaluation.bounded_qc_research import evaluate_candidates
from src.audiobook import asr_worker
from src.audiobook.content_qc import (
    ASRWorkerClient, MODEL_ID, MODEL_REVISION, audio_request, compare_recognition,
    han_tokens, policy_record,
)
from src.audiobook.manifest import create_planning_run
from src.audiobook.pipeline import GenerationError
from src.audiobook.unit_execution import OVERLENGTH_RETRY_POLICY, assemble_units, generate_units
from src.audiobook.unit_planning import prepare_synthesis_unit_run
from tests.test_audiobook_unit_planning import FixtureFrontend


def recognition(text, request_id="test", wav_hash="a" * 64):
    tokens = [{"comparison_token": character, "start_seconds": index * 0.1,
               "end_seconds": (index + 1) * 0.1}
              for index, character in enumerate(text)]
    return {"type": "recognized", "request_id": request_id,
            "wav_sha256": wav_hash, "raw_transcript": text,
            "comparison_tokens": tokens, "comparison_text": text,
            "raw_emitted_tokens": [], "ignored_tokens": []}


class ContentComparisonTests(unittest.TestCase):
    def test_only_validated_qc_policy_is_constructed(self):
        self.assertEqual(policy_record()["policy"],
                         "mandarin_asr_contiguous_han_deletion_v1")

    def test_evaluation_only_mixed_edits_and_endpoint_energy_abstain(self):
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "unit.wav"
            with wave.open(str(wav_path), "wb") as wav:
                wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                wav.writeframes(b"\xff\x7f" * 480)
            result = evaluate_candidates("甲乙丙丁", recognition("庚辛乙丙丁"), wav_path)
            self.assertEqual(result["decision"], "passed")
            self.assertEqual(result["rejection_reasons"], [])
            self.assertTrue(result["measurements"]["local_mixed_edit_clusters"])
            self.assertEqual({item["reason"] for item in result["abstentions"]}, {
                "corroborated_local_content_corruption",
                "corroborated_endpoint_truncation",
            })
            self.assertEqual(result["measurements"]["endpoint"]["trailing_quiet_ms_within_50ms"], 0)

    def test_validated_greedy_ctc_collapse_and_unk_behavior(self):
        try:
            from evaluation.run_mandarin_asr_unit09_feasibility import (
                collapse_greedy_ids, normalize_recognized,
            )
        except ModuleNotFoundError as error:
            self.skipTest(f"requires tts-align dependencies: {error}")
        emitted = collapse_greedy_ids([0, 1, 1, 0, 2, 2, 0, 3], 0)
        self.assertEqual([item["token_id"] for item in emitted], [1, 2, 3])
        tokenizer = Mock(
            word_delimiter_token="|", unk_token="<unk>",
            all_special_tokens=["<unk>", "<pad>"],
        )
        tokenizer.convert_ids_to_tokens.side_effect = {
            1: "甲", 2: "<unk>", 3: "乙",
        }.get
        raw, comparison, ignored = normalize_recognized(
            emitted, tokenizer, 0.02
        )
        self.assertEqual(raw, "甲<unk>乙")
        self.assertEqual([item["comparison_token"] for item in comparison],
                         ["甲", "<unk>", "乙"])
        self.assertEqual(ignored, [])

    def test_nfc_han_only_and_unknown_preservation(self):
        self.assertEqual(han_tokens("A甲，乙〇\n"), ["甲", "乙", "〇"])
        unknown = recognition("甲")
        unknown["comparison_tokens"].append({
            "comparison_token": "<unk>", "start_seconds": 0.1,
            "end_seconds": 0.2,
        })
        unknown["comparison_text"] = "甲<unk>"
        result = compare_recognition("甲乙", unknown)
        self.assertEqual(result["recognized_comparison_text"], "甲<unk>")
        self.assertEqual(result["recognized_han_count"], 1)
        self.assertEqual(result["recognized_comparison_count"], 2)
        self.assertEqual(result["decision"], "passed")

    def test_fixed_three_vs_four_deletions_and_other_edits(self):
        expected = "甲乙丙丁戊己"
        three = compare_recognition(expected, recognition("甲戊己"))
        four = compare_recognition(expected, recognition("甲己"))
        self.assertEqual(three["decision"], "passed")
        self.assertEqual(three["counts"]["deletion"], 3)
        self.assertEqual(four["decision"], "rejected")
        self.assertEqual(four["flagged_deletion_groups"][0]["expected_text"], "乙丙丁戊")
        self.assertEqual(compare_recognition(expected, recognition(expected + "庚"))["decision"],
                         "passed")
        substituted = compare_recognition(expected, recognition("庚辛壬癸子丑"))
        self.assertEqual(substituted["decision"], "passed")
        self.assertEqual(substituted["cer"], 1.0)
        self.assertEqual(substituted["counts"]["substitution"], 6)
        self.assertEqual(compare_recognition(expected, recognition("甲乙寅丁戊己"))[
            "decision"], "passed")

    def test_deterministic_alignment_grouping(self):
        first = compare_recognition("甲乙丙丁戊己", recognition("甲己"))
        second = compare_recognition("甲乙丙丁戊己", recognition("甲己"))
        self.assertEqual(first, second)
        self.assertEqual(first["edit_groups"][0]["expected_start"], 1)
        self.assertEqual(first["edit_groups"][0]["expected_end"], 5)

    def test_production_alignment_matches_validated_evaluation_steps(self):
        try:
            from evaluation.run_mandarin_asr_unit09_feasibility import (
                group_edits as evaluated_groups,
                levenshtein_steps as evaluated_steps,
            )
        except ModuleNotFoundError as error:
            self.skipTest(f"requires tts-align dependencies: {error}")
        from src.audiobook.content_qc import group_edits, levenshtein_steps
        for expected, tokens in (
            ("甲乙丙丁戊己", list("甲己")),
            ("甲乙丙", ["甲", "<unk>", "丙"]),
            ("甲乙", list("甲乙丙")),
            ("甲乙丙", list("庚辛壬")),
        ):
            with self.subTest(expected=expected, observed=tokens):
                # Use the same metadata shape as the validated CTC decoder.
                meta = [{"comparison_token": token, "start_seconds": index * 0.1,
                         "end_seconds": (index + 1) * 0.1}
                        for index, token in enumerate(tokens)]
                ours, distance = levenshtein_steps(list(expected), tokens)
                prior, prior_distance = evaluated_steps(list(expected), tokens)
                self.assertEqual((ours, distance), (prior, prior_distance))
                ours_groups = group_edits(ours, list(expected), meta)
                prior_groups = evaluated_groups(prior, list(expected), meta, 0, 0)
                for group in prior_groups:
                    group.pop("target_overlap_characters")
                self.assertEqual(ours_groups, prior_groups)

    def test_audio_only_request_has_no_intended_fields(self):
        request = audio_request("audio.wav", "a" * 64, "unit:attempt")
        self.assertEqual(set(request), {"type", "request_id", "audio_path", "wav_sha256"})
        self.assertFalse(any("text" in key or "expected" in key for key in request))
        client = ASRWorkerClient("unused", "unused.log")
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            client.recognize({**request, "intended_text": "甲"})
        self.assertIsNone(client.process)


class FakeBackend:
    def __init__(self, frontend_hash, frames=240):
        self.frontend_hash = frontend_hash
        self.frames = frames
        self.calls = []
        self.initialize_calls = 0

    def configuration(self):
        return {"backend": "fake_cosyvoice", "settings": {"stream": False}}

    def initialize_units(self):
        self.initialize_calls += 1
        return {**self.configuration(), "sample_rate_hz": 24000,
                "frontend_identity_sha256": self.frontend_hash}

    def generate_unit(self, text, path, seed):
        self.calls.append((text, path, seed))
        with wave.open(str(path), "wb") as wav:
            wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            wav.writeframes(b"\x01\x00" * self.frames)
        return {"frontend_bypass": True, "cosyvoice_chunks": 1}


class ScriptedWorker:
    def __init__(self, texts, fail_ids=()):
        self.texts = texts
        self.fail_ids = set(fail_ids)
        self.requests = []
        self.model = {"model_id": MODEL_ID, "resolved_revision": MODEL_REVISION}
        self.close_calls = 0

    def recognize(self, request):
        self.requests.append(request)
        if request["request_id"].split(":")[0] in self.fail_ids:
            raise RuntimeError("fake ASR transport error")
        unit_id = request["request_id"].split(":")[0]
        return recognition(self.texts[unit_id], request["request_id"],
                           request["wav_sha256"])

    def close(self):
        self.close_calls += 1


class UnitQCTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        source = root / "source.txt"
        source.write_text("甲乙丙丁戊己。庚辛壬癸。", encoding="utf-8")
        self.run_dir, _ = create_planning_run(source, "chapter", "run", root / "out")
        prepared = prepare_synthesis_unit_run(self.run_dir, FixtureFrontend())
        self.units = prepared["scenes"][0]["synthesis_units"]
        self.frontend_hash = prepared["synthesis_unit_plan"]["frontend_identity_sha256"]
        self.recognized = {
            unit["id"]: "".join(han_tokens(unit["normalized_text"]))
            for unit in self.units
        }
        self.workers = []

    def backend(self):
        return FakeBackend(self.frontend_hash)

    def factory(self, texts=None, fail_ids=()):
        def create(_python, _log):
            worker = ScriptedWorker(texts or self.recognized, fail_ids)
            self.workers.append(worker)
            return worker
        return create

    def generate(self, backend=None, factory=None):
        backend = backend or self.backend()
        manifest = generate_units(self.run_dir, backend, root_seed=19,
                                  worker_factory=factory or self.factory())
        return manifest, backend

    def latest(self, manifest, index=0):
        return manifest["scenes"][0]["synthesis_units"][index]["generation"]["attempts"][-1]

    def save(self, manifest):
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )

    def test_qc_pass_selects_and_one_worker_handles_both_units(self):
        manifest, backend = self.generate()
        self.assertEqual(manifest["status"], "generated")
        self.assertEqual(backend.initialize_calls, 1)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(len(self.workers), 1)
        self.assertEqual(len(self.workers[0].requests), 2)
        self.assertEqual(self.workers[0].close_calls, 1)
        for request in self.workers[0].requests:
            self.assertEqual(set(request), {"type", "request_id", "audio_path", "wav_sha256"})
        for unit in manifest["scenes"][0]["synthesis_units"]:
            state = unit["generation"]
            attempt = state["attempts"][0]
            self.assertEqual(state["selected_attempt_id"], "attempt_001")
            self.assertEqual(attempt["content_qc"]["status"], "passed")
            evidence = json.loads((self.run_dir / attempt["content_qc"]["evidence_path"])
                                  .read_text(encoding="utf-8"))
            self.assertEqual(evidence["binding"]["normalized_text_sha256"],
                             unit["normalized_text_sha256"])
            self.assertEqual(evidence["policy"], policy_record())
            self.assertEqual(evidence["comparison"]["decision"], "passed")
            self.assertEqual(evidence["schema_version"], 1)
            self.assertNotIn("measurements", evidence)
        self.assertEqual(assemble_units(self.run_dir)["assembly"]["status"], "assembled")
        resume_backend = self.backend()
        resumed = generate_units(self.run_dir, resume_backend,
                                 worker_factory=self.factory())
        self.assertEqual(resume_backend.calls, [])
        self.assertEqual(resume_backend.initialize_calls, 0)
        self.assertEqual(len(self.workers), 1)
        self.assertEqual([len(unit["generation"]["attempts"])
                          for unit in resumed["scenes"][0]["synthesis_units"]], [1, 1])

    def test_historical_policy_sidecars_validate_without_v2_reinterpretation(self):
        manifest, _ = self.generate()
        self.assertEqual(manifest["generation"]["content_qc"], policy_record())
        self.assertEqual(manifest["generation"]["content_retry"], OVERLENGTH_RETRY_POLICY)
        self.assertEqual(assemble_units(self.run_dir)["assembly"]["status"], "assembled")
        backend = self.backend()
        resumed = generate_units(self.run_dir, backend, worker_factory=self.factory())
        self.assertEqual(backend.calls, [])
        self.assertEqual(resumed["generation"]["content_qc"], policy_record())

    def test_unpromoted_candidate_policy_is_rejected_before_retry(self):
        manifest, _ = self.generate()
        manifest["generation"]["content_qc"]["policy"] = "mandarin_asr_bounded_evidence_v2"
        self.save(manifest)
        backend = self.backend()
        with self.assertRaisesRegex(GenerationError, "content-QC policy"):
            generate_units(self.run_dir, backend, worker_factory=self.factory())
        self.assertEqual(backend.calls, [])

    def test_reject_keeps_all_wavs_and_exhausted_resume_never_synthesizes_again(self):
        texts = {**self.recognized, self.units[0]["id"]: "甲己"}
        manifest, _ = self.generate(factory=self.factory(texts))
        rejected = self.latest(manifest)
        self.assertEqual(manifest["status"], "generation_failed")
        self.assertEqual(rejected["status"], "generated")
        self.assertEqual(rejected["content_qc"]["status"], "rejected")
        evidence = json.loads((self.run_dir / rejected["content_qc"]["evidence_path"])
                              .read_text(encoding="utf-8"))
        self.assertEqual(evidence["comparison"]["flagged_deletion_groups"][0][
            "expected_text"], "乙丙丁戊")
        self.assertEqual(evidence["policy"][
            "contiguous_expected_han_deletion_threshold"], 4)
        self.assertIsNone(manifest["scenes"][0]["synthesis_units"][0]["generation"][
            "selected_attempt_id"])
        self.assertTrue((self.run_dir / rejected["output_path"]).is_file())
        with self.assertRaises(GenerationError):
            assemble_units(self.run_dir)
        backend = self.backend()
        resumed = generate_units(self.run_dir, backend, worker_factory=self.factory())
        self.assertEqual(backend.calls, [])
        self.assertEqual(len(self.workers), 1)
        self.assertEqual(len(self.latest(resumed)["content_qc"]["evidence_sha256"]), 64)
        state = resumed["scenes"][0]["synthesis_units"][0]["generation"]
        self.assertEqual(len(state["attempts"]), 3)
        self.assertEqual(state["retry_state"]["status"], "exhausted")
        self.assertTrue(all((self.run_dir / attempt["output_path"]).is_file()
                            for attempt in state["attempts"]))

    def test_asr_error_keeps_wav_then_qc_only_resume(self):
        manifest, _ = self.generate(factory=self.factory(fail_ids={self.units[0]["id"]}))
        errored = self.latest(manifest)
        self.assertEqual(errored["content_qc"]["status"], "error")
        self.assertTrue((self.run_dir / errored["output_path"]).is_file())
        self.assertIsNone(manifest["scenes"][0]["synthesis_units"][0]["generation"][
            "selected_attempt_id"])
        backend = self.backend()
        resumed = generate_units(self.run_dir, backend, worker_factory=self.factory())
        self.assertEqual(backend.calls, [])
        self.assertEqual(backend.initialize_calls, 0)
        self.assertEqual(self.latest(resumed)["content_qc"]["status"], "passed")
        self.assertEqual(len(resumed["scenes"][0]["synthesis_units"][0]["generation"][
            "attempts"]), 1)

    def _check_qc_resume_state(self, status):
        manifest, _ = self.generate()
        unit = manifest["scenes"][0]["synthesis_units"][0]
        attempt = unit["generation"]["attempts"][0]
        (self.run_dir / attempt["content_qc"]["evidence_path"]).unlink()
        attempt["content_qc"] = {"status": status}
        unit["generation"]["selected_attempt_id"] = None
        self.save(manifest)
        backend = self.backend()
        resumed = generate_units(self.run_dir, backend,
                                 worker_factory=self.factory())
        self.assertEqual(backend.calls, [])
        self.assertEqual(self.latest(resumed)["content_qc"]["status"], "passed")
        self.assertEqual(len(self.latest(resumed)["wav_sha256"]), 64)

    def test_pending_qc_resume_existing_wav(self):
        self._check_qc_resume_state("pending")

    def test_running_qc_resume_existing_wav(self):
        self._check_qc_resume_state("running")

    def test_config_mismatch_and_corrupt_sidecar_fail_before_model(self):
        manifest, _ = self.generate()
        backend = self.backend()
        with self.assertRaisesRegex(GenerationError, "configuration differs"):
            generate_units(self.run_dir, backend, asr_python="/different/python",
                           worker_factory=self.factory())
        self.assertEqual(backend.initialize_calls, 0)
        sidecar = self.run_dir / self.latest(manifest)["content_qc"]["evidence_path"]
        sidecar.write_bytes(b"corrupt")
        with self.assertRaisesRegex(GenerationError, "SHA-256 differs"):
            generate_units(self.run_dir, backend, worker_factory=self.factory())
        with self.assertRaisesRegex(GenerationError, "SHA-256 differs"):
            assemble_units(self.run_dir)

    def test_missing_or_mismatched_passed_evidence_fails_closed(self):
        manifest, _ = self.generate()
        attempt = self.latest(manifest)
        sidecar = self.run_dir / attempt["content_qc"]["evidence_path"]
        original = sidecar.read_bytes()
        sidecar.unlink()
        with self.assertRaisesRegex(GenerationError, "evidence is missing"):
            assemble_units(self.run_dir)
        sidecar.write_bytes(original)
        evidence = json.loads(original)
        evidence["binding"]["unit_id"] = "different_unit"
        sidecar.write_text(json.dumps(evidence, ensure_ascii=False), encoding="utf-8")
        manifest = json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.latest(manifest)["content_qc"]["evidence_sha256"] = hashlib.sha256(
            sidecar.read_bytes()
        ).hexdigest()
        self.save(manifest)
        with self.assertRaisesRegex(GenerationError, "binding or policy differs"):
            generate_units(self.run_dir, self.backend(), worker_factory=self.factory())

    def test_over_validated_whole_wav_limit_exhausts_without_asr(self):
        backend = FakeBackend(self.frontend_hash, frames=24000 * 31)
        manifest, _ = self.generate(backend=backend)
        self.assertEqual(len(backend.calls), 6)
        self.assertEqual(len(self.workers), 0)
        self.assertEqual(manifest["status"], "generation_failed")
        for unit in manifest["scenes"][0]["synthesis_units"]:
            self.assertEqual(unit["generation"]["retry_state"]["status"], "exhausted")
            for attempt in unit["generation"]["attempts"]:
                self.assertEqual(attempt["status"], "generated")
                self.assertEqual(attempt["content_qc"]["status"], "rejected")
                evidence = json.loads((self.run_dir / attempt["content_qc"][
                    "evidence_path"]).read_text(encoding="utf-8"))
                self.assertEqual(evidence["rejection_reason"],
                                 "wav_exceeds_qc_duration_limit")
                self.assertIs(evidence["asr_performed"], False)
        resume_backend = self.backend()
        resumed = generate_units(self.run_dir, resume_backend,
                                 worker_factory=self.factory())
        self.assertEqual(resume_backend.calls, [])
        self.assertEqual(len(self.workers), 0)
        self.assertEqual([len(unit["generation"]["attempts"])
                          for unit in resumed["scenes"][0]["synthesis_units"]], [3, 3])


class WorkerLifecycleTests(unittest.TestCase):
    def test_one_model_load_serves_two_audio_only_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            wav = Path(temporary) / "audio.wav"
            wav.write_bytes(b"fixture audio identity")
            digest = hashlib.sha256(wav.read_bytes()).hexdigest()
            requests = [audio_request(wav, digest, f"request_{i}") for i in (1, 2)]
            stdin = io.StringIO("".join(json.dumps(item) + "\n" for item in requests))
            stdout = io.StringIO()
            model = {"model_id": MODEL_ID, "resolved_revision": MODEL_REVISION}
            fake_result = {"raw_transcript": "甲", "comparison_tokens": [],
                           "comparison_text": "", "raw_emitted_tokens": [],
                           "ignored_tokens": [], "audio": {},
                           "inference_seconds": 0.1, "emission_frames": 2,
                           "ctc_frame_seconds": 0.02}
            with (patch.object(asr_worker, "_load_model",
                               return_value=(Mock(), Mock(), Mock(), Mock(), model)) as load,
                  patch.object(asr_worker, "_infer", return_value=[fake_result]) as infer,
                  patch.object(asr_worker.sys, "stdin", stdin),
                  patch.object(asr_worker.sys, "stdout", stdout)):
                asr_worker.run()
            messages = [json.loads(line) for line in stdout.getvalue().splitlines()]
            self.assertEqual([item["type"] for item in messages],
                             ["ready", "recognized", "recognized"])
            load.assert_called_once()
            self.assertEqual(infer.call_count, 2)


if __name__ == "__main__":
    unittest.main()
