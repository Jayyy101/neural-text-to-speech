"""Model-free boundary estimates on synthetic PCM16; no real audio fixtures."""

import contextlib
import io
import itertools
import json
from pathlib import Path
import struct
import tempfile
import unittest
import wave

from evaluation import estimate_period_boundaries as estimator


class EstimatePeriodBoundariesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.audio = self.root / "input.wav"
        self.metadata = self.root / "chunks.json"
        self.output = self.root / "proposal.json"
        self.samples = [3000] * 2000
        for start, end in ((430, 480), (900, 980), (1430, 1480), (1900, 1980)):
            self.samples[start:end] = [20] * (end - start)
        self.write_audio(self.samples)
        self.write_chunks([("甲。乙。", 1.0), ("丙。丁。", 1.0)])

    def write_audio(self, samples, channels=1, width=2):
        with wave.open(str(self.audio), "wb") as audio:
            audio.setparams((channels, width, 1000, 0, "NONE", "not compressed"))
            audio.writeframes(struct.pack(f"<{len(samples)}h", *samples) if width == 2
                              else bytes([128] * len(samples)))

    def write_chunks(self, chunks):
        self.metadata.write_text(json.dumps({
            "schema_version": 1,
            "chunks": [{"text": text, "duration_seconds": duration} for text, duration in chunks],
        }, ensure_ascii=False), encoding="utf-8")

    def estimate(self, **kwargs):
        return estimator.estimate_boundaries(self.audio, self.metadata, **kwargs)

    def test_character_rule_ignores_whitespace_but_counts_punctuation(self):
        self.assertEqual(list(estimator.punctuation_positions("甲，\n乙。 丙。")), [(4, 4 / 6), (7, 1)])
        self.assertEqual(list(estimator.punctuation_positions("\n \t")), [])
        self.assertEqual(list(estimator.punctuation_positions("甲？乙！丙。")), [(1, 2 / 6), (3, 4 / 6), (5, 1)])

    def test_multiple_periods_and_cumulative_original_coordinates(self):
        report = self.estimate(add_ms=999)
        periods = report["periods"]
        self.assertEqual([p["initial_seconds"] for p in periods], [0.5, 1.0, 1.5, 2.0])
        self.assertEqual([p["chunk_index"] for p in periods], [1, 1, 2, 2])
        self.assertEqual([p["period_index"] for p in periods], [1, 2, 3, 4])
        self.assertEqual([p["chunk_start_seconds"] for p in periods], [0, 0, 1, 1])
        self.assertEqual([p["chunk_end_seconds"] for p in periods], [1, 1, 2, 2])
        self.assertEqual(report["coordinate_system"], "original_input_wav_seconds")
        self.assertEqual(periods, self.estimate(add_ms=1)["periods"])

    def test_quiet_refinement_preserves_earlier_valley_preference_and_is_deterministic(self):
        samples = self.samples.copy()
        samples[430:480] = [14, -14] * 25
        samples[480:600] = [100, -100] * 60
        samples[600:650] = [10, -10] * 25
        self.write_audio(samples)
        report = self.estimate()
        period = report["periods"][0]
        self.assertEqual(period["selected_seconds"], 0.455)
        self.assertEqual(period["quiet_region_rms_pcm16"], 14)
        self.assertAlmostEqual(period["delta_seconds"], -0.045)
        self.assertEqual(report, self.estimate())

    def test_matching_cannot_borrow_candidates_from_other_chunks(self):
        # First chunk has no valleys; second has only one continuous silent region.
        self.write_audio([3000] * 1000 + [0] * 1000)
        report = self.estimate()
        self.assertEqual([chunk["candidate_count"] for chunk in report["chunks"]], [0, 1])
        self.assertTrue(all(b["status"] == "insufficient_candidates" for b in report["boundaries"]))
        self.assertEqual(report["pauses"], [])

    def test_stereo_energy_does_not_cancel(self):
        samples = [value for sample in self.samples for value in (sample, -sample)]
        self.write_audio(samples, channels=2)
        period = self.estimate()["periods"][0]
        self.assertEqual(period["selected_seconds"], 0.455)
        self.assertEqual(period["quiet_region_rms_pcm16"], 20)

    def test_cli_proposal_json_report_and_input_preservation(self):
        original_audio = self.audio.read_bytes()
        original_metadata = self.metadata.read_bytes()
        args = ["--input", str(self.audio), "--chunks", str(self.metadata)]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(estimator.main(args), 0)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["chunks.json", "input.wav"])
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(estimator.main(args + ["--output", str(self.output)]), 0)
        report = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertTrue(report["proposal_only"])
        self.assertEqual(report["pauses"], [
            {"around": p["selected_seconds"], "add_ms": 140.0, "label": "period", "source": "estimated"}
            for p in report["periods"]
        ])
        for text in ("PROPOSAL ONLY", "Punctuation 01", "chunk 02", "fraction", "expected", "matched", "delta", "RMS", "match cost"):
            self.assertIn(text, stdout.getvalue())
        self.assertEqual(self.audio.read_bytes(), original_audio)
        self.assertEqual(self.metadata.read_bytes(), original_metadata)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["chunks.json", "input.wav", "proposal.json"])

    def test_existing_files_cannot_be_overwritten(self):
        self.output.write_bytes(b"prior proposal")
        for output in (self.audio, self.metadata, self.output):
            original = output.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                estimator.main(["--input", str(self.audio), "--chunks", str(self.metadata), "--output", str(output)])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.read_bytes(), original)

    def test_help_and_report_support_redirected_windows_encoding(self):
        for args in (["--help"], ["--input", str(self.audio), "--chunks", str(self.metadata)]):
            buffer = io.BytesIO()
            with io.TextIOWrapper(buffer, encoding="cp1252") as stdout:
                with contextlib.redirect_stdout(stdout):
                    if args == ["--help"]:
                        with self.assertRaises(SystemExit) as error:
                            estimator.main(args)
                        self.assertEqual(error.exception.code, 0)
                    else:
                        self.assertEqual(estimator.main(args), 0)
                stdout.flush()
                self.assertIn("。", buffer.getvalue().decode("utf-8"))

    def test_no_periods_produces_empty_report(self):
        self.write_chunks([("没有句号！", 2.0)])
        report = self.estimate()
        self.assertEqual([b["punctuation"] for b in report["boundaries"]], ["！"])
        self.assertEqual(report["periods"], [])
        self.assertEqual(report["pauses"], [])
        self.write_chunks([("没有强标点", 2.0)])
        self.assertEqual(self.estimate()["boundaries"], [])

    def test_invalid_metadata_and_encoding(self):
        cases = [None, [], {}, {"schema_version": True, "chunks": []},
                 {"schema_version": 2, "chunks": []}]
        cases += [{"schema_version": 1, "chunks": chunks} for chunks in (
            [], "bad", [None], [{}], [{"text": "", "duration_seconds": 2}],
            [{"text": " \n", "duration_seconds": 2}], [{"text": 42, "duration_seconds": 2}],
        )]
        for case in cases:
            with self.subTest(case=case):
                self.metadata.write_text(json.dumps(case), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.estimate()
        for content in (b"{", b"\xff"):
            self.metadata.write_bytes(content)
            with self.assertRaisesRegex(ValueError, "Invalid UTF-8 JSON"):
                self.estimate()

    def test_invalid_durations_and_cli_numbers(self):
        for value in (None, True, "2", 0, -1, float("nan"), float("inf"), 10**400):
            self.write_chunks([("甲。", value)])
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Chunk 1 duration_seconds"):
                self.estimate()
        self.write_chunks([("甲。", 1e308), ("乙。", 1e308)])
        with self.assertRaisesRegex(ValueError, "Total chunk duration"):
            self.estimate()
        self.write_chunks([("甲。", 2)])
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.estimate(add_ms=value)

    def test_duration_mismatch_subframe_chunks_and_unsupported_audio(self):
        self.write_chunks([("甲。", 1.9)])
        with self.assertRaisesRegex(ValueError, "match the WAV duration"):
            self.estimate()
        self.write_chunks([("甲。", 0.001), ("乙。", 1.999)])
        with self.assertRaisesRegex(ValueError, "at least two audio frames"):
            self.estimate()
        self.write_chunks([("甲。", 2)])
        self.write_audio([128] * 2000, width=1)
        with self.assertRaisesRegex(ValueError, "PCM16"):
            self.estimate()

    def test_dp_is_joint_monotonic_and_does_not_reuse_candidates(self):
        candidates = [{"seconds": seconds, "quiet_cost": quality}
                      for seconds, quality in ((0.3, 0.2), (0.5, 0), (0.7, 0.1), (1.0, 0.1))]
        expected = [0.48, 0.51, 0.95]
        matches = estimator.match_monotonic(expected, candidates, 1.2)
        selected = [index for index, _ in matches]
        # Compare with exhaustive assignments, independently of the DP recurrence.
        assignments = list(itertools.combinations(range(len(candidates)), len(expected)))
        def cost(indices):
            return sum(((candidates[j]["seconds"] - t) / 0.4)**2 + candidates[j]["quiet_cost"]
                       for j, t in zip(indices, expected))
        optimum = min(assignments, key=cost)
        self.assertEqual(tuple(selected), optimum)
        self.assertEqual(selected, sorted(set(selected)))
        self.assertAlmostEqual(sum(c for _, c in matches), cost(optimum))
        self.assertEqual(matches, estimator.match_monotonic(expected, candidates, 1.2))
        self.assertIsNone(estimator.match_monotonic([0.2] * 5, candidates, 1.2))

    def test_soft_timing_prior_quality_cost_and_earlier_tie_break(self):
        candidates = [{"seconds": 0.25, "quiet_cost": 0}, {"seconds": 0.75, "quiet_cost": 0}]
        self.assertEqual(estimator.match_monotonic([0.5], candidates, 1)[0][0], 0)
        # A quieter candidate can beat an exact timing match, even >250 ms away.
        candidates = [{"seconds": 0.2, "quiet_cost": 0}, {"seconds": 0.8, "quiet_cost": 1}]
        self.assertEqual(estimator.match_monotonic([0.8], candidates, 1)[0][0], 0)

    def test_question_and_exclamation_are_distinct_matching_anchors(self):
        self.write_chunks([("甲。乙？丙！丁。", 2)])
        report = self.estimate()
        boundaries = report["boundaries"]
        self.assertEqual([b["punctuation"] for b in boundaries], list("。？！。"))
        self.assertEqual([b["selected_seconds"] for b in boundaries], [0.455, 0.94, 1.455, 1.94])
        self.assertEqual([p["around"] for p in report["pauses"]], [0.455, 1.94])

    def test_final_punctuation_can_precede_long_trailing_silence(self):
        self.write_audio([3000] * 1200 + [0] * 800)
        self.write_chunks([("末句。", 2)])
        period = self.estimate()["periods"][0]
        self.assertEqual(period["initial_seconds"], 2)
        self.assertAlmostEqual(period["selected_seconds"], 1.6, places=3)
        self.assertGreater(abs(period["delta_seconds"]), 0.25)

    def write_references(self, references):
        path = self.root / "references.json"
        path.write_text(json.dumps({"schema_version": 1, "references": references}, ensure_ascii=False), encoding="utf-8")
        return path

    def test_references_evaluate_errors_without_influencing_matching(self):
        report = self.estimate()
        original = json.dumps(report, ensure_ascii=False)
        reference_path = self.write_references([
            {"chunk_index": 1, "context": "甲。", "seconds": 0.4},
            {"chunk_index": 2, "context": "丁。", "seconds": 1.8},
        ])
        evaluation = estimator.evaluate_references(report, reference_path)
        self.assertAlmostEqual(evaluation["references"][0]["absolute_error_seconds"], 0.055)
        self.assertAlmostEqual(evaluation["mae_seconds"], (0.055 + 0.14) / 2)
        self.assertAlmostEqual(evaluation["max_error_seconds"], 0.14)
        self.assertEqual(evaluation["matched_count"], 2)
        self.assertEqual(json.dumps(report, ensure_ascii=False), original)
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            estimator.main(["--input", str(self.audio), "--chunks", str(self.metadata),
                            "--reference", str(reference_path), "--output", str(self.output)])
        saved = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(saved["boundaries"], report["boundaries"])
        self.assertEqual(saved["pauses"], report["pauses"])
        self.assertIn("MAE:", stdout.getvalue())
        # Altered reference targets change only evaluation, not the estimate.
        self.write_references([{"chunk_index": 1, "context": "甲。", "seconds": 0.9}])
        self.assertAlmostEqual(estimator.evaluate_references(report, reference_path)["mae_seconds"], 0.445)
        self.assertEqual(self.estimate(), report)

    def test_unmatched_references_are_reported_without_misleading_error_stats(self):
        self.write_audio([3000] * 2000)
        report = self.estimate()
        path = self.write_references([{"chunk_index": 1, "context": "甲。", "seconds": 0.4}])
        evaluation = estimator.evaluate_references(report, path)
        self.assertEqual(evaluation["matched_count"], 0)
        self.assertEqual(evaluation["reference_count"], 1)
        self.assertIsNone(evaluation["mae_seconds"])
        self.assertIsNone(evaluation["max_error_seconds"])

    def test_invalid_ambiguous_or_duplicate_references_fail_clearly(self):
        valid = {"chunk_index": 1, "context": "甲。", "seconds": 0.4}
        for entry in (None, {}, dict(valid, chunk_index=True), dict(valid, chunk_index=3),
                      dict(valid, context="missing。"), dict(valid, context="甲"),
                      dict(valid, seconds=True), dict(valid, seconds=float("nan")),
                      dict(valid, seconds=1.1)):
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                estimator.evaluate_references(self.estimate(), self.write_references([entry]))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            estimator.evaluate_references(self.estimate(), self.write_references([valid, valid]))
        self.write_chunks([("甲。甲。", 1), ("丙。丁。", 1)])
        with self.assertRaisesRegex(ValueError, "exactly once"):
            estimator.evaluate_references(self.estimate(), self.write_references([valid]))
        path = self.write_references([])
        with self.assertRaisesRegex(ValueError, "nonempty references"):
            estimator.evaluate_references(self.estimate(), path)
        path.write_bytes(b"\xff")
        with self.assertRaisesRegex(ValueError, "Invalid UTF-8 JSON reference"):
            estimator.evaluate_references(self.estimate(), path)


if __name__ == "__main__":
    unittest.main()
