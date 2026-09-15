"""Known-transcript alignment tests: generated audio/emissions, no model downloads."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from evaluation import align_mandarin_ctc as aligner


class FakeTokenizer:
    pad_token_id = 0
    unk_token_id = 1
    all_special_ids = [0, 1]
    init_kwargs = {"_commit_hash": "synthetic-tokenizer-revision"}

    def __init__(self, vocab=None):
        self.vocab = vocab or {"<pad>": 0, "<unk>": 1, "开": 2, "门": 3, "这": 4}

    def get_vocab(self):
        return self.vocab

    def encode(self, character, add_special_tokens=False):
        return [self.vocab.get(character, self.unk_token_id)]


class TextMappingTests(unittest.TestCase):
    def test_nested_set_metadata_is_json_safe_and_deterministic(self):
        first = {"model": {"labels": {"乙", "甲"}, "nested": [{"ids": {3, 1, 2}}]}}
        second = {"model": {"labels": {"甲", "乙"}, "nested": [{"ids": {2, 3, 1}}]}}
        first_text = aligner.json_text(first)
        self.assertEqual(first_text, aligner.json_text(second))
        self.assertEqual(
            json.loads(first_text),
            {"model": {"labels": ["乙", "甲"], "nested": [{"ids": [1, 2, 3]}]}},
        )

    def test_han_extraction_offsets_and_whitespace_are_preserved(self):
        source = " \n开门。\r\n“这？”\t"
        mapping = aligner.map_source(source)
        self.assertEqual(mapping["exact_source_text"], source)
        self.assertEqual(mapping["acoustic_text"], "开门这")
        self.assertEqual([t["source_index"] for t in mapping["tokens"]], [2, 3, 8])
        self.assertEqual(mapping["source_to_token"], [None, None, 0, 1, None, None, None, None, 2, None, None, None])
        self.assertEqual(mapping["normalization_failures"], [])
        self.assertEqual([p["character"] for p in mapping["punctuation"]], list("。“？”"))

    def test_period_and_quote_share_the_correct_neighbors(self):
        mapping = aligner.map_source("打开北门。“这是什么意思？”")
        marks = mapping["punctuation"]
        for mark in marks[:2]:
            self.assertEqual(mark["preceding_token_index"], 3)
            self.assertEqual(mark["following_token_index"], 4)
        self.assertEqual(mapping["tokens"][3]["character"], "门")
        self.assertEqual(mapping["tokens"][4]["character"], "这")
        self.assertIsNone(marks[-1]["following_token_index"])

    def test_punctuation_allowlist_and_leading_marks(self):
        source = "“开，门、这：开；门……这——开（门）！”"
        mapping = aligner.map_source(source)
        self.assertEqual(mapping["normalization_failures"], [])
        self.assertEqual(mapping["acoustic_text"], "开门这开门这开门")
        self.assertIsNone(mapping["punctuation"][0]["preceding_token_index"])

    def test_english_digits_percent_symbols_and_variation_selectors_are_not_dropped(self):
        for source in ("开Python门", "2026年", "100%", "２０２６", "开😀门", "门\ufe00", "\ufeff开门"):
            with self.subTest(source=source):
                result = aligner.map_source(source)
                self.assertTrue(result["normalization_failures"])
                for failure in result["normalization_failures"]:
                    self.assertEqual(source[failure["source_index"]], failure["character"])

    def test_extended_han_reaches_vocabulary_validation(self):
        mapping = aligner.map_source("𠀀〇")
        self.assertEqual(mapping["acoustic_text"], "𠀀〇")
        self.assertEqual(mapping["normalization_failures"], [])
        with self.assertRaisesRegex(ValueError, "Unsupported source characters"):
            aligner.validate_vocabulary(mapping, FakeTokenizer())
        self.assertEqual([t["character"] for t in mapping["unsupported_characters"]], ["𠀀", "〇"])

    def test_vocabulary_validation_retains_every_source_occurrence(self):
        mapping = aligner.map_source("开罕门罕")
        with self.assertRaisesRegex(ValueError, "罕.*source index 1.*罕.*source index 3"):
            aligner.validate_vocabulary(mapping, FakeTokenizer())
        self.assertEqual(mapping["acoustic_text"], "开罕门罕")
        self.assertEqual(len(mapping["unsupported_characters"]), 2)

    def test_tokenizer_must_preserve_identity_without_unknown_or_extra_tokens(self):
        tokenizer = FakeTokenizer()
        with patch.object(tokenizer, "encode", return_value=[1]):
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                aligner.validate_vocabulary(aligner.map_source("开"), tokenizer)
        with patch.object(tokenizer, "encode", return_value=[2, 0]):
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                aligner.validate_vocabulary(aligner.map_source("开"), tokenizer)

    def test_dense_vocabulary_metadata_is_deterministic(self):
        tokenizer = FakeTokenizer()
        mapping = aligner.map_source("开门这")
        ids, labels, metadata = aligner.validate_vocabulary(mapping, tokenizer)
        self.assertEqual(ids, [2, 3, 4])
        self.assertEqual(labels, ["<pad>", "<unk>", "开", "门", "这"])
        self.assertEqual(metadata["blank_token_id"], 0)
        self.assertEqual(metadata, aligner.validate_vocabulary(aligner.map_source("开门这"), tokenizer)[2])
        for vocab in ({"<pad>": 0, "开": 2}, {"<pad>": 0, "开": 0}):
            with self.assertRaisesRegex(ValueError, "unique contiguous"):
                aligner.validate_vocabulary(mapping, FakeTokenizer(vocab))


class AudioAndRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.audio = self.root / "input.wav"
        self.text = self.root / "input.txt"
        self.output = self.root / "evidence.json"
        self.text.write_bytes("开门。\r\n“这？”".encode("utf-8"))
        rate = 24000
        samples = np.sin(2 * np.pi * 200 * np.arange(rate * 2) / rate) * 0.25
        sf.write(self.audio, samples, rate, subtype="PCM_16")
        # Prevent any test, including erroneous paths, from loading a real model.
        patches = contextlib.ExitStack()
        self.addCleanup(patches.close)
        self.tokenizer_loader = patches.enter_context(patch.object(aligner, "load_tokenizer", return_value=FakeTokenizer()))
        self.inference = patches.enter_context(patch.object(aligner, "infer_emissions", side_effect=AssertionError("ML inference forbidden in tests")))

    def test_slice_resample_and_source_file_unchanged(self):
        original = self.audio.read_bytes()
        samples, metadata = aligner.read_analysis_audio(self.audio, 0.4, 1.4)
        self.assertEqual(samples.shape, (16000,))
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(metadata["original_sample_rate"], 24000)
        self.assertEqual(metadata["analysis_sample_rate"], 16000)
        self.assertEqual(metadata["start_frame"], 9600)
        self.assertEqual(metadata["end_frame_exclusive"], 33600)
        self.assertEqual(metadata["start_seconds"], 0.4)
        self.assertEqual(self.audio.read_bytes(), original)

    def test_stereo_downmix_is_deterministic_and_no_resample_needed_at_16k(self):
        stereo = np.column_stack((np.full(16000, 0.2), np.full(16000, 0.4)))
        sf.write(self.audio, stereo, 16000, subtype="FLOAT")
        samples, metadata = aligner.read_analysis_audio(self.audio)
        self.assertTrue(np.allclose(samples, 0.3))
        self.assertEqual(metadata["channels"], 2)
        self.assertTrue(np.array_equal(samples, aligner.read_analysis_audio(self.audio)[0]))

    def test_invalid_spans_fail(self):
        for start, end in ((-1, 1), (1, 1), (1, 0), (0, 3), (float("nan"), 1),
                           (0, float("inf")), (0, 0.001), (0, 0.000001)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                aligner.read_analysis_audio(self.audio, start, end)

    def test_empty_silent_nonfinite_and_multichannel_audio_rejected(self):
        cases = [np.zeros(0), np.zeros(16000), np.full(16000, np.nan),
                 np.ones((16000, 3)), np.column_stack((np.ones(16000), -np.ones(16000)))]
        for samples in cases:
            with self.subTest(shape=samples.shape):
                sf.write(self.audio, samples, 16000, subtype="FLOAT")
                with self.assertRaises(ValueError):
                    aligner.read_analysis_audio(self.audio)

    def test_punctuation_times_use_tokens_and_both_coordinate_systems(self):
        mapping = aligner.map_source("开门。“这？”")
        segments = [{"start": 0.1, "end": 0.2, "ctc_anchor": 0.15, "score": -0.1},
                    {"start": 0.25, "end": 0.4, "ctc_anchor": 0.3, "score": -0.2},
                    {"start": 0.55, "end": 0.7, "ctc_anchor": 0.6, "score": -0.3}]
        tokens, marks = aligner.attach_timings(mapping, segments, {"start_seconds": 20.0, "duration_seconds": 1.0})
        self.assertEqual(tokens[1]["source_end_seconds"], 20.4)
        self.assertEqual(marks[0]["preceding_token"]["character"], "门")
        self.assertEqual(marks[0]["following_token"]["character"], "这")
        self.assertEqual(marks[0]["interval_local_seconds"], [0.4, 0.55])
        self.assertEqual(marks[0]["interval_source_seconds"], [20.4, 20.55])
        self.assertAlmostEqual(marks[0]["estimated_gap_seconds"], 0.15)
        self.assertIsNone(marks[-1]["following_token_start_seconds"])
        self.assertIsNone(marks[-1]["interval_source_seconds"])

    def test_invalid_alignment_spans_fail_without_fabricated_timings(self):
        mapping = aligner.map_source("开门")
        good = {"start": 0.1, "end": 0.2, "ctc_anchor": 0.15, "score": -0.1}
        cases = [[], [good], [good, good], [dict(good, start=-1), good],
                 [good, dict(good, start=0.3, end=2)],
                 [good, dict(good, start=0.3, end=0.4, score=float("nan"))]]
        for segments in cases:
            with self.subTest(segments=segments), self.assertRaises(ValueError):
                aligner.attach_timings(mapping, segments, {"start_seconds": 0, "duration_seconds": 1})

    def test_unsupported_text_never_loads_tokenizer_or_model(self):
        for source in ("开English门", "开2026门", "开100%门", " \r\n。"):
            self.text.write_text(source, encoding="utf-8")
            report = aligner.run_alignment(self.audio, self.text)
            self.assertEqual(report["error"]["stage"], "text_validation")
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["text_mapping"]["exact_source_text"], source)
            self.assertEqual(report["token_timings"], [])
        self.tokenizer_loader.assert_not_called()
        self.inference.assert_not_called()

    def test_unknown_han_never_runs_inference_and_failure_json_is_written(self):
        self.text.write_text("开罕门", encoding="utf-8")
        args = ["--input", str(self.audio), "--text-file", str(self.text), "--output", str(self.output)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(aligner.main(args), 1)
        report = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(report["error"]["stage"], "vocabulary_validation")
        self.assertEqual(report["text_mapping"]["unsupported_characters"][0]["character"], "罕")
        self.assertIn("罕", stderr.getvalue())
        self.inference.assert_not_called()

    def test_mocked_pipeline_emits_deterministic_evidence_and_preserves_input(self):
        original_audio = self.audio.read_bytes()
        original_text = self.text.read_bytes()
        self.inference.side_effect = None
        self.inference.return_value = (np.zeros((100, 5)), 0.02, {"resolved_revision": "fake-model-commit"})
        segments = [{"start": 0.1 + i * 0.2, "end": 0.2 + i * 0.2,
                     "ctc_anchor": 0.15 + i * 0.2, "score": -0.1} for i in range(3)]
        with patch.object(aligner, "align_emissions", return_value=segments) as segmentation:
            report = aligner.run_alignment(self.audio, self.text, start_seconds=0.4, end_seconds=1.4)
            again = aligner.run_alignment(self.audio, self.text, start_seconds=0.4, end_seconds=1.4)
        self.assertEqual(report["status"], "aligned_listening_pending")
        self.assertEqual(aligner.json_text(report), aligner.json_text(again))
        self.assertEqual(report["text_mapping"]["exact_source_text"], original_text.decode("utf-8"))
        self.assertEqual(segmentation.call_args.args[1], [2, 3, 4])
        self.assertEqual(report["token_timings"][0]["source_start_seconds"], 0.5)
        self.assertEqual(self.audio.read_bytes(), original_audio)
        self.assertEqual(self.text.read_bytes(), original_text)

    def test_existing_output_and_input_paths_are_never_overwritten(self):
        self.output.write_bytes(b"prior evidence")
        for output in (self.audio, self.text, self.output):
            original = output.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                aligner.main(["--input", str(self.audio), "--text-file", str(self.text), "--output", str(output)])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.read_bytes(), original)
        self.tokenizer_loader.assert_not_called()
        self.inference.assert_not_called()

    def test_invalid_utf8_and_missing_input_are_evidence_failures(self):
        self.text.write_bytes(b"\xff")
        report = aligner.run_alignment(self.audio, self.text)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"]["type"], "UnicodeDecodeError")
        report = aligner.run_alignment(self.root / "missing.wav", self.text)
        self.assertEqual(report["error"]["type"], "FileNotFoundError")
        self.inference.assert_not_called()

    def test_help_is_model_free(self):
        with contextlib.redirect_stdout(io.StringIO()) as stdout, self.assertRaises(SystemExit) as error:
            aligner.main(["--help"])
        self.assertEqual(error.exception.code, 0)
        self.assertIn("--text-file", stdout.getvalue())
        self.tokenizer_loader.assert_not_called()
        self.inference.assert_not_called()


class SyntheticCtcTests(unittest.TestCase):
    def test_actual_ctc_library_aligns_supplied_tokens_without_asr_or_model(self):
        # Hand-authored probabilities, not outputs of ML inference.
        labels = ["<pad>", "开", "门"]
        probabilities = np.full((60, 3), 0.005)
        probabilities[:, 0] = 0.99
        probabilities[10:15] = [0.005, 0.99, 0.005]
        probabilities[35:40] = [0.005, 0.005, 0.99]
        segments = aligner.align_emissions(np.log(probabilities), [1, 2], labels, 0, 0.02)
        self.assertEqual(len(segments), 2)
        self.assertTrue(0.18 <= segments[0]["ctc_anchor"] <= 0.3)
        self.assertTrue(0.68 <= segments[1]["ctc_anchor"] <= 0.8)
        self.assertTrue(segments[0]["end"] <= segments[1]["start"])
        self.assertTrue(all(np.isfinite(segment["score"]) for segment in segments))

    def test_repeated_characters_and_nonzero_blank_id(self):
        probabilities = np.full((60, 3), 0.005)
        probabilities[:, 2] = 0.99
        probabilities[10:15] = [0.99, 0.005, 0.005]
        probabilities[35:40] = [0.99, 0.005, 0.005]
        segments = aligner.align_emissions(np.log(probabilities), [0, 0], ["门", "开", "<pad>"], 2, 0.02)
        self.assertEqual(len(segments), 2)
        self.assertLess(segments[0]["ctc_anchor"], segments[1]["ctc_anchor"])

    def test_invalid_or_short_emissions_rejected(self):
        for emissions in (np.zeros((2, 3)), np.zeros((20, 2)), np.ones((20, 3)), np.full((20, 3), np.nan)):
            with self.subTest(shape=emissions.shape), self.assertRaises(ValueError):
                aligner.align_emissions(emissions, [1, 2], ["<pad>", "开", "门"], 0, 0.02)


if __name__ == "__main__":
    unittest.main()
