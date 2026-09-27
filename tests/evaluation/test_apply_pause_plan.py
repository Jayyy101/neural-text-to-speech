"""Model-free batch pause tests using only generated PCM16 data."""

import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

from evaluation import apply_pause_plan as batch
from evaluation import repair_pause as single


class ApplyPausePlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "input.wav"
        self.output = self.root / "output.wav"
        self.plan = self.root / "plan.json"
        self.entries = [{"around": 0.6, "add_ms": 400, "label": "period"},
                        {"around": 1.6, "add_ms": 140, "label": "paragraph"}]
        self.write_audio()
        self.write_plan(self.entries)

    def write_audio(self, channels=1, rate=1000, width=2):
        samples = []
        for frame in range(2 * rate):
            quiet = 0.46 <= frame / rate < 0.50 or 1.46 <= frame / rate < 1.50
            for channel in range(channels):
                samples.append(0 if quiet else ((frame * 127 + channel * 997) % 6000) - 3000)
        payload = (struct.pack(f"<{len(samples)}h", *samples) if width == 2
                   else bytes([128] * len(samples)))
        with wave.open(str(self.input), "wb") as audio:
            audio.setparams((channels, width, rate, 0, "NONE", "not compressed"))
            audio.writeframes(payload)
        return payload

    def write_plan(self, entries):
        self.plan.write_text(json.dumps({"pauses": entries}), encoding="utf-8")

    def apply(self, output=None, **kwargs):
        return batch.apply_pause_plan(self.input, self.plan, output or self.output, **kwargs)

    def test_multiple_insertions_preserve_mono_stereo_samples_and_duration(self):
        for channels in (1, 2):
            with self.subTest(channels=channels):
                original = self.write_audio(channels)
                original_file = self.input.read_bytes()
                output = self.root / f"output_{channels}.wav"
                result = self.apply(output)
                with wave.open(str(output), "rb") as audio:
                    self.assertEqual((audio.getframerate(), audio.getnchannels(), audio.getsampwidth()),
                                     (1000, channels, 2))
                    self.assertEqual(audio.getnframes(), 2540)
                    payload = audio.readframes(audio.getnframes())
                expected = bytearray()
                cursor = 0
                for repair in sorted(result["repairs"], key=lambda item: item["insertion_frame"]):
                    split = repair["insertion_frame"] * channels * 2
                    expected.extend(original[cursor:split])
                    expected.extend(b"\0" * repair["added_frames"] * channels * 2)
                    cursor = split
                expected.extend(original[cursor:])
                self.assertEqual(payload, bytes(expected))
                self.assertEqual(self.input.read_bytes(), original_file)
                self.assertAlmostEqual(result["output_duration_seconds"], 2.54)
                self.assertEqual(result["total_actual_add_ms"], 540)

    def test_selection_uses_original_audio_once_without_cumulative_timestamp_shift(self):
        with patch.object(single, "read_pcm16", wraps=single.read_pcm16) as reader, \
                patch.object(single, "plan_pause", wraps=single.plan_pause) as planner:
            result = self.apply()
        reader.assert_called_once()
        self.assertIs(planner.call_args_list[0].args[0], planner.call_args_list[1].args[0])
        first, second = result["repairs"]
        self.assertTrue(460 < first["insertion_frame"] < 500)
        self.assertTrue(1460 < second["insertion_frame"] < 1500)
        self.assertEqual(second["around_seconds"], 1.6)
        self.assertAlmostEqual(second["final_insertion_seconds"], second["insertion_seconds"] + 0.4)
        self.assertEqual([item["quiet_region_rms_pcm16"] for item in result["repairs"]], [0, 0])
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["input.wav", "output.wav", "plan.json"])

    def test_out_of_order_plan_is_deterministic_and_labels_are_metadata(self):
        chronological = self.apply(self.root / "chronological.wav")
        reversed_entries = list(reversed(self.entries))
        reversed_entries[0] = dict(reversed_entries[0], label="arbitrary label")
        self.write_plan(reversed_entries)
        reversed_result = self.apply()
        self.assertEqual(self.output.read_bytes(), (self.root / "chronological.wav").read_bytes())
        self.assertEqual([p["plan_index"] for p in reversed_result["repairs"]], [1, 2])
        self.assertEqual(reversed_result["repairs"][0]["label"], "arbitrary label")
        self.assertEqual(reversed_result["repairs"][0]["insertion_frame"],
                         chronological["repairs"][1]["insertion_frame"])

    def test_single_entry_matches_existing_repair_cli_path(self):
        self.write_plan([self.entries[0]])
        result = self.apply()
        single_result = single.repair_pause(self.input, self.root / "single.wav", 0.6, 400)
        self.assertEqual(self.output.read_bytes(), (self.root / "single.wav").read_bytes())
        self.assertEqual(result["repairs"][0]["insertion_frame"], single_result["insertion_frame"])

    def test_individual_frame_rounding_is_summed(self):
        self.write_audio(rate=44100)
        self.write_plan([dict(entry, add_ms=0.14) for entry in self.entries])
        result = self.apply()
        self.assertEqual(result["total_added_frames"], 12)  # Six frames for each request.
        self.assertAlmostEqual(result["output_duration_seconds"], 2 + 12 / 44100)

    def test_invalid_json_and_utf8_fail_without_output(self):
        for content in (b'{"pauses": [}', b'\xff'):
            with self.subTest(content=content):
                self.plan.write_bytes(content)
                with self.assertRaisesRegex(ValueError, "Invalid UTF-8 JSON"):
                    self.apply()
                self.assertFalse(self.output.exists())

    def test_empty_or_malformed_plan_fails_without_output(self):
        for plan in ({"pauses": []}, {}, [], {"pauses": "no"}, {"pauses": [None]},
                     {"pauses": [{"around": 0.6}]}, {"pauses": [{"add_ms": 140}]},
                     {"pauses": [{"around": 0.6, "add_ms": 140, "label": 7}]}):
            with self.subTest(plan=plan):
                self.plan.write_text(json.dumps(plan), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.apply()
                self.assertFalse(self.output.exists())

    def test_invalid_values_even_in_later_entries_prevent_any_output(self):
        for key in ("around", "add_ms"):
            for value in (None, True, "140", 0, -1, float("nan"), float("inf"), 10**400):
                with self.subTest(key=key, value=value):
                    self.write_plan([self.entries[0], dict(self.entries[1], **{key: value})])
                    with self.assertRaisesRegex(ValueError, "index 2"):
                        self.apply()
                    self.assertFalse(self.output.exists())
        for change in ({"around": 2}, {"around": 999}, {"add_ms": 0.0001}):
            self.write_plan([self.entries[0], dict(self.entries[1], **change)])
            with self.assertRaisesRegex(ValueError, "index 2"):
                self.apply()
            self.assertFalse(self.output.exists())

    def test_duplicate_resolved_points_rejected(self):
        self.write_plan([self.entries[0], dict(self.entries[0], around=0.61)])
        with self.assertRaisesRegex(ValueError, "Duplicate or ambiguous.*indices 1 and 2"):
            self.apply()
        self.assertFalse(self.output.exists())

    def test_near_duplicate_points_within_ten_ms_rejected(self):
        # Flat energy with shifted windows gives different, but adjacent, points.
        with wave.open(str(self.input), "wb") as audio:
            audio.setparams((1, 2, 1000, 0, "NONE", "not compressed"))
            audio.writeframes(b"\0" * 4000)
        self.write_plan([self.entries[0], dict(self.entries[0], around=0.605)])
        with self.assertRaisesRegex(ValueError, "within 10 ms"):
            self.apply()
        self.assertFalse(self.output.exists())

    def test_overlapping_search_windows_with_distinct_points_are_allowed(self):
        with wave.open(str(self.input), "wb") as audio:
            audio.setparams((1, 2, 1000, 0, "NONE", "not compressed"))
            audio.writeframes(b"\0" * 4000)
        self.write_plan([self.entries[0], dict(self.entries[0], around=0.8)])
        result = self.apply()
        self.assertEqual(result["repair_count"], 2)

    def test_unsupported_audio_and_existing_output_rejected(self):
        self.write_audio(width=1)
        with self.assertRaisesRegex(ValueError, "PCM16"):
            self.apply()
        self.assertFalse(self.output.exists())
        self.write_audio()
        self.output.write_bytes(b"prior evidence")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.apply()
        self.assertEqual(self.output.read_bytes(), b"prior evidence")

    def test_cli_report_and_invalid_plan_exit_code(self):
        args = ["--input", str(self.input), "--plan", str(self.plan), "--output", str(self.output)]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(batch.main(args), 0)
        for text in ("Plan index 1", "period", "requested ORIGINAL", "search", "selected ORIGINAL",
                     "frame", "RMS", "add", "Original duration", "repairs: 2", "final duration", "Output:"):
            self.assertIn(text, stdout.getvalue())
        self.plan.write_text("{", encoding="utf-8")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            batch.main(args)
        self.assertEqual(error.exception.code, 2)
        self.assertIn("Invalid UTF-8 JSON", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
