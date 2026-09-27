"""Synthetic, model-free checks for manual PCM16 pause repair."""

import contextlib
import io
from pathlib import Path
import struct
import tempfile
import unittest
import wave

from evaluation.repair_pause import main, repair_pause, select_quiet_point


def write_pcm(path, samples, rate=24000, channels=1, width=2):
    data = (struct.pack(f"<{len(samples)}h", *samples) if width == 2
            else bytes(samples))
    with wave.open(str(path), "wb") as audio:
        audio.setparams((channels, width, rate, 0, "NONE", "not compressed"))
        audio.writeframes(data)
    return data


def read_pcm(path):
    with wave.open(str(path), "rb") as audio:
        return audio.getparams(), audio.readframes(audio.getnframes())


class PauseRepairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "input.wav"
        self.output = self.root / "repaired.wav"

    def test_duration_format_and_every_original_sample_preserved(self):
        for channels in (1, 2):
            with self.subTest(channels=channels):
                rate = 24000
                # Both polarities, extremes, and distinct channels exercise raw preservation.
                samples = [((i * 7919) % 65536) - 32768 for i in range(rate * channels)]
                original = write_pcm(self.input, samples, rate, channels)
                input_file = self.input.read_bytes()
                output = self.root / f"repaired_{channels}.wav"
                result = repair_pause(self.input, output, around=0.6, add_ms=140)
                params, repaired = read_pcm(output)
                self.assertEqual((params.framerate, params.nchannels, params.sampwidth),
                                 (rate, channels, 2))
                self.assertEqual(params.nframes, rate + 3360)
                self.assertAlmostEqual(result["output_duration_seconds"], 1.14)
                split = result["insertion_frame"] * channels * 2
                inserted = 3360 * channels * 2
                self.assertEqual(repaired[:split], original[:split])
                self.assertEqual(repaired[split:split + inserted], b"\0" * inserted)
                self.assertEqual(repaired[split + inserted:], original[split:])
                self.assertEqual(self.input.read_bytes(), input_file)

    def test_finds_quiet_region_instead_of_requested_time_or_noise_tail(self):
        samples = [3000] * 1000
        samples[460:500] = [0] * 40
        samples[500:590] = [100] * 90  # A breath/noise tail, quieter than speech.
        write_pcm(self.input, samples, rate=1000)
        result = repair_pause(self.input, self.output, around=0.6, add_ms=140)
        self.assertTrue(460 < result["insertion_frame"] < 500)
        self.assertEqual(result["quiet_region_rms_pcm16"], 0)
        self.assertEqual(result["search_start_seconds"], 0.35)
        self.assertEqual(result["search_end_seconds"], 0.85)

    def test_near_zero_boundary_inside_quiet_frame_and_determinism(self):
        samples = [3000] * 1000
        samples[470:480] = [20, 20, 20, 20, 1, -1, 20, 20, 20, 20]
        data = struct.pack("<1000h", *samples)
        first = select_quiet_point(data, 1000, 1, 0.6, 250)
        self.assertEqual(first[0], 475)
        self.assertEqual(first, select_quiet_point(data, 1000, 1, 0.6, 250))

    def test_earlier_valley_before_noise_bump_preferred_within_three_db(self):
        # Earlier valley is 2.92 dB above the later minimum, just inside 3 dB.
        samples = [3000] * 1000
        samples[460:510] = [14, -14] * 25
        samples[510:620] = [100, -100] * 55  # Small breath/noise-like rise.
        samples[620:670] = [10, -10] * 25
        write_pcm(self.input, samples, rate=1000)
        result = repair_pause(self.input, self.output, around=0.6, add_ms=140)
        self.assertEqual(result["insertion_frame"], 485)  # Center of earlier flat valley.
        self.assertEqual(result["quiet_region_rms_pcm16"], 14)
        self.assertEqual(result["quiet_valley_tolerance_db"], 3.0)
        data = struct.pack("<1000h", *samples)
        self.assertEqual(select_quiet_point(data, 1000, 1, 0.6, 250)[0], 485)

    def test_earlier_valley_outside_tolerance_is_not_preferred(self):
        # 15 versus 10 RMS is 3.52 dB: the earlier valley must not qualify.
        samples = [3000] * 1000
        samples[460:510] = [15, -15] * 25
        samples[510:620] = [100, -100] * 55
        samples[620:670] = [10, -10] * 25
        data = struct.pack("<1000h", *samples)
        point, _, _, rms = select_quiet_point(data, 1000, 1, 0.6, 250)
        self.assertEqual(point, 645)
        self.assertEqual(rms, 10)

    def test_zero_minimum_accepts_only_zero_energy_and_prefers_earlier_tie(self):
        for early_amplitude, expected in ((0, 485), (1, 645)):
            with self.subTest(early_amplitude=early_amplitude):
                samples = [3000] * 1000
                samples[460:510] = [early_amplitude] * 50
                samples[510:620] = [100] * 110
                samples[620:670] = [0] * 50
                data = struct.pack("<1000h", *samples)
                point, _, _, rms = select_quiet_point(data, 1000, 1, 0.6, 250)
                self.assertEqual(point, expected)
                self.assertEqual(rms, 0)

    def test_stereo_opposite_polarity_does_not_cancel_energy(self):
        samples = [value for _ in range(1000) for value in (2000, -2000)]
        samples[460 * 2:500 * 2] = [0] * 80
        write_pcm(self.input, samples, rate=1000, channels=2)
        result = repair_pause(self.input, self.output, around=0.6, add_ms=140)
        self.assertTrue(460 < result["insertion_frame"] < 500)
        self.assertEqual(result["quiet_region_rms_pcm16"], 0)

    def test_fractional_silence_rounds_to_nearest_audio_frame(self):
        rate = 44100
        write_pcm(self.input, [0] * rate, rate=rate)
        result = repair_pause(self.input, self.output, around=0.5, add_ms=0.14)
        params, _ = read_pcm(self.output)
        self.assertEqual(params.nframes - rate, 6)
        self.assertLessEqual(abs(result["actual_add_ms"] - 0.14), 500 / rate)

    def test_search_is_clipped_at_file_edges(self):
        write_pcm(self.input, [0] * 1000, rate=1000)
        for around in (0.005, 0.995):
            with self.subTest(around=around):
                result = repair_pause(self.input, self.root / f"{around}.wav", around, 140)
                self.assertGreater(result["insertion_frame"], 0)
                self.assertLess(result["insertion_frame"], 1000)
                self.assertGreaterEqual(result["search_start_seconds"], 0)
                self.assertLessEqual(result["search_end_seconds"], 1)

    def test_invalid_parameters_rejected_before_output_creation(self):
        write_pcm(self.input, [0] * 1000, rate=1000)
        cases = [
            ({"around": value}, "around")
            for value in (-1, 0, 1, 2, float("nan"), float("inf"))
        ] + [
            ({"add_ms": value}, "add-ms")
            for value in (-1, 0, 0.001, float("nan"), float("inf"), 1e308)
        ] + [
            ({"search_ms": value}, "[Ss]earch")
            for value in (-1, 0, 0.001, float("nan"), float("inf"))
        ]
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                args = {"around": 0.5, "add_ms": 140, "search_ms": 250}
                args.update(overrides)
                with self.assertRaisesRegex(ValueError, message):
                    repair_pause(self.input, self.output, **args)
                self.assertFalse(self.output.exists())

    def test_unsupported_pcm_width_rejected(self):
        write_pcm(self.input, [128] * 1000, rate=1000, width=1)
        with self.assertRaisesRegex(ValueError, "PCM16"):
            repair_pause(self.input, self.output, 0.5, 140)
        self.assertFalse(self.output.exists())

    def test_float_wav_rejected_without_conversion(self):
        fmt = struct.pack("<HHIIHH", 3, 1, 1000, 4000, 4, 32)
        data = struct.pack("<1000f", *([0.0] * 1000))
        body = b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt
        body += b"data" + struct.pack("<I", len(data)) + data
        self.input.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
        with self.assertRaisesRegex(ValueError, "PCM16"):
            repair_pause(self.input, self.output, 0.5, 140)
        self.assertFalse(self.output.exists())

    def test_empty_and_truncated_audio_rejected(self):
        write_pcm(self.input, [], rate=1000)
        with self.assertRaisesRegex(ValueError, "empty"):
            repair_pause(self.input, self.output, 0.5, 140)
        write_pcm(self.input, [0] * 1000, rate=1000)
        self.input.write_bytes(self.input.read_bytes()[:-2])
        with self.assertRaisesRegex(ValueError, "truncated"):
            repair_pause(self.input, self.output, 0.5, 140)
        self.assertFalse(self.output.exists())

    def test_input_and_existing_output_cannot_be_overwritten(self):
        write_pcm(self.input, [0] * 1000, rate=1000)
        original = self.input.read_bytes()
        with self.assertRaisesRegex(ValueError, "different files"):
            repair_pause(self.input, self.input, 0.5, 140)
        self.output.write_bytes(b"existing evidence")
        with self.assertRaisesRegex(ValueError, "already exists"):
            repair_pause(self.input, self.output, 0.5, 140)
        self.assertEqual(self.input.read_bytes(), original)
        self.assertEqual(self.output.read_bytes(), b"existing evidence")

    def test_cli_report_and_clear_argument_error(self):
        write_pcm(self.input, [0] * 1000, rate=1000)
        args = ["--input", str(self.input), "--output", str(self.output),
                "--around", "0.5", "--add-ms", "140"]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(args), 0)
        for label in ("Input:", "Requested boundary:", "Search window:",
                      "Selected insertion:", "region RMS", "Silence:", "Output:"):
            self.assertIn(label, stdout.getvalue())
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            main(args[:-1] + ["nan"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("add-ms must be a finite positive", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
