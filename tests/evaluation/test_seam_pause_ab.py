"""Model-free checks for the experimental seam comparison tool."""

from array import array
import sys
import tempfile
import unittest
from pathlib import Path
import wave

from evaluation.run_seam_pause_ab import (
    _write_clip, edge_quiet_frames, inserted_frames,
)
from evaluation.run_edge_cleanup_ab import render_variant


def pcm16(values):
    samples = array("h", values)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


class SeamPauseExperimentTests(unittest.TestCase):
    def test_edge_measurement_requires_contiguous_quiet_windows(self):
        # 24 kHz makes each 10 ms window exactly 240 frames.
        active = [1000] * 240
        quiet = [0] * 240
        left = pcm16(active + quiet * 2)
        right = pcm16(quiet * 3 + active + quiet)
        self.assertEqual(edge_quiet_frames(left, 24000, "trailing"), 480)
        self.assertEqual(edge_quiet_frames(right, 24000, "leading"), 720)
        # A single peak above the gate stops the edge run even when RMS is low.
        spiked = pcm16(quiet + [0] * 239 + [200] + quiet)
        self.assertEqual(edge_quiet_frames(spiked, 24000, "leading"), 240)

    def test_insertion_only_fills_missing_silence(self):
        self.assertEqual(inserted_frames(200, 2400, 24000), 2400)
        self.assertEqual(inserted_frames(150, 4800, 24000), 0)
        self.assertEqual(inserted_frames(300, 12000, 24000), 0)

    def test_clip_preserves_source_samples_and_inserts_only_zeros(self):
        left, right = pcm16([1234, -100]), pcm16([-987, 4321])
        inserted = inserted_frames(150, 0, 24000)
        expected = left + b"\x00\x00" * inserted + right
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.wav"
            _write_clip(path, 24000, expected)
            with wave.open(str(path), "rb") as audio:
                self.assertEqual(audio.getnframes(), 4 + inserted)
                self.assertEqual(audio.readframes(audio.getnframes()), expected)

    def test_copy_only_cuts_remeasure_edges_before_minimum_pause(self):
        active = pcm16([1000] * 240)
        quiet = pcm16([0] * 240)
        left = active + quiet + active  # terminal artifact
        right = quiet * 2 + active
        left_clip, left_evidence = render_variant(
            left, right, 24000, "left", 10, 200, context_seconds=1)
        self.assertEqual(left_evidence["left_cut_frames"], 240)
        self.assertEqual(left_evidence["resulting_left_trailing_silence_ms"], 10)
        self.assertEqual(left_evidence["resulting_right_leading_silence_ms"], 20)
        self.assertEqual(left_evidence["inserted_silence_ms"], 170)
        self.assertEqual(left_clip, active + quiet + quiet * 17 + right)

        right_with_artifact = active + quiet + active
        right_clip, right_evidence = render_variant(
            left, right_with_artifact, 24000, "right", 10, None,
            context_seconds=1)
        self.assertEqual(right_evidence["right_cut_frames"], 240)
        self.assertEqual(right_evidence["resulting_right_leading_silence_ms"], 10)
        self.assertEqual(right_evidence["inserted_silence_ms"], 0)
        self.assertEqual(right_clip, left + quiet + active)


if __name__ == "__main__":
    unittest.main()
