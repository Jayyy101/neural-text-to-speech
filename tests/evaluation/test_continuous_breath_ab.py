"""Model-free exact-PCM checks for the continuous sparse-breath experiment."""

import unittest

from evaluation.run_continuous_breath_ab import render_passage
from evaluation.run_heldout_breath_ab import DONOR_FRAMES, RATE


class ContinuousBreathTests(unittest.TestCase):
    def test_two_sparse_edits_preserve_every_other_source_sample(self):
        # First 240 ms of unit 21 is measured quiet, though not necessarily zero.
        units = (
            b"\x01\x00" * RATE,
            b"\x02\x00" * DONOR_FRAMES + b"\x03\x00" * (RATE - DONOR_FRAMES),
            b"\x04\x00" * RATE,
            b"\x05\x00" * RATE,
        )
        donor = b"\x06\x00" * DONOR_FRAMES
        quiet, q_edits, q_joins, _ = render_passage(units, donor, False)
        breath, b_edits, b_joins, _ = render_passage(units, donor, True)
        self.assertEqual(q_joins, b_joins)
        self.assertEqual(q_edits, b_edits)
        self.assertEqual(len(quiet), len(breath))
        self.assertEqual(len(quiet) // 2, 4 * RATE + DONOR_FRAMES)
        for after in (20, 22):
            start = q_edits[after]["output_start_frame"] * 2
            end = start + DONOR_FRAMES * 2
            self.assertEqual(quiet[start:end], b"\x00\x00" * DONOR_FRAMES)
            self.assertEqual(breath[start:end], donor)
            quiet = quiet[:start] + donor + quiet[end:]
        self.assertEqual(quiet, breath)
        self.assertEqual(q_joins[21], (2 * RATE))


if __name__ == "__main__":
    unittest.main()
