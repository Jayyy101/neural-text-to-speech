"""Model-free checks for the copy-only held-out breath rendering rules."""

import unittest

from evaluation.run_heldout_breath_ab import CONTEXT_FRAMES, DONOR_FRAMES, render_pair


class HeldoutBreathTests(unittest.TestCase):
    def test_insert_pair_differs_only_in_equal_length_middle(self):
        left = b"\x01\x00" * CONTEXT_FRAMES
        right = b"\x02\x00" * CONTEXT_FRAMES
        donor = b"\x03\x00" * DONOR_FRAMES
        quiet, _, _ = render_pair(left, right, donor, "insert", False)
        breath, _, _ = render_pair(left, right, donor, "insert", True)
        self.assertEqual(len(quiet), len(breath))
        self.assertEqual(quiet, left + b"\x00\x00" * DONOR_FRAMES + right)
        self.assertEqual(breath, left + donor + right)

    def test_long_seam_keeps_native_quiet_and_all_following_speech(self):
        left = b"\x01\x00" * CONTEXT_FRAMES
        native_quiet = b"\x02\x00" * DONOR_FRAMES
        right_speech = b"\x03\x00" * (CONTEXT_FRAMES - DONOR_FRAMES)
        right = native_quiet + right_speech
        donor = b"\x04\x00" * DONOR_FRAMES
        quiet, _, _ = render_pair(left, right, donor, "replace_right_quiet", False)
        breath, _, _ = render_pair(left, right, donor, "replace_right_quiet", True)
        self.assertEqual(quiet, left + right)
        self.assertEqual(breath, left + donor + right_speech)
        self.assertEqual(len(quiet), len(breath))


if __name__ == "__main__":
    unittest.main()
