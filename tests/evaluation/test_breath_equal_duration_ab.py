"""Model-free PCM identity checks for the experimental breath comparison."""

import unittest

from evaluation.run_breath_equal_duration_ab import (
    CONTEXT_FRAMES, DONOR_END, DONOR_START, RATE, render_case,
)


class BreathEqualDurationTests(unittest.TestCase):
    def test_transplant_pairs_preserve_speech_and_match_duration(self):
        left = b"\x01\x00" * CONTEXT_FRAMES
        right = b"\x02\x00" * CONTEXT_FRAMES
        donor = b"\x03\x00" * (DONOR_END - DONOR_START)
        for after, quiet_name, donor_name in (
                (5, "quiet_equal", "donor_equal"),
                (58, "quiet_equal", "donor_equal")):
            quiet, _, _, _ = render_case(after, quiet_name, left, right, donor)
            breath, _, _, _ = render_case(after, donor_name, left, right, donor)
            self.assertEqual(len(quiet), len(breath))
            self.assertTrue(quiet.startswith(left) and quiet.endswith(right))
            self.assertTrue(breath.startswith(left) and breath.endswith(right))
            self.assertIn(donor, breath)
            self.assertNotIn(donor, quiet)
        baseline, _, _, _ = render_case(5, "baseline", left, right, donor)
        self.assertEqual(baseline, left + right)

    def test_native_control_replaces_only_approved_donor_span(self):
        left = b"\x01\x00" * CONTEXT_FRAMES
        right = (b"\x02\x00" * (DONOR_START - 400 * RATE // 1000)
                 + b"\x03\x00" * (DONOR_END - DONOR_START)
                 + b"\x04\x00" * (CONTEXT_FRAMES - (DONOR_END - 400 * RATE // 1000)))
        donor = b"\x03\x00" * (DONOR_END - DONOR_START)
        native, _, _, _ = render_case(69, "native", left, right, donor)
        quiet, _, _, _ = render_case(69, "quiet_equal", left, right, donor)
        self.assertEqual(len(native), len(quiet))
        start = len(left) + (DONOR_START - 400 * RATE // 1000) * 2
        end = start + len(donor)
        self.assertEqual(native[:start], quiet[:start])
        self.assertEqual(native[end:], quiet[end:])
        self.assertEqual(native[start:end], donor)
        self.assertEqual(quiet[start:end], b"\x00" * len(donor))


if __name__ == "__main__":
    unittest.main()
