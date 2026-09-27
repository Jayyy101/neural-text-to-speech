import unittest

from evaluation.run_ch02_prospective_breath_ab import render_full_chapter
from evaluation.run_heldout_breath_ab import DONOR_FRAMES


class Chapter2ProspectiveBreathTests(unittest.TestCase):
    def test_render_preserves_units_and_inserts_only_at_selected_joins(self):
        pcms = [bytes([number, 0]) * (number + 2) for number in range(1, 7)]
        block = b"\x35\x12" * DONOR_FRAMES
        rendered, edits = render_full_chapter(pcms, block, (2, 5))

        expected = b"".join((pcms[0], pcms[1], block, pcms[2], pcms[3],
                             pcms[4], block, pcms[5]))
        self.assertEqual(rendered, expected)
        self.assertEqual(edits[2], (len(pcms[0]) + len(pcms[1])) // 2)
        self.assertEqual(len(rendered), sum(map(len, pcms)) + 2 * len(block))

    def test_render_rejects_invalid_boundaries_and_block_length(self):
        pcms = [b"\x01\x00" * 4, b"\x02\x00" * 4]
        block = b"\x00\x00" * DONOR_FRAMES
        with self.assertRaises(ValueError):
            render_full_chapter(pcms, block, (2,))
        with self.assertRaises(ValueError):
            render_full_chapter(pcms, block[:-2], (1,))


if __name__ == "__main__":
    unittest.main()
