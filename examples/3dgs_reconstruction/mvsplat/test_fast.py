"""Frame selection contracts for arbitrary-length fast video inputs."""
import unittest

from prepare_fast import frame_indices


class FrameSelectionTests(unittest.TestCase):
    def test_original_sequence_is_preserved(self):
        self.assertEqual(frame_indices(30, [0, -1], None), ([0, 29], [7, 15, 22]))

    def test_length_and_explicit_negative_indices(self):
        self.assertEqual(frame_indices(120, [0, -1], None), ([0, 119], [30, 60, 90]))
        self.assertEqual(frame_indices(12, [0, -1], [-6]), ([0, 11], [6]))

    def test_invalid_indices_or_no_target_fail(self):
        for count, context, targets in ((30, [0, -1], []), (30, [0, 0], [7]),
                                        (30, [0, -1], [29]), (30, [0, -1], [30]),
                                        (3, [0, -1], None)):
            with self.subTest(count=count, context=context, targets=targets):
                with self.assertRaises(ValueError):
                    frame_indices(count, context, targets)


if __name__ == "__main__":
    unittest.main()
