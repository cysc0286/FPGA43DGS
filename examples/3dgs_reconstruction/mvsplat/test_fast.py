"""Frame selection contracts for arbitrary-length fast video inputs."""
from pathlib import Path
import sys
import unittest

# ``unittest discover`` is normally launched from the parent reconstruction
# directory.  Put this module's directory first so the local ``evaluate``
# adapter is not confused with the parent package's similarly named module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_fast import first_frame_timing
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


class TimingContractTests(unittest.TestCase):
    def test_new_receipt_uses_verified_frame_endpoint(self):
        timing = first_frame_timing(dict(
            input_end_to_first_verified_fpga_frame_seconds=12.5,
            pipeline_start_to_first_verified_fpga_frame_seconds=12.4,
            first_verified_fpga_target=dict(frame_sha256="abc"),
        ))
        self.assertEqual(timing["input_end_to_first_verified_fpga_frame_seconds"], 12.5)
        self.assertEqual(timing["first_verified_fpga_target"]["frame_sha256"], "abc")

    def test_legacy_timing_is_explicitly_not_input_end_timing(self):
        timing = first_frame_timing(dict(video_to_first_fpga_image_seconds=66.68))
        self.assertIsNone(timing["input_end_to_first_verified_fpga_frame_seconds"])
        self.assertIn("No input-end receipt", timing["legacy_timing_note"])


if __name__ == "__main__":
    unittest.main()
