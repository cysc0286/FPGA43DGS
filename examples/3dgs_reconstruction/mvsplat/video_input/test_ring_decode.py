"""The ring decoder preserves presentation indices without converting unused frames."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from video_input.ring import decode_ring


class FakeCapture:
    def __init__(self, count=21, *, metadata_count=None, fps=29.97, opened=True,
                 failed_retrieve=None, empty_retrieve=None, changed_shape=None):
        self.frames = [np.full((128, 256, 3), i, np.uint8) for i in range(count)]
        if changed_shape is not None:
            self.frames[changed_shape] = np.full((256, 256, 3), changed_shape, np.uint8)
        self.metadata_count = count if metadata_count is None else metadata_count
        self.fps, self.opened = fps, opened
        self.failed_retrieve, self.empty_retrieve = failed_retrieve, empty_retrieve
        self.position, self.grabs = -1, 0
        self.retrieved, self.released = [], False

    def isOpened(self):
        return self.opened

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.metadata_count
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        raise AssertionError('Unexpected capture metadata')

    def read(self):
        raise AssertionError('Unused ring frames must not be materialized by read()')

    def grab(self):
        self.grabs += 1
        self.position += 1
        return self.position < len(self.frames)

    def retrieve(self):
        self.retrieved.append(self.position)
        if self.position == self.failed_retrieve:
            return False, None
        if self.position == self.empty_retrieve:
            return True, None
        return True, self.frames[self.position].copy()

    def release(self):
        self.released = True


class RingDecodeTests(unittest.TestCase):
    def decode(self, capture, **kwargs):
        with patch.object(cv2, 'VideoCapture', return_value=capture) as construct:
            result = decode_ring('source.mp4', candidates=8, **kwargs)
        construct.assert_called_once_with('source.mp4', cv2.CAP_FFMPEG,
                                         [cv2.CAP_PROP_N_THREADS, 1])
        return result

    def test_only_selected_presentation_frames_are_retrieved_and_resized(self):
        capture = FakeCapture()
        frames, count, fps = self.decode(capture, width=128)
        selected = np.linspace(0, 20, 8, dtype=int).tolist()
        self.assertEqual(list(frames), selected)
        self.assertEqual(capture.retrieved, selected)
        self.assertEqual((count, fps), (21, 29.97))
        self.assertEqual(capture.grabs, 22)  # Includes the EOF check.
        self.assertTrue(capture.released)
        for index, frame in frames.items():
            expected = cv2.resize(capture.frames[index], (128, 64), interpolation=cv2.INTER_AREA)
            np.testing.assert_array_equal(frame, expected)

    def test_calibration_only_receives_selected_frames_and_requested_width(self):
        capture = FakeCapture()
        calls = []
        class Calibration:
            def rectify(self, frame, width):
                calls.append((int(frame[0, 0, 0]), width))
                return frame[::2, ::2].copy()
        frames, _, _ = self.decode(capture, width=200, calibration=Calibration())
        selected = np.linspace(0, 20, 8, dtype=int).tolist()
        self.assertEqual(calls, [(index, 200) for index in selected])
        self.assertEqual({frame.shape for frame in frames.values()}, {(64, 128, 3)})
        self.assertTrue(capture.released)

    def test_estimated_frame_count_is_resampled_from_observed_eof(self):
        for metadata_count in (20, 22):
            with self.subTest(metadata_count=metadata_count):
                first = FakeCapture(metadata_count=metadata_count)
                second = FakeCapture(metadata_count=metadata_count)
                with patch.object(cv2, 'VideoCapture', side_effect=[first, second]):
                    frames, count, fps = decode_ring('source.mkv', candidates=8)
                selected = np.linspace(0, 20, 8, dtype=int).tolist()
                self.assertEqual((count, fps), (21, 29.97))
                self.assertEqual(list(frames), selected)
                self.assertEqual(second.retrieved, selected)
                for index, frame in frames.items():
                    np.testing.assert_array_equal(frame, second.frames[index])
                self.assertTrue(first.released and second.released)

    def test_changing_decoded_stream_is_rejected_on_count_recheck(self):
        first = FakeCapture(metadata_count=22)
        second = FakeCapture(count=20, metadata_count=22)
        with patch.object(cv2, 'VideoCapture', side_effect=[first, second]):
            with self.assertRaisesRegex(ValueError, 'Decoded frame count differs'):
                decode_ring('source.mkv', candidates=8)
        self.assertTrue(first.released and second.released)

    def test_actual_short_stream_is_rejected_even_when_metadata_is_long(self):
        capture = FakeCapture(count=7, metadata_count=21)
        with self.assertRaisesRegex(ValueError, 'Too few timed frames'):
            self.decode(capture)
        self.assertTrue(capture.released)

    def test_selected_retrieve_failure_is_rejected_and_releases_capture(self):
        for failure in ('failed_retrieve', 'empty_retrieve'):
            with self.subTest(failure=failure):
                capture = FakeCapture(**{failure: 2})
                with self.assertRaisesRegex(ValueError, 'Cannot retrieve selected panorama frame'):
                    self.decode(capture)
                self.assertTrue(capture.released)

    def test_selected_frame_shape_change_is_rejected(self):
        capture = FakeCapture(changed_shape=2)
        with self.assertRaisesRegex(ValueError, 'Capture dimensions changed'):
            self.decode(capture)
        self.assertTrue(capture.released)

    def test_too_few_frames_or_invalid_fps_is_rejected_before_grabbing(self):
        for kwargs in ({'metadata_count': 7}, {'fps': 0.}, {'fps': float('nan')}, {'fps': -1.}):
            with self.subTest(kwargs=kwargs):
                capture = FakeCapture(**kwargs)
                with self.assertRaisesRegex(ValueError, 'Too few timed frames'):
                    self.decode(capture)
                self.assertEqual(capture.grabs, 0)
                self.assertTrue(capture.released)

    def test_unopened_video_is_rejected(self):
        capture = FakeCapture(opened=False)
        with self.assertRaisesRegex(ValueError, 'Cannot open panorama video'):
            self.decode(capture)

    def test_invalid_budget_is_rejected_before_opening_video(self):
        for kwargs in ({'candidates': 7}, {'candidates': 65}, {'width': 127}, {'threads': 0}):
            with self.subTest(kwargs=kwargs), patch.object(cv2, 'VideoCapture') as construct:
                with self.assertRaisesRegex(ValueError, 'Invalid rotation sampling budget'):
                    decode_ring('source.mp4', **kwargs)
                construct.assert_not_called()


if __name__ == '__main__':
    unittest.main()
