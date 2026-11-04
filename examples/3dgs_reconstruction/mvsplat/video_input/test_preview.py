"""MJPEG pipe fragmentation, stale-frame and storage-limited capture regressions."""
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from video_input.preview import JpegFrames, Preview
from video_input.capture import capture_clip
from video_input.decode import decode_selected,temporal_candidate_spans


class PreviewTests(unittest.TestCase):
    def test_fifteen_second_recording_limits_both_ffmpeg_outputs(self):
        # FFmpeg output options are scoped per output. Limiting just Matroska
        # leaves the preview pipe alive and prevents automatic finalization.
        process = MagicMock()
        process.pid = 12345
        process.poll.return_value = 0
        resource = SimpleNamespace(RLIMIT_FSIZE=1, prlimit=MagicMock())
        limit = 128*1024**2
        with tempfile.TemporaryDirectory() as folder:
            recording = Path(folder)/'capture.partial.mkv'
            with patch('video_input.preview.Path.exists', return_value=True), \
                    patch('video_input.preview.subprocess.Popen', return_value=process) as spawn, \
                    patch('video_input.preview.threading.Thread'), \
                    patch.dict('sys.modules', {'resource': resource}):
                preview = Preview('/dev/video1', 1280, 720, 60, Path(folder)/'ffmpeg.log',
                                  recording=recording, storage_limit=limit, max_seconds=15)
                try:
                    command = spawn.call_args.args[0]
                    source = command.index('-i')+1
                    saved = command.index(str(recording))
                    piped = command.index('pipe:1')
                    self.assertEqual(command[source], '/dev/video1')
                    self.assertLess(saved, piped)
                    self.assertNotIn('-t', command[:source])
                    for output_options in (command[source+1:saved], command[saved+1:piped]):
                        self.assertEqual(output_options.count('-t'), 1)
                        self.assertEqual(output_options[output_options.index('-t')+1], '15')
                    resource.prlimit.assert_called_once_with(12345, 1, (limit, limit))
                finally:
                    preview.close()

    def test_local_temporal_search_has_fixed_budget_and_valid_original_spans(self):
        self.assertEqual(temporal_candidate_spans(1000,60),[(760,999),(380,619),(0,239)])
        self.assertEqual(temporal_candidate_spans(240,60),[])

    def test_fragmented_jpeg_markers_and_multiple_frames(self):
        parser = JpegFrames()
        frames = []
        stream = b'junk\xff\xd8image1\xff\xd9\xff\xd8image2\xff\xd9'
        for byte in stream:
            frames.extend(parser.feed(bytes([byte])))
        self.assertEqual(frames, [b'\xff\xd8image1\xff\xd9', b'\xff\xd8image2\xff\xd9'])
        self.assertLessEqual(len(parser.buffer), 1)

    def test_corrupt_or_oversized_stream_has_bounded_buffer(self):
        parser = JpegFrames(limit=16)
        parser.feed(b'no jpeg'*100)
        self.assertLessEqual(len(parser.buffer), 1)
        with self.assertRaisesRegex(ValueError, 'limit'):
            parser.feed(b'\xff\xd8'+b'x'*20)
        with self.assertRaisesRegex(ValueError, 'limit'):
            JpegFrames(limit=16).feed(b'\xff\xd8'+b'x'*20+b'\xff\xd9')

    def test_stale_or_disconnected_camera_never_repeats_last_frame_as_live(self):
        preview = Preview.__new__(Preview)
        preview.lock = threading.Lock()
        preview.error = None
        preview.latest = b'jpeg'
        preview.count = 61
        preview.first = time.monotonic()-1
        preview.last = time.monotonic()
        image, count, fps = preview.snapshot()
        self.assertEqual((image, count), (b'jpeg', 61))
        self.assertAlmostEqual(fps, 60, delta=.1)
        preview.last -= 4
        with self.assertRaisesRegex(RuntimeError, 'fresh'):
            preview.snapshot()
        preview.error = 'camera disconnected'
        with self.assertRaisesRegex(RuntimeError, 'disconnected'):
            preview.snapshot()

    def test_capture_storage_limit_and_mkv_command_on_small_board_filesystem(self):
        stream = dict(codec_name='mjpeg', width=1280, height=720, fps=60.)
        times = [i/60 for i in range(180)]
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'capture'
            def simulate(command, **kwargs):
                Path(command[-1]).write_bytes(b'mock MJPEG Matroska')
                self.assertEqual(command[-1].split('.')[-1], 'mkv')
                self.assertEqual(command[command.index('-fs')+1], str(256*1024**2))
            with patch('video_input.capture.Path.exists', return_value=True), \
                    patch('video_input.capture.shutil.disk_usage', return_value=SimpleNamespace(free=373*1024**2)), \
                    patch('video_input.capture.subprocess.run', side_effect=simulate), \
                    patch('video_input.capture.inspect_clip', return_value=(stream, times)):
                result = capture_clip('/dev/video0', out, fps=60, seconds=3)
            self.assertTrue(result['complete'])
            self.assertEqual(Path(result['path']).name, 'capture.mkv')

    def test_selection_uses_actual_packet_count_instead_of_container_slots(self):
        import numpy as np
        frames = [np.zeros((90, 160, 3), np.uint8) for _ in range(5)]
        class Capture:
            def isOpened(self):
                return True
            def get(self, prop):
                import cv2
                return 10 if prop == cv2.CAP_PROP_FRAME_COUNT else 60
            def read(self):
                return (True, frames.pop()) if frames else (False, None)
            def release(self):
                self.released = True
        cap = Capture()
        with patch('video_input.decode.cv2.VideoCapture', return_value=cap), \
                patch('video_input.capture.inspect_clip', return_value=(dict(fps=60), [i/60 for i in range(5)])):
            result, count, _, _, _ = decode_selected('unused', [0, -1], None, 160, 1, candidate_count=3)
        self.assertEqual((count, sorted(result)), (5, [0, 2, 4]))
        self.assertTrue(cap.released)


if __name__ == '__main__':
    unittest.main()
