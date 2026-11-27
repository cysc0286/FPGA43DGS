"""Accept measured camera rates without accepting corrupt or stalled clips."""
import json
import math
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from video_input.capture import validate_recorded_capture
from video_input.recording import CameraRecorder
from video_input.test_recording import FakePreview


class CaptureRateTests(unittest.TestCase):
    def stream(self, fps=60):
        return dict(codec_name='mjpeg',width=1280,height=720,fps=fps)

    def test_measured_rate_is_preserved_at_30_42_51_and_variable_fps(self):
        for rate in (12,30,42.52,51.48,59.66):
            with self.subTest(rate=rate):
                times=[i/rate for i in range(120)]
                observed=validate_recorded_capture(self.stream(),times,1280,720,60)
                self.assertAlmostEqual(observed['measured_fps'],rate)
                self.assertEqual(observed['requested_fps'],60)
                self.assertEqual(observed['fps_policy'],'measured')
        times=[0.]
        for i in range(100):times.append(times[-1]+(1/30 if i%2 else 1/50))
        observed=validate_recorded_capture(self.stream(),times,1280,720,60)
        self.assertAlmostEqual(observed['duration_seconds'],times[-1])
        self.assertIn('未补帧',observed['fps_warning'])

    def test_negotiated_30fps_is_allowed_and_old_strict_gate_is_opt_in(self):
        times=[i/30 for i in range(100)]
        self.assertEqual(validate_recorded_capture(self.stream(30),times,1280,720,60)['reported_fps'],30)
        with self.assertRaisesRegex(ValueError,'严格帧率'):
            validate_recorded_capture(self.stream(30),times,1280,720,60,strict_fps=True)
        with self.assertRaisesRegex(ValueError,'严格帧率'):
            validate_recorded_capture(self.stream(),times,1280,720,60,strict_fps=True)
        self.assertEqual(validate_recorded_capture(self.stream(),[i/60 for i in range(100)],1280,720,60,strict_fps=True)['fps_policy'],'strict')

    def test_wrong_codec_dimensions_and_nonfinite_metadata_remain_rejected(self):
        for changes in (dict(codec_name='h264'),dict(width=640),dict(height=360),dict(fps=0),dict(fps=math.nan)):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                validate_recorded_capture(self.stream()|changes,[i/30 for i in range(100)],1280,720,60)

    def test_bad_timestamps_and_obvious_gap_remain_rejected(self):
        examples=([0.,.1],[0.,.1,.1],[0.,math.nan,.2],[0.,math.inf,.2],[0.,.2,.1],
                  [i/30 for i in range(100)]+[5.0+i/30 for i in range(100)])
        for times in examples:
            with self.subTest(times=times[:4]),self.assertRaises(ValueError):
                validate_recorded_capture(self.stream(),times,1280,720,60)

    def test_low_uniform_rate_is_not_misclassified_as_a_stream_pause(self):
        self.assertAlmostEqual(validate_recorded_capture(self.stream(1),[0.,1.,2.,3.],1280,720,1)['measured_fps'],1.)

    def record(self, folder, times, strict=False):
        camera=CameraRecorder('/dev/video0',1280,720,60,folder,factory=FakePreview,strict_fps=strict)
        with patch('video_input.recording.inspect_clip',return_value=(self.stream(),times)):
            camera.start()
            deadline=time.monotonic()+3
            while camera.state=='starting' and time.monotonic()<deadline:time.sleep(.01)
            self.assertEqual(camera.state,'recording')
            camera.stop();camera.worker.join(3)
        return camera

    def test_recorder_publishes_actual_rate_and_does_not_rewrite_pts(self):
        times=[i/42.52 for i in range(180)]
        with tempfile.TemporaryDirectory() as folder:
            camera=self.record(folder,times)
            try:
                self.assertEqual(camera.state,'recorded',camera.error)
                self.assertEqual(camera.clip['frame_pts_seconds'],times)
                self.assertAlmostEqual(camera.status()['clip']['measured_fps'],42.52)
                self.assertEqual(camera.status()['capture_fps_policy'],'measured')
                self.assertIn('42.52',camera.status()['clip']['fps_warning'])
                meta=json.loads(next(Path(folder).glob('clip_*/capture.json')).read_text(encoding='utf-8'))
                self.assertTrue(meta['complete'])
                self.assertEqual(meta['requested']['fps'],60)
                self.assertEqual(meta['actual']['fps'],60)
            finally:camera.close()

    def test_recorder_strict_mode_still_keeps_failed_clip_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            camera=self.record(folder,[i/42.52 for i in range(180)],strict=True)
            try:
                self.assertEqual(camera.state,'error')
                self.assertIsNone(camera.clip)
                self.assertFalse(list(Path(folder).glob('clip_*/capture.mkv')))
                self.assertTrue(list(Path(folder).glob('clip_*/capture.partial.mkv')))
            finally:camera.close()


if __name__=='__main__':unittest.main()
