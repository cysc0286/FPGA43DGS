"""Capture safety and independent calibration validation; never open a device."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from video_input.calibration import CalibrationCapture, capture_server, inspect_board, validate_fit


class CaptureTests(unittest.TestCase):
    def session(self, root):
        with patch.object(CalibrationCapture, '_poll'):
            session = CalibrationCapture('http://127.0.0.1:18433/frame.jpg', Path(root)/'capture')
        session.size = [1280, 720]
        self.addCleanup(session.close)
        return session

    @staticmethod
    def frame(session, offset=0):
        points = np.mgrid[:9, :6].T.reshape(-1, 2).astype(np.float32)*30+[200+offset, 200]
        session.source_received = time.monotonic()
        session.latest = dict(jpeg=b'fixture bytes, not decoded for capture', corners=points.reshape(-1, 1, 2),
            info=dict(center=[.4, .4], bounds=[[200, 200], [440, 350]], area_fraction=.05),
            sequence=str(offset), detected_at=time.monotonic(), valid=True)

    def test_dark_input_and_stale_or_repeated_frames_never_saved(self):
        corners, message, _ = inspect_board(np.zeros((720, 1280, 3), np.uint8), (9, 6))
        self.assertIsNone(corners)
        self.assertIn('过暗', message)
        with tempfile.TemporaryDirectory() as root:
            session = self.session(root)
            self.frame(session)
            session.source_received -= 4
            with self.assertRaises(ValueError): session.capture()
            self.frame(session)
            session.capture()
            with self.assertRaisesRegex(ValueError, '已经保存'): session.capture()
            self.frame(session, 5)
            with self.assertRaisesRegex(ValueError, '太相近'): session.capture()
            self.assertEqual(len(session.samples), 1)

    def test_undo_retains_original_and_holdout_is_not_in_fit_directory(self):
        with tempfile.TemporaryDirectory() as root:
            session = self.session(root)
            self.frame(session)
            session.capture(); original = Path(session.samples[0]['path']).read_bytes()
            session.undo()
            self.assertEqual(len(session.samples), 0)
            self.assertEqual(next((session.out/'unused').iterdir()).read_bytes(), original)
            for i in range(25):
                self.frame(session, i*20)
                session.capture()
            self.assertEqual(len(list((session.out/'fit').glob('*.jpg'))), 20)
            self.assertEqual(len(list((session.out/'validation').glob('*.jpg'))), 5)
            manifest = json.loads((session.out/'capture.json').read_text(encoding='utf-8'))
            self.assertEqual([s['role'] for s in manifest['samples']][-5:], ['validation']*5)
            with self.assertRaises(ValueError): session.capture()
            session.solving = True
            with self.assertRaises(ValueError): session.undo()
            session.solving = False

    def test_post_needs_session_token_and_premature_fit_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            session = self.session(root)
            server = capture_server(session, port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            url = f'http://127.0.0.1:{server.server_port}'
            try:
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(url+'/api/capture', method='POST'))
                self.assertEqual(error.exception.code, 403)
                page = urlopen(url).read().decode('utf-8')
                token = json.loads(page.split('const token=', 1)[1].split(',el=', 1)[0])
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(url+'/api/solve', method='POST', headers={'X-Session-Token': token}))
                self.assertEqual(error.exception.code, 409)
                self.assertIsNone(session.result)
            finally:
                server.shutdown(); server.server_close(); thread.join(3)

    def test_resume_preserves_samples_and_rejects_changed_source_files(self):
        with tempfile.TemporaryDirectory() as root:
            session = self.session(root)
            self.frame(session); session.capture()
            with patch.object(CalibrationCapture, '_poll'):
                resumed = CalibrationCapture(session.source_url, session.out, resume=True)
            try:
                self.assertEqual(resumed.samples, session.samples)
                self.assertEqual(resumed.status()['count'], 1)
            finally: resumed.close()
            Path(session.samples[0]['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, '校验失败'):
                CalibrationCapture(session.source_url, session.out, resume=True)

    def test_holdout_validator_keeps_intrinsics_fixed_and_detects_wrong_aspect(self):
        k = np.array([[800., 0, 639.5], [0, 810., 359.5], [0, 0, 1.]])
        grid = np.zeros((54, 3), np.float32)
        grid[:, :2] = np.mgrid[:9, :6].T.reshape(-1, 2)*20
        observations = [cv2.projectPoints(grid, np.array([.3, -.4, .1]),
                        np.array([-70., -50., 600.]), k, np.zeros(5))[0]]
        record = dict(K=k.tolist(), D=[0]*5, pattern_inner_corners=[9, 6], square_mm=20,
                      image_size=[1280, 720])
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'holdout.png'; cv2.imwrite(str(path), np.zeros((720, 1280), np.uint8))
            with patch('cv2.findChessboardCornersSB', return_value=(True, observations[0])):
                good = validate_fit(record, [path])
                bad_record = dict(record, K=[[1000., 0, 639.5], [0, 700., 359.5], [0, 0, 1.]])
                bad = validate_fit(bad_record, [path])
            self.assertLess(good['mean_rms_px'], 1e-3)
            self.assertGreater(bad['mean_rms_px'], 1.)
            self.assertEqual(record['K'], k.tolist())


if __name__ == '__main__':
    unittest.main()
