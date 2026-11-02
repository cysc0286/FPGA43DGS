"""Offline camera contracts, using real MJPEG and OpenCV geometry where possible."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from common import save
from initialize.events import EventLog
from video_input.calibration import Calibration, calibrate
from video_input.capture import capture_clip, inspect_clip, validate_capture
from video_input.decode import decode_selected


def calibration_file(root, width=640, height=360):
    path = Path(root)/"camera.json"
    save(path, dict(schema="uvc_pinhole_v1", image_size=[width, height],
                   K=[[400, 0, width/2+11], [0, 410, height/2-7], [0, 0, 1]],
                   D=[-.12, .025, .001, -.002, 0]))
    return path


class CalibrationTests(unittest.TestCase):
    def test_rectification_centers_an_off_axis_camera_and_preserves_rays(self):
        with tempfile.TemporaryDirectory() as folder:
            cal = Calibration(calibration_file(folder))
            image = np.zeros((360, 640, 3), np.uint8)
            corrected = cal.rectify(image, 480)
            self.assertEqual(corrected.shape, (270, 480, 3))
            np.testing.assert_allclose(cal.intrinsic[:2, 2], [239.5, 134.5])
            # Independent projection into the original distorted camera must match
            # the remap at selected rectified pixel coordinates (resize half-pixels).
            pixels = np.array([[100., 80.], [240., 135.], [380., 200.]])
            rays = (np.linalg.inv(cal.intrinsic) @ np.c_[pixels, np.ones(3)].T).T
            projected, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), cal.k, cal.d)
            projected = (projected.reshape(-1, 2)+.5)*.75-.5
            mapped = np.array([[cal.maps[0][int(y), int(x)], cal.maps[1][int(y), int(x)]] for x, y in pixels])
            np.testing.assert_allclose(mapped, projected, atol=3e-5)
            with self.assertRaisesRegex(ValueError, "recalibrate"):
                cal.rectify(np.zeros((480, 640, 3), np.uint8), 480)

    def test_rejects_nonfinite_or_invalid_calibration(self):
        with tempfile.TemporaryDirectory() as folder:
            path = calibration_file(folder)
            record = json.loads(path.read_text())
            for field, value in (("K", [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]), ("D", [0, 0])):
                bad = dict(record, **{field: value})
                save(path, bad)
                with self.assertRaises(ValueError):
                    Calibration(path)

    def test_too_few_calibration_images_never_publish_a_fit(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"calibration.json"
            with self.assertRaisesRegex(ValueError, "at least 10"):
                calibrate(folder, (9, 6), 25., output)
            self.assertFalse(output.exists())

    def test_calibration_fit_from_projected_chessboard_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root/"images"
            images.mkdir()
            k = np.array([[800., 0, 639.5], [0, 810., 359.5], [0, 0, 1.]])
            rng = np.random.default_rng(31)
            for i in range(12):
                image = np.full((720, 1280), 160, np.uint8)
                rotation = rng.uniform([-.45, -.5, -.2], [.45, .5, .2])
                translation = np.array([rng.uniform(-180, 20), rng.uniform(-140, 0), rng.uniform(600, 900)])
                for y in range(7):
                    for x in range(10):
                        corners = np.float64([[x*25, y*25, 0], [(x+1)*25, y*25, 0],
                                              [(x+1)*25, (y+1)*25, 0], [x*25, (y+1)*25, 0]])
                        projected, _ = cv2.projectPoints(corners, rotation, translation, k, None)
                        cv2.fillConvexPoly(image, np.round(projected[:, 0]*16).astype(np.int32),
                                           255 if (x+y)%2 else 0, lineType=cv2.LINE_AA, shift=4)
                self.assertTrue(cv2.imwrite(str(images/f"board_{i:02d}.png"), image))
            result = calibrate(images, (9, 6), 25., root/"fit.json")
            self.assertEqual(len(result["images"]), 12)
            self.assertLess(result["rms_px"], .5)
            np.testing.assert_allclose(np.array(result["K"])[[0, 1], [0, 1]], [800, 810], rtol=.03)
            Calibration(root/"fit.json")


class CaptureLifecycleTests(unittest.TestCase):
    def test_capture_events_keep_existing_first_frame_clock(self):
        with tempfile.TemporaryDirectory() as folder:
            events = EventLog(Path(folder)/"events.json")
            with patch("initialize.events.time.monotonic", side_effect=[0, 2, 8, 9, 20, 21]):
                for event in ("READY", "CAPTURE_STARTED", "CAPTURE_COMPLETE", "VIDEO_COMPLETE", "SCENE_READY", "FRAME_COMPLETE"):
                    events.emit(event)
            self.assertEqual(events.summary()["scene_preparation_seconds"], 12)
            self.assertEqual(events.summary()["capture_seconds_record_only"], 6)
            self.assertEqual(events.summary()["capture_start_to_frame_seconds"], 19)

    def test_incomplete_capture_cannot_publish_video_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            events = EventLog(Path(folder)/"events.json")
            events.emit("READY")
            events.emit("CAPTURE_STARTED")
            for name in ("CAPTURE_STARTED", "VIDEO_COMPLETE", "FRAME_COMPLETE"):
                with self.assertRaises(ValueError):
                    events.emit(name)

    def test_invalid_limits_and_source_rejected_without_starting_process(self):
        for fps, seconds in ((0, 5), (30, float("nan")), (30, 61)):
            with self.assertRaises(ValueError):
                validate_capture(1280, 720, fps, seconds, 1)
        with tempfile.TemporaryDirectory() as folder, patch("video_input.capture.subprocess.run") as run:
            with self.assertRaises(ValueError):
                capture_clip("http://example.invalid/video", Path(folder)/"capture")
            run.assert_not_called()

    def test_ffmpeg_failure_retains_diagnostics_and_does_not_publish_clip(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/"capture"
            with patch("video_input.capture.Path.exists", return_value=True), \
                    patch("video_input.capture.subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 30)):
                with self.assertRaises(subprocess.TimeoutExpired):
                    capture_clip("/dev/video0", out)
            self.assertFalse((out/"capture.mkv").exists())
            self.assertFalse(json.loads((out/"capture.json").read_text())["complete"])


@unittest.skipUnless(shutil.which("ffprobe") and shutil.which("ffmpeg"), "FFmpeg tools required")
class RealClipTests(unittest.TestCase):
    def test_mjpeg_capture_file_is_consumable_by_both_input_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"camera.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 360))
            self.assertTrue(writer.isOpened())
            try:
                for i in range(10):
                    writer.write(np.full((360, 640, 3), i*20, np.uint8))
            finally:
                writer.release()
            actual, timestamps = inspect_clip(path)
            self.assertEqual((actual["codec_name"], len(timestamps)), ("mjpeg", 10))
            old, count, fps, context, targets = decode_selected(path, [0, -1], None, 480, 1)
            self.assertEqual((context, targets), ([0, 9], [2, 5, 7]))
            cal = Calibration(calibration_file(folder))
            candidates, *_ = decode_selected(path, [0, -1], None, 480, 1, calibration=cal, candidate_count=8)
            self.assertEqual(len(candidates), 8)
            self.assertEqual(next(iter(candidates.values())).shape, (270, 480, 3))
            self.assertEqual((count, fps), (10, 10))
            self.assertEqual(len(old), 5)

    def test_capture_handoff_with_real_ffmpeg_muxing_and_probe(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/"source.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 360))
            self.assertTrue(writer.isOpened())
            for i in range(10):
                writer.write(np.full((360, 640, 3), i*20, np.uint8))
            writer.release()
            actual_run = subprocess.run
            def replace_uvc_only(command, **kwargs):
                if command[0] == "ffmpeg":
                    # Replace only the hardware source; exercise the real muxer,
                    # probe, finalization, file receipt and monotonic timestamps.
                    command = ["ffmpeg", "-hide_banner", "-nostdin", "-n", "-i", str(source),
                               "-t", "1", "-c:v", "copy", command[-1]]
                return actual_run(command, **kwargs)
            log = EventLog(root/"events.json")
            log.emit("READY")
            with patch("video_input.capture.Path.exists", return_value=True), \
                    patch("video_input.capture.subprocess.run", side_effect=replace_uvc_only):
                result = capture_clip("/dev/video0", root/"capture", width=640, height=360,
                                      fps=10, seconds=1, settle=0, events=log)
            self.assertTrue(result["complete"])
            self.assertEqual(result["frames"], 10)
            self.assertAlmostEqual(result["actual"]["measured_fps"], 10)
            self.assertTrue(Path(result["path"]).is_file())
            self.assertFalse((root/"capture/capture.partial.mkv").exists())

    def test_textured_multidepth_video_to_calibrated_context_with_real_sift(self):
        from contextlib import redirect_stdout
        import io
        from scene_preparation import main as prepare
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rng = np.random.default_rng(123)
            centers = [(x, y) for y in range(50, 320, 60) for x in range(70, 600, 65)]
            depths = rng.uniform(3, 8, len(centers))
            patches = [rng.integers(0, 256, (32, 32, 3), np.uint8) for _ in centers]
            video = root/"translated_camera.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 360))
            self.assertTrue(writer.isOpened())
            try:
                for baseline in np.linspace(0, .4, 8):
                    image = np.full((360, 640, 3), 100, np.uint8)
                    for (x, y), depth, texture in zip(centers, depths, patches):
                        left, top = round(x-400*baseline/depth-16), y-16
                        image[top:top+32, left:left+32] = texture
                    writer.write(image)
            finally:
                writer.release()
            calibration = root/"calibration.json"
            save(calibration, dict(schema="uvc_pinhole_v1", image_size=[640, 360],
                K=[[400, 0, 319.5], [0, 400, 179.5], [0, 0, 1]], D=[0]*5))
            with redirect_stdout(io.StringIO()):
                prepare(["--video", str(video), "--out", str(root/"prepared"),
                         "--calibration", str(calibration), "--auto-select"])
            result = json.loads((root/"prepared/input.json").read_text())
            self.assertEqual(len(set(result["context_indices"]+result["target_indices"])), 3)
            self.assertGreaterEqual(result["geometry"]["pair"]["median_parallax_deg"], 1)
            target = result["geometry"]["targets"][str(result["target_indices"][0])]
            self.assertLessEqual(target["median_reprojection_px"], 2)
            with np.load(root/"prepared/context.npz") as arrays:
                self.assertEqual(arrays["image"].shape, (2, 3, 128, 128))
                np.testing.assert_allclose(arrays["intrinsics"][:, :2, 2], .5)


if __name__ == "__main__":
    unittest.main()
