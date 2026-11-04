"""Orchestration regressions with fake model/device; actual receipt/event files."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from initialize.run import main
from video_input.test_camera import calibration_file


class CameraPipelineTests(unittest.TestCase):
    def test_failed_global_sampling_uses_local_span_and_keeps_original_indices(self):
        from initialize.scene_preparation import main as prepare
        image=np.zeros((270,480,3),np.uint8)
        pose=np.eye(4)
        pair=(pose,{},[],None,dict(translation_scale=1.))
        calls=[]
        def decode(*args,**kwargs):
            span=kwargs.get('candidate_span');calls.append(span)
            keys=[0,500,999] if span is None else [760,879,999]
            return {i:image for i in keys},1000,60,[keys[0],keys[-1]],[keys[1]]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);video=root/'input.mkv';video.write_bytes(b'input')
            with patch('initialize.scene_preparation.decode_selected',side_effect=decode),patch(
                    'initialize.scene_preparation.select_views',side_effect=[ValueError('No valid context pair'),
                    ([760,999],[879],pair,{879:(pose,dict(pnp_inliers=80))},dict(kind='test'))]),redirect_stdout(io.StringIO()):
                prepare(['--video',str(video),'--out',str(root/'out'),'--auto-select'])
            record=json.loads((root/'out/input.json').read_text(encoding='utf-8'))
            self.assertEqual(calls,[None,(760,999)])
            self.assertEqual(record['context_indices'],[760,999])
            self.assertEqual(record['target_indices'],[879])
            self.assertEqual(record['source_frames'],1000)
            self.assertEqual(record['frame_selection']['sampling']['selected_span'],[760,999])
            self.assertEqual(len(record['frame_selection']['sampling']['prior_failed_attempts']),1)

    def test_camera_first_frame_then_viewer_then_release(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            weights = root/"weights"
            weights.write_bytes(b"fake test model, never loaded")
            video = root/"camera.avi"
            video.write_bytes(b"closed test clip, decode covered separately")
            cal = calibration_file(root)
            session = SimpleNamespace(record={}, model=object(), renderer=SimpleNamespace(runtime=object()), closed=False)
            session.close = lambda: setattr(session, "closed", True)
            def capture(device, out, **kwargs):
                kwargs["events"].emit("CAPTURE_STARTED")
                kwargs["events"].emit("CAPTURE_COMPLETE")
                return dict(path=str(video), complete=True)
            def prepare(path, out, model, renderer, events, **kwargs):
                self.assertEqual(path, str(video.resolve()))
                self.assertEqual(kwargs["calibration"], cal)
                self.assertTrue(kwargs["auto_select"])
                events.emit("SCENE_READY")
                events.emit("FRAME_COMPLETE")
                return dict(frame={"complete": True})
            def viewer(*args, **kwargs):
                self.assertFalse(session.closed)
                return dict(frames=3)
            with patch("initialize.run.platform.machine", return_value="aarch64"), \
                    patch("initialize.session.WarmSession", return_value=session), \
                    patch("video_input.capture.capture_clip", side_effect=capture), \
                    patch("initialize.pipeline.prepare_scene", side_effect=prepare), \
                    patch("web.viewer.serve_scene", side_effect=viewer), redirect_stdout(io.StringIO()):
                main(["--camera", "/dev/video0", "--camera-width", "640", "--camera-height", "360",
                      "--capture-now", "--calibration", str(cal), "--out", str(root/"run"),
                      "--weights", str(weights), "--vendor", "unused", "--renderer", "unused",
                      "--live-renderer", "unused", "--interactive"])
            result = json.loads((root/"run/warm_result.json").read_text())
            self.assertTrue(result["complete"])
            self.assertEqual(result["interactive"]["frames"], 3)
            self.assertTrue(session.closed)
            self.assertEqual([event["event"] for event in result["events"]],
                ["READY", "CAPTURE_STARTED", "CAPTURE_COMPLETE", "VIDEO_COMPLETE", "SCENE_READY", "FRAME_COMPLETE"])

    def test_camera_requires_calibration_before_any_warmup(self):
        with patch("initialize.session.WarmSession") as warm, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(["--camera", "/dev/video0", "--out", "unused", "--weights", "unused",
                      "--vendor", "unused", "--renderer", "unused"])
            self.assertEqual(error.exception.code, 2)
            warm.assert_not_called()

    def test_calibrated_preparation_publishes_centered_network_and_camera(self):
        from initialize.scene_preparation import main as prepare
        image = np.zeros((270, 480, 3), np.uint8)
        k = np.array([[300., 0, 239.5], [0, 310., 134.5], [0, 0, 1.]])
        pose = np.eye(4)
        pair = (pose, {}, [], None, dict(translation_scale=2.))
        def decoded(*args, **kwargs):
            cal = kwargs["calibration"]
            cal.intrinsic, cal.output_size = k, (480, 270)
            return {0: image, 1: image, 2: image}, 3, 30, [0, 2], [1]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root/"camera.avi"
            video.write_bytes(b"input")
            cal = calibration_file(root)
            with patch("initialize.scene_preparation.decode_selected", side_effect=decoded), \
                    patch("initialize.scene_preparation.select_views", return_value=([0, 2], [1], pair,
                        {1: (pose, {"pnp_inliers": 100})}, {"kind": "test"})), redirect_stdout(io.StringIO()):
                prepare(["--video", str(video), "--out", str(root/"out"), "--calibration", str(cal), "--auto-select"])
            with np.load(root/"out/context.npz") as context:
                expected = [[300/270, 0, .5], [0, 310/270, .5], [0, 0, 1]]
                np.testing.assert_allclose(context["intrinsics"], [expected, expected], atol=1e-7)
                self.assertEqual(context["image"].shape, (2, 3, 128, 128))
            meta = json.loads((root/"out/input.json").read_text())
            from gaussian_generation.adapter import camera_bytes
            self.assertEqual(len(camera_bytes(meta["views"][-1])), 136)


if __name__ == "__main__":
    unittest.main()
