"""Resident pre-video readiness and verified stop-to-process handoff contracts.

These tests use fake devices/models; they do not claim physical capture or NPU
execution. Actual event/receipt files and the existing orchestration are used.
"""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from web.camera_ui import (DEFAULT_MAX_SECONDS, DEFAULT_RING_PAIRS, ReconstructionJob,
                                  parse_args, reconstruction_error, ring_pipeline_args)
from initialize.run import main as initialize
from video_input.recording import CameraRecorder


class Preview:
    ready = False
    captures = 0

    def __init__(self, device, width, height, fps, log, **kwargs):
        self.process = SimpleNamespace(poll=lambda: None, returncode=0)
        self.closed = False
        if kwargs:
            if not type(self).ready:
                raise AssertionError('Video opened before retained process READY')
            type(self).captures += 1
            Path(kwargs['recording']).write_bytes(b'closed fake capture bytes')

    def snapshot(self):
        return b'jpeg', 60, 60.

    def finish_recording(self):
        pass

    def close(self):
        self.closed = True


class ResidentCameraTests(unittest.TestCase):
    def job(self, root):
        camera = SimpleNamespace(storage_root=Path(root), video_complete_monotonic=10.,
                                 video_complete_epoch=20.)
        return ReconstructionJob(camera, SimpleNamespace(out=Path(root), backend='npu'))

    def wait_state(self, camera, expected):
        deadline = time.monotonic()+3
        while camera.state != expected and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertEqual(camera.state, expected, camera.error)

    def test_ring_auto_minimum_group_count_reaches_pipeline(self):
        args=SimpleNamespace(ring_pairs=8,ring_candidates=64,ring_max_gaussians=65536,
                             ring_auto_coverage=True,ring_coverage_target=1.,ring_min_pairs=8)
        self.assertEqual(ring_pipeline_args(args),[
            '--ring-pairs','8','--ring-candidates','64','--ring-max-gaussians','65536',
            '--ring-auto-coverage','--ring-coverage-target','1.0','--ring-min-pairs','8'])

    def test_default_capture_uses_eight_sparse_ring_pairs(self):
        required=['--out','sessions','--weights','weights','--vendor','vendor',
                  '--renderer','renderer','--live-renderer','live','--focal-ratio','.525',
                  '--viewer-public-url','http://127.0.0.1:18434/']
        _,args=parse_args(required)
        self.assertEqual(DEFAULT_RING_PAIRS,8)
        self.assertEqual(DEFAULT_MAX_SECONDS,15)
        self.assertEqual(args.max_seconds,15)
        self.assertEqual(args.backend,'npu')
        self.assertEqual(args.ring_candidates,64)
        self.assertTrue(args.ring_auto_coverage)
        self.assertEqual(args.ring_coverage_target,1.)
        self.assertEqual(args.ring_min_pairs,8)
        self.assertEqual(ring_pipeline_args(args),[
            '--ring-pairs','8','--ring-candidates','64','--ring-max-gaussians','65536',
            '--ring-auto-coverage','--ring-coverage-target','1.0','--ring-min-pairs','8'])
        with self.assertRaisesRegex(ValueError,'Non-CPU backend needs a bundle'):
            from gaussian_generation.npu_branch.config import BackendConfig
            BackendConfig.from_args(args)

    def test_camera_cpu_path_requires_explicit_selection(self):
        required=['--out','sessions','--weights','weights','--vendor','vendor',
                  '--renderer','renderer','--live-renderer','live','--focal-ratio','.525',
                  '--viewer-public-url','http://127.0.0.1:18434/']
        _,args=parse_args(required+['--backend','cpu'])
        self.assertEqual(args.backend,'cpu')

    def test_ring_selection_mode_can_be_overridden_or_disabled(self):
        required=['--out','sessions','--weights','weights','--vendor','vendor',
                  '--renderer','renderer','--live-renderer','live','--focal-ratio','.525',
                  '--viewer-public-url','http://127.0.0.1:18434/']
        _,fixed=parse_args(required+['--ring-pairs','2','--ring-fixed-pairs'])
        self.assertFalse(fixed.ring_auto_coverage)
        self.assertEqual(fixed.ring_min_pairs,1)
        _,ordinary=parse_args(required+['--ring-pairs','0'])
        self.assertFalse(ordinary.ring_auto_coverage)
        self.assertEqual(ordinary.ring_min_pairs,1)
        self.assertEqual(ring_pipeline_args(ordinary),[])

    def test_eight_fixed_pairs_are_rejected_before_model_warmup(self):
        required=['--out','sessions','--weights','weights','--vendor','vendor',
                  '--renderer','renderer','--live-renderer','live','--focal-ratio','.525']
        for entry in (parse_args, initialize):
            with self.subTest(entry=entry.__name__), redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit):
                    flags=(['--ring-fixed-pairs','--viewer-public-url','http://127.0.0.1:18434/']
                           if entry is parse_args else [])
                    entry(required+['--backend','cpu','--ring-pairs','8']+flags)
                self.assertIn('eight panorama pairs require automatic coverage', errors.getvalue())

    def test_ring_failure_guidance_matches_fixed_center_capture(self):
        message=reconstruction_error('No panorama pair with real cropped-frustum overlap',
                                     ring_mode=True)
        self.assertIn('固定拍摄中心',message)
        self.assertIn('水平缓慢转满一圈',message)

    def test_warmed_handoff_reuses_process_and_writes_one_path(self):
        with tempfile.TemporaryDirectory() as folder:
            job = self.job(folder)
            process = SimpleNamespace(poll=lambda:None, stdin=io.BytesIO())
            job.process, job.phase, job.warm_pending = process, 'warmed', True
            with patch('web.camera_ui.threading.Thread') as worker:
                job.start(Path(folder)/'capture.mkv')
                worker.assert_not_called()
            self.assertIs(job.process, process)
            self.assertEqual(process.stdin.getvalue(), (str(Path(folder)/'capture.mkv')+'\n').encode())
            self.assertFalse(job.warm_pending)
            self.assertEqual(job.phase, 'reconstructing')
            self.assertEqual(job.processing_started, 10.)
            with self.assertRaisesRegex(ValueError, '已有重建'):
                job.start(Path(folder)/'capture.mkv')
            self.assertEqual(process.stdin.getvalue().count(b'\n'), 1)

    def test_warmed_handoff_rejects_dead_process_or_injected_line(self):
        with tempfile.TemporaryDirectory() as folder:
            job = self.job(folder)
            job.process = SimpleNamespace(poll=lambda:1, stdin=io.BytesIO())
            job.phase, job.warm_pending = 'warmed', True
            with self.assertRaisesRegex(RuntimeError, '预热进程已退出'):
                job.start('clip')
            job.process.poll = lambda:None
            with self.assertRaisesRegex(ValueError, '无效录像路径'):
                job.start('clip\nother')
            self.assertEqual(job.process.stdin.getvalue(), b'')

    def test_repeated_prime_does_not_reload_already_warmed_resources(self):
        with tempfile.TemporaryDirectory() as folder:
            job = self.job(folder)
            job.phase = 'warmed'
            with patch('web.camera_ui.threading.Thread') as worker:
                job.prime()
                worker.assert_not_called()

    def test_ui_latency_freezes_at_frame_event(self):
        with tempfile.TemporaryDirectory() as folder:
            job = self.job(folder)
            job.phase, job.processing_started, job.processing_ended = 'ready', 10., 16.25
            job.processing_complete = True
            with patch('web.camera_ui.time.monotonic', return_value=100.):
                self.assertEqual(job.status()['processing_elapsed_seconds'], 6.25)

    def test_initialize_waits_for_video_after_ready_and_keeps_one_session(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            weights, video, run = root/'weights', root/'clip', root/'scene'
            weights.write_bytes(b'fake checkpoint, never loaded')
            video.write_bytes(b'closed video')
            model, renderer = object(), object()
            session = SimpleNamespace(record={'loaded_once':True}, model=model, renderer=renderer,
                                      close=lambda:None)
            def accept_input():
                events = json.loads((run/'events.json').read_text())['events']
                self.assertEqual([e['event'] for e in events], ['READY'])
                return str(video)
            def prepare(path, out, actual_model, actual_renderer, events, **kwargs):
                self.assertIs(actual_model, model)
                self.assertIs(actual_renderer, renderer)
                events.emit('SCENE_READY')
                events.emit('FRAME_COMPLETE')
                return {'frame':{'complete':True}}
            with patch('initialize.run.platform.machine', return_value='aarch64'), \
                    patch('initialize.session.WarmSession', return_value=session) as warm, \
                    patch('builtins.input', side_effect=accept_input), \
                    patch('initialize.pipeline.prepare_scene', side_effect=prepare), redirect_stdout(io.StringIO()):
                initialize(['--out', str(run), '--weights', str(weights),
                            '--vendor', 'unused', '--renderer', 'unused'])
            warm.assert_called_once()
            record = json.loads((run/'warm_result.json').read_text())
            self.assertTrue(record['complete'])
            self.assertEqual([e['event'] for e in record['events']],
                             ['READY','VIDEO_COMPLETE','SCENE_READY','FRAME_COMPLETE'])

    def test_capture_opens_only_after_ready_and_auto_submits_once(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch('video_input.recording.inspect_clip', return_value=(
                    dict(codec_name='mjpeg', width=1280, height=720, fps=60),
                    [i/60 for i in range(121)])):
            Preview.ready, Preview.captures = False, 0
            camera = CameraRecorder('/dev/video0', 1280, 720, 60, folder,
                                    factory=Preview, auto_process=True)
            submitted = []
            def ready():
                Preview.ready = True
            build = SimpleNamespace(status=lambda:{'phase':'warmed'}, prepare_capture=ready,
                start=lambda path:submitted.append(path), close=lambda:None)
            camera.build = build
            try:
                camera.start()
                self.wait_state(camera, 'recording')
                with camera.lock:
                    camera.stop()
                    camera.stop()  # Repeated stop cannot schedule a second inference.
                camera.worker.join(3)
                self.assertFalse(camera.worker.is_alive())
                self.assertEqual(camera.state, 'recorded', camera.error)
                self.assertEqual(Preview.captures, 1)
                self.assertEqual(submitted, [camera.clip['path']])
                self.assertTrue(Path(submitted[0]).is_file())
                self.assertIsNotNone(camera.video_complete_monotonic)
            finally:
                camera.close()

    def test_invalid_clip_never_submits_to_warmed_process(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch('video_input.recording.inspect_clip', return_value=(
                    dict(codec_name='mjpeg', width=1280, height=720, fps=60),
                    [i/60 for i in range(10)])):
            Preview.ready = False
            camera = CameraRecorder('/dev/video0', 1280, 720, 60, folder,
                                    factory=Preview, auto_process=True)
            submitted = []
            def ready():
                Preview.ready = True
            camera.build = SimpleNamespace(status=lambda:{'phase':'warmed'}, prepare_capture=ready,
                start=lambda path:submitted.append(path), close=lambda:None)
            try:
                camera.start()
                self.wait_state(camera, 'recording')
                camera.stop()
                camera.worker.join(3)
                self.assertEqual(camera.state, 'error')
                self.assertEqual(submitted, [])
            finally:
                camera.close()


if __name__ == '__main__':
    unittest.main()
