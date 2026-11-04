"""Dual-input geometry, one-owner replacement and FRAME_COMPLETE timing.

Subprocess/thread objects are controlled fakes. No device, model or board opens.
"""
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from web.camera_ui import ReconstructionJob, input_geometry_args, parse_args, ring_pipeline_args


class InputModeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        _, self.args = parse_args(['--out', str(self.root), '--weights', 'weights',
            '--vendor', 'vendor', '--renderer', 'renderer', '--live-renderer', 'live',
            '--calibration', 'usb-only.json', '--upload-focal-ratio', '.63',
            '--viewer-public-url', 'http://127.0.0.1:18434/', '--backend', 'cpu'])
        self.camera = SimpleNamespace(storage_root=self.root, video_complete_monotonic=10.,
            video_complete_epoch=20., input_kind='usb_camera', pause_preview=Mock(), resume_preview=Mock())
        self.job = ReconstructionJob(self.camera, self.args)

    def test_default_capture_limit_is_fifteen_seconds(self):
        self.assertEqual(self.args.max_seconds, 15.)

    def test_weak_texture_option_is_explicit_in_pipeline_command(self):
        self.assertNotIn('--ring-feature-contrast', ring_pipeline_args(self.args))
        self.args.ring_feature_contrast=.01
        cli=ring_pipeline_args(self.args)
        self.assertEqual(cli[cli.index('--ring-feature-contrast')+1], '0.01')

    def test_explicit_capture_limit_is_preserved(self):
        for limit in (7.5, 60.):
            with self.subTest(seconds=limit):
                _, args = parse_args(['--out', str(self.root), '--weights', 'weights',
                    '--vendor', 'vendor', '--renderer', 'renderer', '--live-renderer', 'live',
                    '--calibration', 'usb-only.json', '--backend', 'cpu',
                    '--viewer-public-url', 'http://127.0.0.1:18434/',
                    '--max-seconds', str(limit)])
                self.assertEqual(args.max_seconds, limit)

    def test_geometry_is_explicit_and_upload_never_inherits_usb_intrinsics(self):
        self.assertEqual(input_geometry_args(self.args, 'usb_camera'), ['--calibration', 'usb-only.json'])
        self.assertEqual(input_geometry_args(self.args, 'uploaded_video'), ['--focal-ratio', '0.63'])
        self.args.calibration = None
        self.args.focal_ratio = .72
        self.assertEqual(input_geometry_args(self.args, 'usb_camera'), ['--focal-ratio', '0.72'])
        self.assertEqual(input_geometry_args(self.args, 'uploaded_video'), ['--focal-ratio', '0.63'])
        with self.assertRaises(ValueError):
            input_geometry_args(self.args, 'unknown')

    def test_same_mode_ready_owner_is_reused_without_second_warmup(self):
        self.job.phase, self.job.input_kind = 'warmed', 'uploaded_video'
        original = self.job.cancel
        with patch('web.camera_ui.threading.Thread') as thread:
            self.job.prime(input_kind='uploaded_video')
        thread.assert_not_called()
        self.assertIs(self.job.cancel, original)
        self.assertFalse(original.is_set())

    def test_switching_mode_cancels_previous_owner_and_joins_it_in_new_worker(self):
        for before, after in (('usb_camera', 'uploaded_video'), ('uploaded_video', 'usb_camera')):
            with self.subTest(before=before):
                self.job.phase, self.job.input_kind = 'warmed', before
                previous = Mock()
                self.job.worker = previous
                old_cancel = self.job.cancel
                self.job.url, self.job.processing_complete = 'http://old-result/', True
                self.job.processing_started, self.job.processing_ended = 10., 12.
                replacement = Mock()
                with patch('web.camera_ui.threading.Thread', return_value=replacement) as thread:
                    self.job.prime(input_kind=after)
                self.assertTrue(old_cancel.is_set())
                self.assertFalse(self.job.cancel.is_set())
                self.assertIsNot(old_cancel, self.job.cancel)
                self.assertEqual(thread.call_args.kwargs['args'], (None, self.job.cancel, previous))
                replacement.start.assert_called_once()
                self.assertEqual(self.job.input_kind, after)
                self.assertEqual(self.job.phase, 'initializing')
                self.assertIsNone(self.job.url)
                self.assertIsNone(self.job.processing_started)
                self.assertFalse(self.job.processing_complete)

    def test_switching_input_while_busy_and_invalid_kind_fail_before_thread(self):
        for phase in ('initializing', 'reconstructing', 'streaming', 'stopping'):
            with self.subTest(phase=phase), patch('web.camera_ui.threading.Thread') as thread:
                self.job.phase = phase
                with self.assertRaises(ValueError):
                    self.job.prime(input_kind='uploaded_video')
                thread.assert_not_called()
        self.job.phase = 'warmed'
        with patch('web.camera_ui.threading.Thread') as thread:
            with self.assertRaises(ValueError):
                self.job.prime(input_kind='unknown')
            with self.assertRaises(ValueError):
                self.job.start('video', input_kind='unknown')
            thread.assert_not_called()

    def test_matching_warm_owner_uses_one_stdin_handoff_and_preserves_input_clock(self):
        self.job.phase, self.job.input_kind, self.job.warm_pending = 'warmed', 'uploaded_video', True
        process = SimpleNamespace(poll=lambda: None, stdin=io.BytesIO())
        self.job.process = process
        with patch('web.camera_ui.threading.Thread') as thread:
            self.job.start('uploaded.mp4', input_kind='uploaded_video')
        thread.assert_not_called()
        self.assertEqual(process.stdin.getvalue(), b'uploaded.mp4\n')
        self.assertEqual(self.job.processing_started, 10.)
        self.assertEqual(self.job.processing_started_epoch, 20.)
        self.assertEqual(self.job.phase, 'reconstructing')
        self.assertFalse(self.job.processing_complete)
        with self.assertRaises(ValueError):
            self.job.start('uploaded.mp4', input_kind='uploaded_video')
        self.assertEqual(process.stdin.getvalue(), b'uploaded.mp4\n')

    def test_mismatched_warm_owner_is_not_given_video_under_wrong_geometry(self):
        self.job.phase, self.job.input_kind, self.job.warm_pending = 'warmed', 'usb_camera', True
        old_process = SimpleNamespace(poll=lambda: None, stdin=io.BytesIO())
        self.job.process = old_process
        previous = Mock()
        self.job.worker = previous
        old_cancel = self.job.cancel
        with patch('web.camera_ui.threading.Thread') as thread:
            self.job.start('uploaded.mp4', input_kind='uploaded_video')
        self.assertEqual(old_process.stdin.getvalue(), b'')
        self.assertTrue(old_cancel.is_set())
        self.assertEqual(self.job.input_kind, 'uploaded_video')
        self.assertEqual(thread.call_args.kwargs['args'][:3], ('uploaded.mp4', self.job.cancel, previous))
        self.assertFalse(self.job.warm_pending)

    def test_new_and_restored_usb_stream_reset_previous_upload_mode_and_clock(self):
        from common import sha
        folder = self.root/'stream_20261103_090000_abcdef'
        frame = folder/'slot_a/board/first_frame/frame.bin'
        frame.parent.mkdir(parents=True)
        frame.write_bytes(b'local stream fixture')
        (folder/'latest.json').write_text(json.dumps(dict(slot='slot_a', frame_sha256=sha(frame))))
        for restore in (False, True):
            with self.subTest(restore=restore):
                self.job.phase, self.job.input_kind, self.job.warm_pending = 'ready', 'uploaded_video', True
                self.job.processing_started, self.job.processing_started_epoch = 10., 20.
                self.job.processing_ended, self.job.processing_complete = 18., True
                self.job.url = 'http://old-upload/'
                with patch('web.camera_ui.threading.Thread') as thread:
                    if restore:
                        self.job.restore_stream(folder)
                    else:
                        self.job.start_stream()
                thread.return_value.start.assert_called_once()
                self.assertEqual(self.job.input_kind, 'usb_camera')
                self.assertFalse(self.job.warm_pending)
                self.assertIsNone(self.job.processing_started)
                self.assertIsNone(self.job.processing_started_epoch)
                self.assertIsNone(self.job.processing_ended)
                self.assertFalse(self.job.processing_complete)
                self.assertIsNone(self.job.url)
                self.assertEqual(self.job.mode, 'streaming')

    def test_replacement_waits_for_previous_worker_and_stops_process_before_launch(self):
        order = []
        previous = SimpleNamespace(join=lambda seconds: order.append('join'), is_alive=lambda: False)
        run = self.root/'scene_fake'
        run.mkdir()
        (run/'warm_result.json').write_text('{"complete":true}')
        Path(str(run)+'.measurement.json').write_text('{"status":"passed"}')
        self.job.run_path, self.job.input_kind = str(run), 'uploaded_video'
        def launch(command, **kwargs):
            order.append('launch')
            self.assertEqual(order, ['join', 'stop', 'launch'])
            self.assertEqual(kwargs['cwd'], Path(__file__).resolve().parents[1])
            self.assertTrue(Path(command[1]).is_file(), 'memory monitor must move with the CLI')
            self.assertNotIn('--calibration', command)
            self.assertEqual(command[command.index('--focal-ratio')+1], '0.63')
            return SimpleNamespace(poll=lambda: 0, returncode=0)
        with patch.object(self.job, 'stop_viewer', side_effect=lambda: order.append('stop')), \
                patch('web.camera_ui.subprocess.Popen', side_effect=launch) as popen:
            self.job.run('uploaded.mp4', threading.Event(), previous)
        popen.assert_called_once()
        self.assertEqual(order, ['join', 'stop', 'launch'])
        self.assertEqual(self.job.phase, 'complete')
        self.camera.pause_preview.assert_called_once()

    def test_previous_worker_that_wont_exit_prevents_second_process(self):
        previous = SimpleNamespace(join=lambda seconds: None, is_alive=lambda: True)
        self.job.run_path = str(self.root/'scene_fake')
        with patch.object(self.job, 'stop_viewer') as stop, \
                patch('web.camera_ui.subprocess.Popen') as popen:
            self.job.run(None, threading.Event(), previous)
        popen.assert_not_called()
        stop.assert_called_once()
        self.assertEqual(self.job.phase, 'error')
        self.assertIn('上次查看器尚未退出', self.job.error_details)

    def test_viewer_ready_does_not_end_timer_before_frame_complete(self):
        run = self.root/'scene_events'
        run.mkdir()
        events = [dict(event='READY', monotonic=1.), dict(event='VIDEO_COMPLETE', monotonic=10.),
                  dict(event='SCENE_READY', monotonic=17.)]
        (run/'events.json').write_text(json.dumps(dict(events=events)))
        self.job.run_path, self.job.input_kind = str(run), 'uploaded_video'
        self.job.processing_started, self.job.processing_started_epoch = 10., 20.
        self.job.clip_path = 'uploaded.mp4'
        cancel, observations = threading.Event(), []
        def launch(command, **kwargs):
            kwargs['stdout'].write((json.dumps(dict(event='VIEWER_READY',
                url='http://127.0.0.1:8080/#token=new-view'))+'\n').encode())
            kwargs['stdout'].flush()
            return SimpleNamespace(poll=lambda: None, returncode=None)
        def advance(seconds):
            observations.append(self.job.status())
            if len(observations) == 1:
                events.append(dict(event='FRAME_COMPLETE', monotonic=18.25))
                (run/'events.json').write_text(json.dumps(dict(events=events)))
            else:
                cancel.set()
        with patch.object(self.job, 'stop_viewer'), \
                patch('web.camera_ui.subprocess.Popen', side_effect=launch), \
                patch('web.camera_ui.time.monotonic', return_value=100.), \
                patch('web.camera_ui.time.sleep', side_effect=advance):
            self.job.run(None, cancel)
        self.assertEqual(len(observations), 2)
        self.assertEqual(observations[0]['phase'], 'ready')
        self.assertFalse(observations[0]['processing_complete'])
        self.assertEqual(observations[0]['processing_elapsed_seconds'], 90.)
        self.assertTrue(observations[1]['processing_complete'])
        self.assertEqual(observations[1]['processing_elapsed_seconds'], 8.25)
        with patch('web.camera_ui.time.monotonic', return_value=200.):
            self.assertEqual(self.job.status()['processing_elapsed_seconds'], 8.25)
        self.assertEqual(self.job.url, 'http://127.0.0.1:18434/#token=new-view')

    def test_uploaded_viewer_success_survives_absent_usb_at_ready_and_finally(self):
        run = self.root/'scene_no_usb'
        run.mkdir()
        (run/'events.json').write_text(json.dumps(dict(events=[
            dict(event='READY', monotonic=1.), dict(event='VIDEO_COMPLETE', monotonic=10.),
            dict(event='SCENE_READY', monotonic=17.), dict(event='FRAME_COMPLETE', monotonic=18.)])))
        self.job.run_path, self.job.input_kind = str(run), 'uploaded_video'
        self.job.processing_started, self.job.processing_started_epoch = 10., 20.
        self.camera.lock = threading.RLock()
        self.camera.preview_error = None
        self.camera.resume_preview = Mock(side_effect=OSError('USB device absent'))
        cancel = threading.Event()
        def launch(command, **kwargs):
            kwargs['stdout'].write((json.dumps(dict(event='VIEWER_READY',
                url='http://127.0.0.1:8080/#token=no-usb'))+'\n').encode())
            kwargs['stdout'].flush()
            return SimpleNamespace(poll=lambda: None, returncode=None)
        with patch.object(self.job, 'stop_viewer') as stop, \
                patch('web.camera_ui.subprocess.Popen', side_effect=launch), \
                patch('web.camera_ui.time.sleep', side_effect=lambda seconds: cancel.set()):
            self.job.run(None, cancel)
        stop.assert_called_once()  # Old owner cleanup only, never error cleanup.
        self.assertEqual(self.job.phase, 'ready')
        self.assertIsNone(self.job.error)
        self.assertTrue(self.job.processing_complete)
        self.assertEqual(self.job.processing_ended, 18.)
        self.assertEqual(self.job.url, 'http://127.0.0.1:18434/#token=no-usb')
        self.assertIn('USB', self.camera.preview_error)

    def test_saved_scene_restore_replays_old_frame_without_claiming_new_processing_time(self):
        run = self.root/'scene_20261103_120000_abcdef'
        run.mkdir()
        (run/'events.json').write_text(json.dumps(dict(events=[
            dict(event='VIDEO_COMPLETE', path='uploaded.mp4', monotonic=10.),
            dict(event='FRAME_COMPLETE', monotonic=18.)])), encoding='utf-8')
        with patch('web.camera_ui.threading.Thread'):
            self.job.start('uploaded.mp4', run, input_kind='uploaded_video')
        self.assertIsNone(self.job.processing_started)
        self.assertIsNone(self.job.processing_started_epoch)
        cancel = threading.Event()
        def launch(command, **kwargs):
            self.assertIn('web.viewer', command)
            self.assertNotIn('initialize', command)
            kwargs['stdout'].write((json.dumps(dict(event='VIEWER_READY',
                url='http://127.0.0.1:8080/#token=restored'))+'\n').encode())
            kwargs['stdout'].flush()
            return SimpleNamespace(poll=lambda: None, returncode=None)
        with patch.object(self.job, 'stop_viewer'), \
                patch('web.camera_ui.subprocess.Popen', side_effect=launch), \
                patch('web.camera_ui.time.monotonic', return_value=1000.), \
                patch('web.camera_ui.time.sleep', side_effect=lambda seconds: cancel.set()):
            self.job.run('uploaded.mp4', cancel, scene=run)
        state = self.job.status()
        self.assertEqual(state['phase'], 'ready')
        self.assertTrue(state['processing_complete'])
        self.assertTrue(state['scene_restored'])
        self.assertEqual(state['processing_timing_scope'], 'restored_scene_no_new_measurement')
        self.assertIsNone(state['processing_elapsed_seconds'])
        self.assertIsNone(state['processing_started_epoch'])
        self.assertIsNone(self.job.processing_ended)
        self.assertEqual(state['url'], 'http://127.0.0.1:18434/#token=restored')


if __name__ == '__main__':
    unittest.main()
