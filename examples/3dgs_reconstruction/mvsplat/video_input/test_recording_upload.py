"""Local HTTP/upload coordination contracts; no USB, inference or board activity."""
import http.client
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.parse import quote

from video_input.recording import CameraRecorder
from web.recording_server import make_server
from video_input.test_recording import FakePreview
from video_input import upload
from video_input import test_recording as recording_tests


class Build:
    def __init__(self):
        self.record = dict(phase='idle', url=None, clip_path=None, mode='clip',
                           processing_complete=False, processing_elapsed_seconds=0.)
        self.starts, self.primes = [], []
        self.args = SimpleNamespace(calibration=Path('usb-only-calibration.json'))

    def status(self):
        return dict(self.record)

    def start(self, path, *, input_kind=None):
        self.starts.append((path, input_kind))
        self.record.update(phase='reconstructing', url=None, clip_path=path,
                           processing_complete=False)

    def prime(self, *, input_kind=None):
        self.primes.append(input_kind)
        self.record.update(phase='initializing', url=None)

    def close(self):
        pass

    def reset(self):
        self.record.update(phase='idle', url=None, clip_path=None,
                           processing_complete=False, processing_elapsed_seconds=0.)

    def start_stream(self):
        self.record.update(phase='streaming', url=None, clip_path=None)


def clip_record(root, name='new.mp4'):
    return dict(complete=True, source_kind='uploaded_video', path=str(Path(root)/name),
        original_filename=name, duration_seconds=2., frames=61, measured_fps=30.,
        stop_label='本地视频上传完成', upload_complete_monotonic=10.,
        upload_complete_epoch=20., actual=dict(width=1280, height=720, fps=30.))


class UploadCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.camera = CameraRecorder('/dev/video-do-not-open', 1280, 720, 60.,
                                     self.root, factory=FakePreview)
        self.camera.build = Build()
        self.addCleanup(self.camera.close)

    def test_upload_submits_once_without_camera_snapshot_or_usb_geometry(self):
        record = clip_record(self.root)
        with patch.object(self.camera, 'snapshot', side_effect=AssertionError('USB must not be read')), \
                patch('video_input.upload.stream_receive', return_value=record) as receiver:
            actual = self.camera.receive_upload(io.BytesIO(b'bytes'), 5, 'new.mp4')
        self.assertEqual(actual, record)
        receiver.assert_called_once()
        self.assertEqual(self.camera.build.starts, [(record['path'], 'uploaded_video')])
        state = self.camera.status()
        self.assertEqual(state['input_kind'], 'uploaded_video')
        self.assertEqual(state['processing_started_epoch'], 20.)
        self.assertTrue(state['build']['matches_clip'])

    def test_failed_upload_does_not_bind_old_result_or_old_timer(self):
        old_preview = self.camera.preview
        self.camera.clip = clip_record(self.root, 'old.mp4')
        self.camera.upload = dict(self.camera.clip)
        self.camera.build.record.update(phase='ready', clip_path=self.camera.clip['path'],
            processing_complete=True, processing_elapsed_seconds=4., url='http://old-result/')
        with patch('video_input.upload.stream_receive', side_effect=ValueError('invalid new video')):
            with self.assertRaisesRegex(ValueError, 'invalid new video'):
                self.camera.receive_upload(io.BytesIO(b'bad'), 3, 'bad.mp4')
        state = self.camera.status()
        self.assertEqual(state['state'], 'error')
        self.assertIsNone(state['clip'])
        self.assertIsNone(state['upload'])
        self.assertFalse(state['build']['matches_clip'])
        self.assertIsNone(state['processing_started_epoch'])
        self.assertEqual(state['postcapture_elapsed_seconds'], 0.)
        self.assertEqual(self.camera.build.starts, [])
        self.assertIs(self.camera.preview, old_preview)
        self.assertFalse(old_preview.closed)

    def test_job_rejection_does_not_claim_new_frame_completed(self):
        record = clip_record(self.root)
        self.camera.build.record.update(phase='ready', clip_path='old',
            url='http://old-result/', processing_complete=True, processing_elapsed_seconds=4.)
        with patch('video_input.upload.stream_receive', return_value=record), \
                patch.object(self.camera.build, 'start', side_effect=RuntimeError('owner unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'owner unavailable'):
                self.camera.receive_upload(io.BytesIO(b'new'), 3, 'new.mp4')
        state = self.camera.status()
        self.assertEqual(state['state'], 'error')
        self.assertFalse(state['build']['matches_clip'])
        self.assertIsNotNone(state['error'])
        self.assertIsNotNone(self.camera.preview)

    def test_verified_upload_pauses_preview_before_preheated_stdin_handoff(self):
        from web.camera_ui import ReconstructionJob
        old_preview = self.camera.preview
        job = ReconstructionJob(self.camera, SimpleNamespace())
        job.phase, job.input_kind, job.warm_pending = 'warmed', 'uploaded_video', True
        stdin = io.BytesIO()
        job.process = SimpleNamespace(poll=lambda: None, stdin=stdin)
        self.camera.build = job
        self.camera.factory = Mock(side_effect=AssertionError('packet must not reopen USB'))
        record = clip_record(self.root)
        try:
            with patch('video_input.upload.stream_receive', return_value=record), \
                    patch.object(self.camera, 'snapshot', side_effect=AssertionError('no USB read')), \
                    patch('web.camera_ui.threading.Thread') as thread:
                self.camera.receive_upload(io.BytesIO(b'abc'), 3, 'new.mp4')
            self.assertTrue(old_preview.closed)
            self.assertIsNone(self.camera.preview)
            self.assertEqual(stdin.getvalue(), (record['path']+'\n').encode())
            self.assertEqual(job.phase, 'reconstructing')
            thread.assert_not_called()
            with self.assertRaises(RuntimeError):
                self.camera.packet()
            self.camera.factory.assert_not_called()
        finally:
            job.process = None

    def test_dispatch_failure_keeps_primary_error_if_preview_recovery_fails(self):
        old_preview = self.camera.preview
        record = clip_record(self.root)
        self.camera.factory = Mock(side_effect=OSError('USB recovery disconnected'))
        with patch('video_input.upload.stream_receive', return_value=record), \
                patch.object(self.camera.build, 'start', side_effect=RuntimeError('primary dispatch failure')):
            with self.assertRaisesRegex(RuntimeError, 'primary dispatch failure'):
                self.camera.receive_upload(io.BytesIO(b'abc'), 3, 'new.mp4')
        self.assertTrue(old_preview.closed)
        self.assertEqual(self.camera.state, 'error')
        self.assertIn('上传或视频校验失败', self.camera.error)
        self.assertIn('USB recovery disconnected', self.camera.preview_error)

    def test_postupload_clock_freezes_only_for_matching_frame_complete(self):
        self.camera.clip = clip_record(self.root)
        self.camera.upload = dict(self.camera.clip)
        self.camera.video_complete_monotonic = 10.
        self.camera.video_complete_epoch = 20.
        self.camera.build.record.update(phase='ready', clip_path='other',
            processing_complete=True, processing_elapsed_seconds=4.)
        with patch('video_input.recording.time.monotonic', return_value=100.):
            self.assertEqual(self.camera.status()['postcapture_elapsed_seconds'], 90.)
            self.camera.build.record['clip_path'] = self.camera.clip['path']
            self.assertEqual(self.camera.status()['postcapture_elapsed_seconds'], 4.)
            self.camera.build.record['processing_complete'] = False
            self.assertEqual(self.camera.status()['postcapture_elapsed_seconds'], 90.)
            self.camera.build.record.update(scene_restored=True, processing_complete=True,
                                            processing_elapsed_seconds=None)
            state = self.camera.status()
            self.assertIsNone(state['postcapture_elapsed_seconds'])
            self.assertIsNone(state['processing_started_epoch'])

    def test_upload_clock_includes_final_write_flush_and_probe_without_success_reset(self):
        clock, observed = [100.], []
        original_open, original_receive = Path.open, upload.stream_receive
        camera = self.camera
        class Body(io.BytesIO):
            def read(self, count):
                block = super().read(count)
                clock[0] += 1.
                return block
        class SlowSink:
            def __init__(self, target):
                self.target = target
            def __enter__(self):
                self.target.__enter__()
                return self
            def __exit__(self, *args):
                return self.target.__exit__(*args)
            def write(self, block):
                observed.append(('before_write', camera.video_complete_monotonic))
                clock[0] += 2.
                return self.target.write(block)
            def flush(self):
                clock[0] += 3.
                observed.append(('after_flush', camera.status()['postcapture_elapsed_seconds']))
                return self.target.flush()
        def slow_open(path, *args, **kwargs):
            target = original_open(path, *args, **kwargs)
            return SlowSink(target) if path.name == 'video.partial' and args == ('xb',) else target
        def probe(path, *, decoded_frames):
            clock[0] += 5.
            observed.append(('after_probe', camera.status()['postcapture_elapsed_seconds']))
            return dict(codec_name='h264', width=1280, height=720, fps=2.), [0., .5, 1.]
        def receive(stream, length, root, filename, **kwargs):
            return original_receive(stream, length, root, filename, inspector=probe, **kwargs)
        with patch.object(Path, 'open', slow_open), \
                patch('video_input.upload.stream_receive', side_effect=receive), \
                patch('video_input.upload.time.monotonic', side_effect=lambda: clock[0]), \
                patch('video_input.upload.time.time', side_effect=lambda: clock[0]+1000.), \
                patch('video_input.upload.shutil.disk_usage',
                      return_value=SimpleNamespace(free=2*1024**3)):
            record = self.camera.receive_upload(Body(b'abc'), 3, 'a.mp4')
            self.assertEqual(observed, [('before_write', 101.), ('after_flush', 5.), ('after_probe', 10.)])
            self.assertEqual(self.camera.video_complete_monotonic, 101.)
            self.assertEqual(self.camera.video_complete_epoch, 1101.)
            self.assertEqual(self.camera.status()['postcapture_elapsed_seconds'], 10.)
            self.assertEqual(self.camera.status()['processing_started_epoch'], 1101.)
            self.assertEqual(record['upload_complete_monotonic'], 101.)
            self.assertEqual(self.camera.build.starts, [(record['path'], 'uploaded_video')])

    def owned_upload(self, name, *, complete=True):
        folder = self.root/name
        folder.mkdir()
        video = folder/'video.mp4'
        video.write_bytes(b'private fixture bytes')
        record = clip_record(folder, 'video.mp4')
        record.update(path=str(video.resolve()), complete=complete,
            schema='mvsplat_uploaded_video_v1', sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
            declared_bytes=video.stat().st_size, received_bytes=video.stat().st_size)
        (folder/'upload.json').write_text(json.dumps(record), encoding='utf-8')
        return folder, record

    def test_restore_upload_preserves_input_kind_and_clears_historical_processing_clock(self):
        folder, record = self.owned_upload('upload_20261103_100000_abcde123')
        self.camera.video_complete_monotonic, self.camera.video_complete_epoch = 100., 200.
        self.camera.restore_clip(folder/'upload.json')
        self.assertEqual(self.camera.clip, record)
        self.assertEqual(self.camera.upload, record)
        self.assertEqual(self.camera.input_kind, 'uploaded_video')
        self.assertIsNone(self.camera.video_complete_monotonic)
        self.assertIsNone(self.camera.video_complete_epoch)
        self.assertEqual(self.camera.status()['postcapture_elapsed_seconds'], 0.)
        self.camera.reconstruct()
        self.assertEqual(self.camera.build.starts, [(record['path'], 'uploaded_video')])

    def test_restore_upload_rejects_schema_completion_hash_size_and_cross_directory_paths(self):
        folder, record = self.owned_upload('upload_20261103_100100_abcde124')
        other, other_record = self.owned_upload('upload_20261103_100101_abcde125')
        metadata = folder/'upload.json'
        for change in (dict(schema='other'), dict(source_kind='usb_camera'),
                       dict(complete=False), dict(complete=1), dict(sha256='0'*64),
                       dict(declared_bytes=0), dict(received_bytes=True),
                       dict(received_bytes=record['received_bytes']+1),
                       dict(path=other_record['path']), dict(path='video.mp4')):
            with self.subTest(change=change):
                metadata.write_text(json.dumps(dict(record, **change)), encoding='utf-8')
                with self.assertRaises(ValueError):
                    self.camera.restore_clip(metadata)
                self.assertIsNone(self.camera.clip)
                self.assertIsNone(self.camera.upload)
        metadata.write_text(json.dumps(record), encoding='utf-8')
        Path(record['path']).write_bytes(b'tampered original file')
        with self.assertRaises(ValueError):
            self.camera.restore_clip(metadata)

    def test_restore_upload_rejects_outside_unowned_and_replaced_capture_metadata(self):
        folder, record = self.owned_upload('upload_20261103_100200_abcde126')
        with tempfile.TemporaryDirectory() as outside:
            metadata = Path(outside)/'upload.json'
            metadata.write_text(json.dumps(record), encoding='utf-8')
            with self.assertRaises(ValueError):
                self.camera.restore_clip(metadata)
        unowned = self.root/'not_an_owned_upload'
        unowned.mkdir()
        video = unowned/'video.mp4'
        video.write_bytes(Path(record['path']).read_bytes())
        metadata = unowned/'upload.json'
        metadata.write_text(json.dumps(dict(record, path=str(video.resolve()))), encoding='utf-8')
        with self.assertRaises(ValueError):
            self.camera.restore_clip(metadata)
        # Renaming/replacing metadata with a USB-looking capture cannot bypass
        # the stricter upload provenance checks in an owned upload directory.
        capture = folder/'capture.mkv'
        capture.write_bytes(b'USB-looking replacement')
        forged = dict(record, path=str(capture.resolve()),
                      sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
                      requested=dict(width=1280, height=720, fps=60))
        forged.pop('schema')
        forged.pop('source_kind')
        metadata = folder/'capture.json'
        metadata.write_text(json.dumps(forged), encoding='utf-8')
        with self.assertRaises(ValueError):
            self.camera.restore_clip(metadata)
        self.assertIsNone(self.camera.clip)

    def test_restored_upload_saved_scene_uses_upload_geometry_without_old_timestamps(self):
        from web.camera_ui import ReconstructionJob
        folder, record = self.owned_upload('upload_20261103_100300_abcde127')
        self.camera.restore_clip(folder/'upload.json')
        scene = self.root/'scene_20261103_100301_abcdef'
        scene.mkdir()
        (scene/'events.json').write_text(json.dumps(dict(events=[
            dict(event='VIDEO_COMPLETE', path=record['path'])])), encoding='utf-8')
        job = ReconstructionJob(self.camera, SimpleNamespace(out=self.root))
        self.camera.build = job
        with patch('web.camera_ui.threading.Thread') as thread, \
                patch('web.camera_ui.time.monotonic', return_value=500.), \
                patch('web.camera_ui.time.time', return_value=600.):
            job.start(self.camera.clip['path'], scene)
        self.assertEqual(job.input_kind, 'uploaded_video')
        self.assertIsNone(job.processing_started)
        self.assertIsNone(job.processing_started_epoch)
        self.assertTrue(job.scene_restored)
        self.assertFalse(job.processing_complete)
        self.assertEqual(thread.call_args.kwargs['args'][3], scene.resolve())

    def test_restoring_usb_capture_clears_prior_upload_identity_and_clock(self):
        folder, record = self.owned_upload('upload_20261103_100400_abcde128')
        self.camera.restore_clip(folder/'upload.json')
        capture = recording_tests.RecordingTests().clip(self.root, 'clip_20261103_100401_abcdef')
        self.camera.video_complete_monotonic, self.camera.video_complete_epoch = 10., 20.
        self.camera.restore_clip(capture/'capture.json')
        self.assertEqual(self.camera.input_kind, 'usb_camera')
        self.assertIsNone(self.camera.upload)
        self.assertIsNone(self.camera.video_complete_monotonic)
        self.assertIsNone(self.camera.video_complete_epoch)

    def test_upload_cleanup_keeps_latest_scene_source_and_current_and_removes_old_failed(self):
        stale, _ = self.owned_upload('upload_20261103_080000_aaaaaaaa')
        source, source_record = self.owned_upload('upload_20261103_081000_bbbbbbbb')
        newest, _ = self.owned_upload('upload_20261103_082000_cccccccc')
        failed, _ = self.owned_upload('upload_20261103_083000_dddddddd', complete=False)
        for index, folder in enumerate((stale, source, newest, failed)):
            os.utime(folder, (1_700_000_000+index, 1_700_000_000+index))
        scene = recording_tests.RecordingTests().scene(self.root, 'scene_20261103_081100_aaaaaa', source)
        events = json.loads((scene/'events.json').read_text())
        events['events'][0]['path'] = source_record['path']
        (scene/'events.json').write_text(json.dumps(events))
        self.camera.build.record.update(phase='ready', run_path=str(scene))
        unknown = self.root/'upload_20261103_000000_unknown'
        unknown.mkdir()
        (unknown/'upload.json').write_text('{}')
        result = self.camera.cleanup_history()
        self.assertEqual(set(result['removed']), {stale.name, failed.name})
        self.assertFalse(stale.exists())
        self.assertFalse(failed.exists())
        self.assertTrue(source.exists())
        self.assertTrue(newest.exists())
        self.assertTrue(scene.exists())
        self.assertTrue(unknown.exists())
        self.assertIn(source.name, result['kept'])
        self.assertIn(newest.name, result['kept'])

    def test_clear_all_removes_upload_sessions_and_resets_cross_input_status(self):
        folder, record = self.owned_upload('upload_20261103_090000_abcdef12')
        failed, _ = self.owned_upload('upload_20261103_090100_abcdef34', complete=False)
        self.camera.upload = self.camera.clip = record
        self.camera.input_kind, self.camera.state = 'uploaded_video', 'recorded'
        self.camera.video_complete_monotonic, self.camera.video_complete_epoch = 10., 20.
        result = self.camera.clear_all_history()
        self.assertEqual(set(result['removed']), {folder.name, failed.name})
        self.assertIsNone(self.camera.upload)
        self.assertIsNone(self.camera.clip)
        self.assertEqual(self.camera.input_kind, 'usb_camera')
        self.assertEqual(self.camera.state, 'idle')
        self.assertIsNone(self.camera.video_complete_monotonic)
        self.assertIsNone(self.camera.video_complete_epoch)

    def test_new_usb_stream_drops_upload_status_before_dispatch(self):
        record = clip_record(self.root)
        self.camera.upload = self.camera.clip = record
        self.camera.input_kind, self.camera.state = 'uploaded_video', 'recorded'
        self.camera.video_complete_monotonic, self.camera.video_complete_epoch = 10., 20.
        self.camera.elapsed, self.camera.started = 2., 9.
        with patch.object(self.camera, 'cleanup_history', return_value=dict(kept=[])), \
                patch('video_input.recording.shutil.disk_usage',
                      return_value=SimpleNamespace(free=2*1024**3)):
            self.camera.start_stream()
        self.assertEqual(self.camera.input_kind, 'usb_camera')
        self.assertIsNone(self.camera.upload)
        self.assertIsNone(self.camera.clip)
        self.assertIsNone(self.camera.video_complete_monotonic)
        self.assertIsNone(self.camera.video_complete_epoch)
        self.assertEqual(self.camera.state, 'idle')
        self.assertEqual(self.camera.elapsed, 0.)
        self.assertIsNone(self.camera.started)
        self.assertEqual(self.camera.build.record['phase'], 'streaming')
        with self.assertRaises(ValueError):
            self.camera.reconstruct()

    def test_missing_usb_is_only_preview_error_and_does_not_block_uploaded_video(self):
        for missing in (OSError('USB device absent'), ValueError('USB mode unavailable')):
            with self.subTest(missing=type(missing).__name__):
                factory = Mock(side_effect=missing)
                camera = CameraRecorder('/dev/no-usb', 1280, 720, 60.,
                                        self.root, factory=factory)
                camera.build = Build()
                record = clip_record(self.root)
                try:
                    self.assertIsNone(camera.preview)
                    self.assertIsNone(camera.error)
                    self.assertEqual(camera.state, 'idle')
                    self.assertIsNotNone(camera.status()['preview_error'])
                    with patch('video_input.upload.stream_receive', return_value=record):
                        camera.receive_upload(io.BytesIO(b'x'), 1, 'new.mp4')
                    self.assertEqual(camera.build.starts, [(record['path'], 'uploaded_video')])
                    self.assertEqual(camera.state, 'recorded')
                    self.assertIsNone(camera.error)
                finally:
                    camera.close()

    def test_missing_usb_start_retries_but_never_dispatches_recording_or_reconstruction(self):
        factory = Mock(side_effect=OSError('USB device absent'))
        camera = CameraRecorder('/dev/no-usb', 1280, 720, 60., self.root, factory=factory)
        camera.build = Build()
        try:
            with patch('video_input.recording.threading.Thread') as thread:
                with self.assertRaisesRegex(RuntimeError, 'USB 摄像头不可用'):
                    camera.start()
                thread.assert_not_called()
            self.assertEqual(factory.call_count, 2)
            self.assertEqual(camera.build.starts, [])
            self.assertIsNone(camera.error)
            self.assertEqual(camera.state, 'idle')
            self.assertFalse(list(self.root.glob('clip_*')))
        finally:
            camera.close()

    def test_upload_status_omits_all_frame_pts_without_mutating_saved_record(self):
        record = clip_record(self.root)
        record['frame_pts_seconds'] = [i/30 for i in range(61)]
        self.camera.upload = self.camera.clip = record
        self.camera.input_kind = 'uploaded_video'
        state = self.camera.status()
        self.assertNotIn('frame_pts_seconds', state['upload'])
        self.assertEqual(len(self.camera.upload['frame_pts_seconds']), 61)

    def test_upload_owns_all_input_and_cleanup_admission_until_receipt(self):
        entered, release = threading.Event(), threading.Event()
        outcome = []
        def receive(*args, **kwargs):
            entered.set()
            if not release.wait(3):
                raise RuntimeError('test did not release receiver')
            return clip_record(self.root)
        def run():
            try:
                outcome.append(self.camera.receive_upload(io.BytesIO(b'x'), 1, 'new.mp4'))
            except BaseException as exc:
                outcome.append(exc)
        with patch('video_input.upload.stream_receive', side_effect=receive) as receiver:
            worker = threading.Thread(target=run)
            worker.start()
            try:
                self.assertTrue(entered.wait(2))
                for action in (self.camera.start, self.camera.prepare_upload,
                               self.camera.start_stream, self.camera.cleanup_history,
                               self.camera.clear_all_history):
                    with self.subTest(action=action.__name__), self.assertRaises(ValueError):
                        action()
                with self.assertRaises(ValueError):
                    self.camera.receive_upload(io.BytesIO(b'y'), 1, 'other.mp4')
                self.assertEqual(receiver.call_count, 1)
                self.assertEqual(self.camera.build.starts, [])
            finally:
                release.set()
                worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(outcome), 1)
        self.assertIsInstance(outcome[0], dict)
        self.assertEqual(len(self.camera.build.starts), 1)


class RawUploadHttpTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.camera = CameraRecorder('/dev/video-do-not-open', 1280, 720, 60.,
                                     self.root, factory=FakePreview)
        self.camera.build = Build()
        with patch('video_input.recording.secrets.token_urlsafe', return_value='test-session'):
            self.server = make_server(self.camera, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.shutdown)

    def shutdown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(3)
        self.camera.close()

    def request(self, route, headers=(), body=b'', method='POST'):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        try:
            connection.putrequest(method, route)
            for key, value in headers:
                connection.putheader(key, value)
            connection.endheaders(body)
            reply = connection.getresponse()
            return reply.status, reply.read(), dict(reply.getheaders())
        finally:
            connection.close()

    def test_dual_pages_share_auth_and_upload_page_does_not_claim_usb_calibration(self):
        for page in ('/capture', '/upload'):
            code, data, headers = self.request(page, method='GET')
            self.assertEqual(code, 200)
            self.assertEqual(headers['Cache-Control'], 'no-store')
            self.assertIn(b'"test-session"', data)
            self.assertNotIn(b'__TOKEN__', data)
        _, data, _ = self.request('/upload', method='GET')
        self.assertIn('不会默认套用 USB 摄像头的标定'.encode(), data)

    def test_both_actual_pages_label_restored_scene_without_new_performance_time(self):
        node = shutil.which('node')
        if not node:
            self.skipTest('Node unavailable for actual page-script execution')
        program = r'''
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
for (const [path,kind] of [[process.argv[1],'uploaded_video'],[process.argv[2],'usb_camera']]) {
  const elements=new Map();
  const element=id=>{if(!elements.has(id))elements.set(id,{textContent:'',files:[],dataset:{},style:{},
    addEventListener(){},getAttribute(k){return this[k]||null;},removeAttribute(k){delete this[k];}});return elements.get(id);};
  const context=vm.createContext({document:{getElementById:element},window:{addEventListener(){}},
    performance:{now:()=>100},fetch:()=>new Promise(()=>{}),setInterval:()=>0,URL,console});
  const source=fs.readFileSync(path,'utf8').match(/<script>([\s\S]*?)<\/script>/)[1].replace('__TOKEN__','"test"');
  vm.runInContext(source,context,{timeout:1000});
  const record={complete:true,path:'/owned/video.mp4',source_kind:kind,actual:{width:1280,height:720},
    duration_seconds:2,frames:61,measured_fps:30,stop_label:'fixture'};
  const payload={state:'recorded',input_kind:kind,clip:record,upload:kind==='uploaded_video'?record:null,
    elapsed_seconds:2,max_seconds:60,free_mib:256,postcapture_elapsed_seconds:2,processing_started_epoch:100,
    build:{phase:'reconstructing',mode:'clip',clip_path:record.path,matches_clip:true,
      scene_restored:false,processing_complete:false,processing_elapsed_seconds:2,processing_started_epoch:100}};
  context.applyStatus(payload);
  assert.notEqual(element('processing-time').textContent,'—');
  payload.build={...payload.build,scene_restored:true,phase:'ready',processing_complete:true,
    processing_elapsed_seconds:null,processing_started_epoch:null,url:'http://viewer/#saved'};
  payload.postcapture_elapsed_seconds=payload.processing_started_epoch=null;
  context.applyStatus(payload);
  assert.equal(element('processing-time').textContent,'—');
  assert.equal(element('timer-caption').textContent,'已恢复历史场景 · 未重新计时');
  assert.equal(element('build-status').textContent,'历史场景已恢复，可以选择观察方向');
  assert.equal(element('result-viewer').hidden,false);
  assert.equal(element('result-viewer').src,'http://viewer/#saved');
  payload.build={...payload.build,scene_restored:false,phase:'reconstructing',processing_complete:false,
    processing_elapsed_seconds:3,processing_started_epoch:200,url:null};
  payload.postcapture_elapsed_seconds=3;payload.processing_started_epoch=200;
  context.applyStatus(payload);
  assert.equal(element('processing-time').textContent,'00:03.0');
  assert.equal(element('result-viewer').hidden,true);
}
'''
        result = subprocess.run([node, '-e', program,
            str(Path(__file__).resolve().parents[1]/'web/upload_page.html'),
            str(Path(__file__).resolve().parents[1]/'web/recording_page.html')],
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_auth_is_checked_before_reading_any_upload_or_preparing_owner(self):
        with patch('video_input.upload.stream_receive') as receiver:
            for route in ('/api/upload?filename=clip.mp4', '/api/prepare/upload'):
                for headers in ([], [('Authorization', 'Bearer wrong')],
                                [('X-Session-Token', 'test-session'), ('Authorization', 'Bearer wrong')]):
                    code, _, _ = self.request(route, headers)
                    self.assertEqual(code, 403)
            receiver.assert_not_called()
        self.assertEqual(self.camera.state, 'idle')
        self.assertEqual(self.camera.build.primes, [])

    def test_raw_upload_rejects_missing_duplicate_chunked_and_invalid_lengths(self):
        auth = [('Authorization', 'Bearer test-session')]
        cases = [[], [('Content-Length', '1'), ('Content-Length', '1')],
            [('Content-Length', '1'), ('Transfer-Encoding', 'chunked')],
            [('Content-Length', '-1')], [('Content-Length', 'nan')],
            [('Content-Length', '0')], [('Content-Length', str(upload.MAX_UPLOAD_BYTES+1))]]
        original = upload.stream_receive
        with patch('video_input.upload.stream_receive', wraps=original) as receiver:
            for extra in cases:
                with self.subTest(headers=extra):
                    before = receiver.call_count
                    code, body, _ = self.request('/api/upload?filename=clip.mp4', auth+extra)
                    self.assertEqual(code, 409, body)
                    self.assertEqual(self.camera.build.starts, [])
                    if not extra or len(extra)>1 or extra[0][1] in ('-1', 'nan'):
                        self.assertEqual(receiver.call_count, before)
        self.assertEqual(list(self.root.glob('upload_*')), [])

    def test_authenticated_raw_upload_preserves_exact_body_and_dispatches_uploaded_video(self):
        payload = b'\x00raw-video-payload\xff'
        inspector = Mock(return_value=(dict(codec_name='h264', width=1280, height=720, fps=2.),
                                       [0., .5, 1.]))
        original = upload.stream_receive
        def receive(stream, length, root, filename, **kwargs):
            return original(stream, length, root, filename, inspector=inspector, **kwargs)
        with patch('video_input.upload.stream_receive', side_effect=receive), \
                patch('video_input.upload.shutil.disk_usage',
                      return_value=SimpleNamespace(free=2*1024**3)):
            code, body, _ = self.request('/api/upload?filename='+quote('我的视频.mp4'),
                [('Authorization', 'Bearer test-session'), ('Content-Length', str(len(payload))),
                 ('Content-Type', 'application/octet-stream')], payload)
        self.assertEqual(code, 202, body)
        record = json.loads(body)['upload']
        self.assertEqual(Path(record['path']).read_bytes(), payload)
        self.assertEqual(record['original_filename'], '我的视频.mp4')
        self.assertEqual(self.camera.build.starts, [(record['path'], 'uploaded_video')])
        self.assertTrue(self.camera.status()['build']['matches_clip'])

    def test_second_http_upload_rejected_while_first_is_receiving(self):
        entered, release = threading.Event(), threading.Event()
        results = []
        def receive(stream, length, root, filename, **kwargs):
            self.assertEqual(stream.read(length), b'a')
            entered.set()
            if not release.wait(3):
                raise RuntimeError('test receiver release timed out')
            return clip_record(root)
        def first():
            results.append(self.request('/api/upload?filename=a.mp4',
                [('X-Session-Token', 'test-session'), ('Content-Length', '1')], b'a'))
        with patch('video_input.upload.stream_receive', side_effect=receive) as receiver:
            worker = threading.Thread(target=first)
            worker.start()
            try:
                self.assertTrue(entered.wait(2))
                code, _, _ = self.request('/api/upload?filename=b.mp4',
                    [('Authorization', 'Bearer test-session'), ('Content-Length', '1')], b'b')
                self.assertEqual(code, 409)
                self.assertEqual(receiver.call_count, 1)
            finally:
                release.set()
                worker.join(4)
        self.assertFalse(worker.is_alive())
        self.assertEqual(results[0][0], 202)
        self.assertEqual(len(self.camera.build.starts), 1)


if __name__ == '__main__':
    unittest.main()
