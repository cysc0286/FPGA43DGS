"""Bounded raw upload, decoded-frame contracts and timing regressions."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from video_input.upload import MAX_UPLOAD_BYTES, RESERVE_BYTES, stream_receive


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'received'
        self.inspector = Mock(return_value=(
            dict(codec_name='h264', width=1280, height=720, fps=2.), [0., .5, 1.]))
        self.space = patch('video_input.upload.shutil.disk_usage',
                           return_value=SimpleNamespace(free=4 * 1024 ** 3)).start()
        self.addCleanup(patch.stopall)

    def receive(self, payload=b'complete video bytes', **kwargs):
        return stream_receive(io.BytesIO(payload), len(payload), self.root,
                              kwargs.pop('filename', 'clip.mp4'),
                              inspector=kwargs.pop('inspector', self.inspector), **kwargs)

    def failed_record(self):
        files = list(self.root.glob('upload_*/upload.json'))
        self.assertEqual(len(files), 1)
        record = json.loads(files[0].read_text(encoding='utf-8'))
        self.assertFalse(record['complete'])
        return files[0].parent, record

    def test_streamed_success_is_owned_atomic_copy_with_hash_and_decoded_probe(self):
        payload = b'x' * 31
        reads = []
        class Body(io.BytesIO):
            def read(self, count):
                reads.append(count)
                return super().read(count)
        record = stream_receive(Body(payload), len(payload), self.root,
            r'C:\fakepath\我的视频.MP4', chunk_bytes=7, inspector=self.inspector)
        path = Path(record['path'])
        self.assertEqual(path.read_bytes(), payload)
        self.assertEqual(path.name, 'video.mp4')
        self.assertEqual(record['original_filename'], '我的视频.MP4')
        self.assertEqual(record['sha256'], hashlib.sha256(payload).hexdigest())
        self.assertEqual(record['received_bytes'], len(payload))
        self.assertTrue(record['complete'])
        self.assertEqual(record['source_kind'], 'uploaded_video')
        self.assertEqual(reads, [7, 7, 7, 7, 3])
        self.assertFalse(path.with_name('video.partial').exists())
        self.inspector.assert_called_once_with(path.with_name('video.partial'), decoded_frames=True)
        self.assertEqual(json.loads(path.with_name('upload.json').read_text(encoding='utf-8')), record)

    def test_filename_cannot_select_server_directories(self):
        record = self.receive(filename='../../outside/my video.weird')
        self.assertEqual(record['original_filename'], 'my video.weird')
        self.assertEqual(Path(record['path']).name, 'video.video')
        self.assertEqual(Path(record['path']).parent.parent, self.root.resolve())

    def test_original_source_is_never_deleted_or_changed(self):
        source = Path(self.temporary.name) / 'source.mp4'
        source.write_bytes(b'input source untouched')
        with source.open('rb') as body:
            record = stream_receive(body, source.stat().st_size, self.root,
                                    str(source), inspector=self.inspector)
        self.assertEqual(source.read_bytes(), b'input source untouched')
        self.assertNotEqual(Path(record['path']), source)
        self.assertEqual(record['original_filename'], source.name)

    def test_new_upload_never_overwrites_previous_owned_file(self):
        first, second = self.receive(b'first'), self.receive(b'second')
        self.assertNotEqual(first['path'], second['path'])
        self.assertEqual(Path(first['path']).read_bytes(), b'first')

    def test_limits_reject_before_writing_or_probing(self):
        cases = [dict(length=0), dict(length=-1), dict(length=True),
                 dict(length=MAX_UPLOAD_BYTES + 1), dict(max_bytes=None),
                 dict(max_bytes=MAX_UPLOAD_BYTES + 1), dict(reserve_bytes=0),
                 dict(chunk_bytes=8 * 1024 ** 2), dict(max_receive_seconds=float('nan')),
                 dict(max_fps=0), dict(min_edge=0), dict(on_received=False), dict(filename='..'),
                 dict(filename='bad\n.mp4')]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                options = dict(args)
                stream_receive(io.BytesIO(b'abc'), options.pop('length', 3), self.root,
                               options.pop('filename', 'a.mp4'), inspector=self.inspector, **options)
        self.assertFalse(self.root.exists())
        self.inspector.assert_not_called()

    def test_initial_disk_reserve_rejects_before_folder_is_created(self):
        self.space.return_value.free = RESERVE_BYTES + 2
        with self.assertRaises(OSError):
            self.receive(b'abc')
        self.assertEqual(list(self.root.iterdir()), [])
        self.inspector.assert_not_called()

    def test_disk_pressure_during_stream_keeps_partial_and_failure_record(self):
        self.space.side_effect = [SimpleNamespace(free=4 * 1024 ** 3),
                                 SimpleNamespace(free=RESERVE_BYTES + 2)]
        with self.assertRaises(OSError):
            self.receive(b'abcd')
        folder, record = self.failed_record()
        self.assertEqual(record['received_bytes'], 0)
        self.assertTrue((folder / 'video.partial').exists())
        self.assertFalse((folder / 'video.mp4').exists())

    def test_truncated_body_is_not_published_or_probed(self):
        with self.assertRaises(ValueError):
            stream_receive(io.BytesIO(b'ab'), 3, self.root, 'a.mp4', inspector=self.inspector)
        folder, record = self.failed_record()
        self.assertEqual(record['received_bytes'], 2)
        self.assertEqual((folder / 'video.partial').read_bytes(), b'ab')
        self.inspector.assert_not_called()

    def test_oversized_or_invalid_read_result_is_rejected(self):
        for value in (b'abcd', 'not bytes'):
            with tempfile.TemporaryDirectory() as root, self.subTest(value=value):
                with self.assertRaises(ValueError):
                    stream_receive(SimpleNamespace(read=lambda n: value), 3, root,
                                   'a.mp4', inspector=self.inspector)
        self.inspector.assert_not_called()

    def test_total_receive_timeout_leaves_no_ready_clip(self):
        clock = [100.]
        def slow_read(count):
            clock[0] += 121.
            return b'abc'
        with patch('video_input.upload.time.monotonic', side_effect=lambda: clock[0]):
            with self.assertRaises(TimeoutError):
                stream_receive(SimpleNamespace(read=slow_read), 3, self.root,
                               'a.mp4', inspector=self.inspector)
        folder, record = self.failed_record()
        self.assertIn('TimeoutError', record['error'])
        self.assertFalse((folder / 'video.mp4').exists())
        self.inspector.assert_not_called()

    def test_disconnect_error_is_preserved_and_diagnostic_retained(self):
        def disconnected(count):
            raise ConnectionResetError('peer disconnected')
        with self.assertRaises(ConnectionResetError):
            stream_receive(SimpleNamespace(read=disconnected), 3, self.root,
                           'a.mp4', inspector=self.inspector)
        _, record = self.failed_record()
        self.assertIn('peer disconnected', record['error'])

    def test_processing_starts_at_last_byte_before_decoded_probe(self):
        clock = [100.]
        def probe(path, *, decoded_frames):
            self.assertTrue(decoded_frames)
            self.assertTrue(path.exists())
            clock[0] += 10.
            return self.inspector.return_value
        with patch('video_input.upload.time.monotonic', side_effect=lambda: clock[0]), \
             patch('video_input.upload.time.time', return_value=1234567.):
            record = self.receive(inspector=probe)
        self.assertEqual(record['upload_complete_monotonic'], 100.)
        self.assertEqual(record['upload_complete_epoch'], 1234567.)
        self.assertEqual(record['upload_seconds'], 0.)
        self.assertEqual(clock[0] - record['upload_complete_monotonic'], 10.)

    def test_last_byte_timestamp_precedes_final_write_and_flush(self):
        clock = [100.]
        original_open = Path.open
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
                clock[0] += 2.
                return self.target.write(block)
            def flush(self):
                clock[0] += 3.
                return self.target.flush()
        def slow_open(path, *args, **kwargs):
            target = original_open(path, *args, **kwargs)
            return SlowSink(target) if path.name == 'video.partial' and args == ('xb',) else target
        def probe(path, *, decoded_frames):
            clock[0] += 5.
            return self.inspector.return_value
        with patch.object(Path, 'open', slow_open), \
                patch('video_input.upload.time.monotonic', side_effect=lambda: clock[0]), \
                patch('video_input.upload.time.time', side_effect=lambda: clock[0]+1000.):
            record = stream_receive(Body(b'abc'), 3, self.root, 'a.mp4', inspector=probe)
        self.assertEqual(record['upload_complete_monotonic'], 101.)
        self.assertEqual(record['upload_complete_epoch'], 1101.)
        self.assertEqual(record['upload_seconds'], 1.)
        self.assertEqual(clock[0]-record['upload_complete_monotonic'], 10.)

    def test_receipt_callback_publishes_once_only_after_the_complete_body(self):
        clock, events = [100.], []
        class Body(io.BytesIO):
            def read(self, count):
                clock[0] += 1.
                return super().read(count)
        def received(monotonic, epoch):
            events.append((monotonic, epoch))
            self.assertFalse(self.inspector.called)
            self.assertEqual(next(self.root.glob('upload_*/video.partial')).stat().st_size, 0)
        with patch('video_input.upload.time.monotonic', side_effect=lambda: clock[0]), \
                patch('video_input.upload.time.time', side_effect=lambda: clock[0]+1000.):
            record = stream_receive(Body(b'abcde'), 5, self.root, 'a.mp4', chunk_bytes=2,
                                    inspector=self.inspector, on_received=received)
        self.assertEqual(events, [(103., 1103.)])
        self.assertEqual(record['upload_complete_monotonic'], events[0][0])
        self.assertEqual(record['upload_complete_epoch'], events[0][1])
        self.assertEqual(record['upload_seconds'], 3.)

    def test_truncated_body_never_publishes_a_receipt_timestamp(self):
        received = Mock()
        with self.assertRaises(ValueError):
            stream_receive(io.BytesIO(b'ab'), 3, self.root, 'a.mp4',
                           inspector=self.inspector, on_received=received)
        received.assert_not_called()
        _, record = self.failed_record()
        self.assertNotIn('upload_complete_monotonic', record)
        self.assertNotIn('upload_complete_epoch', record)

    def test_decoded_probe_failure_does_not_publish_video(self):
        with self.assertRaises(ValueError):
            self.receive(inspector=Mock(side_effect=ValueError('invalid PTS')))
        folder, record = self.failed_record()
        self.assertIn('invalid PTS', record['error'])
        self.assertTrue((folder / 'video.partial').exists())
        self.assertFalse((folder / 'video.mp4').exists())

    def test_portrait_source_keeps_actual_geometry_and_no_usb_calibration(self):
        self.inspector.return_value = (dict(codec_name='h264', width=720, height=1280, fps=2.), [0., .5, 1.])
        record = self.receive()
        self.assertEqual(record['actual']['width'], 720)
        self.assertEqual(record['actual']['height'], 1280)
        self.assertAlmostEqual(record['preprocessing']['source_aspect_ratio'], .5625)
        self.assertIn('not supplied', record['preprocessing']['source_calibration'])

    def test_invalid_geometry_duration_pts_and_fps_fail_closed(self):
        base = self.inspector.return_value[0]
        cases = [(dict(base, width=63), [0., .5, 1.]),
                 (dict(base, width=4097), [0., .5, 1.]),
                 (dict(base, width=4096, height=4096), [0., .5, 1.]),
                 (dict(base, fps=float('nan')), [0., .5, 1.]),
                 (dict(base, fps=121), [0., .5, 1.]),
                 (base, [0., .5, float('nan')]), (base, [0., 1., .5]),
                 (base, [0., .1, .2]), (base, [0., 60., 121.]),
                 (base, [i / 200 for i in range(201)])]
        for info, times in cases:
            with tempfile.TemporaryDirectory() as root, self.subTest(info=info, times=times[:3]):
                with self.assertRaises(ValueError):
                    stream_receive(io.BytesIO(b'abc'), 3, root, 'a.mp4',
                                   inspector=lambda *a, **kw: (info, times))
                self.assertFalse(any(Path(root).glob('upload_*/video.mp4')))

    def test_disk_reserve_after_probe_still_blocks_handoff(self):
        self.space.side_effect = [SimpleNamespace(free=4 * 1024 ** 3),
                                 SimpleNamespace(free=4 * 1024 ** 3),
                                 SimpleNamespace(free=RESERVE_BYTES - 1)]
        with self.assertRaises(OSError):
            self.receive()
        folder, _ = self.failed_record()
        self.assertFalse((folder / 'video.mp4').exists())


if __name__ == '__main__':
    unittest.main()
