"""Host-only acceptance regressions; no SDK, camera, PWM or network is opened."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import common, stage01_motor, stage02_trajectory, stage03_sync_capture, stage04_reconstruction


class FakeClock:
    def __init__(self):
        self.value = 10.

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class FakeWorker:
    def join(self, _timeout):
        pass

    def is_alive(self):
        return False


class FakeRecorder:
    def __init__(self, clock, order):
        self.clock, self.order = clock, order
        self.state, self.error = 'idle', None
        self.worker = FakeWorker()
        self.clip = None
        self.failed = False

    def snapshot(self):
        return b'jpeg', 1, 30.

    def packet(self):
        if self.failed:
            raise RuntimeError('camera disconnected')
        return b'jpeg', 1, 30., self.clock.now()

    def start(self):
        self.order.append('recording')
        self.state = 'recording'

    def status(self):
        return dict(state=self.state, error=self.error)

    def stop(self):
        self.order.append('stop_recording')
        self.state = 'recorded'
        self.clip = dict(path='fixture.mkv')


class FakeProcess:
    def __init__(self, recorder, order, *, fail=False, disconnect=False, forever=False):
        self.recorder, self.order = recorder, order
        self.returncode = None
        self.fail, self.disconnect, self.forever = fail, disconnect, forever
        self.polls = 0

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        self.polls += 1
        if self.disconnect:
            self.recorder.failed = True
        if self.forever or self.polls == 1:
            return None
        self.order.append('motion_end')
        self.returncode = 1 if self.fail else 0
        return self.returncode

    def terminate(self):
        self.order.append('terminate_motion')
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode


class StageTests(unittest.TestCase):
    def capture_fixture(self, root):
        folder = root / 'clip_fixture'
        folder.mkdir()
        video = folder / 'capture.mkv'
        video.write_bytes(b'camera fixture, not a playable video')
        record = dict(complete=True, path=str(video.resolve()), sha256=common.digest(video),
                      requested=dict(width=1280, height=720, fps=30.))
        receipt = folder / 'capture.json'
        common.write_json(receipt, record)
        return receipt, video

    def deployment_fixture(self, root):
        calibration = root / 'calibration.json'
        common.write_json(calibration, dict(schema='uvc_pinhole_v1', image_size=[1280, 720]))
        config = root / 'deployment.json'
        common.write_json(config, dict(backend='npu', weights='weights', vendor='vendor',
                          renderer='renderer', live_renderer='live', calibration=str(calibration),
                          partition_bundle='bundle.json', npu_library='bridge.so', partitions=['backbone_cnn']))
        return config

    def test_stage01_dry_run_never_launches_native_driver(self):
        with tempfile.TemporaryDirectory() as temp, patch('subprocess.Popen') as spawn, redirect_stdout(io.StringIO()):
            out = Path(temp) / 'stage01'
            stage01_motor.main(['--out', str(out), '--action', 'both'])
            spawn.assert_not_called()
            report = json.loads((out / 'report.json').read_text(encoding='utf-8'))
            self.assertFalse(report['complete'])
            self.assertTrue(report['dry_run'])
            self.assertEqual(len(report['commands']), 2)

    def test_stage02_reuses_full_production_bow_plan(self):
        with tempfile.TemporaryDirectory() as temp, patch('subprocess.Popen') as spawn, redirect_stdout(io.StringIO()):
            out = Path(temp) / 'stage02'
            stage02_trajectory.main(['--out', str(out)])
            spawn.assert_not_called()
            meta = json.loads((out / 'trajectory.csv.json').read_text(encoding='utf-8'))
            self.assertEqual(meta['profile']['scan']['tilt_rows_deg'], [30, 0, -30])
            self.assertEqual(meta['samples'], 833)
            self.assertAlmostEqual(meta['nominal_duration_seconds'], 16.66)

    def test_stage03_dry_run_does_not_start_camera_or_native(self):
        with tempfile.TemporaryDirectory() as temp, patch('subprocess.Popen') as spawn, redirect_stdout(io.StringIO()):
            stage03_sync_capture.main(['--out', str(Path(temp) / 'stage03')])
            spawn.assert_not_called()
            report = json.loads((Path(temp)/'stage03/report.json').read_text(encoding='utf-8'))
            self.assertAlmostEqual(report['planned_video_seconds'], 17.66)
            self.assertAlmostEqual(report['capture_budget_seconds'], 19.66)
            self.assertEqual(report['target_video_seconds'], 20)

    def test_current_plan_is_shared_at_60fps_and_capture_mode_reaches_stage04(self):
        with tempfile.TemporaryDirectory() as temp, patch('subprocess.Popen') as spawn, redirect_stdout(io.StringIO()):
            root = Path(temp)
            stage02_trajectory.main(['--out', str(root/'stage02')])
            stage03_sync_capture.main(['--out', str(root/'stage03'), '--fps', '60'])
            plans = [json.loads((root/name/'trajectory.csv.json').read_text(encoding='utf-8'))
                     for name in ('stage02', 'stage03')]
            for key in ('profile_sha256', 'csv_sha256'):
                self.assertEqual(plans[0][key], plans[1][key])
            report = json.loads((root/'stage03/report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['camera']['fps'], 60)
            self.assertAlmostEqual(report['planned_video_seconds'], 17.66)
            self.assertEqual(plans[0]['profile']['scan']['pan_speed_deg_s'], 20)
            receipt, _ = self.capture_fixture(root)
            clip = json.loads(receipt.read_text(encoding='utf-8'))
            clip['requested']['fps'] = 60.
            common.write_json(receipt, clip)
            stage04_reconstruction.main(['--capture', str(receipt), '--deployment',
                str(self.deployment_fixture(root)), '--out', str(root/'stage04')])
            report = json.loads((root/'stage04/report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['capture_mode'], dict(width=1280, height=720, fps=60.))
            self.assertFalse(report['complete'])
            spawn.assert_not_called()

    def test_stage03_rejects_slow_profile_with_15_second_recording_limit(self):
        with tempfile.TemporaryDirectory() as temp, patch('subprocess.Popen') as spawn:
            with self.assertRaisesRegex(ValueError, 'too short'):
                stage03_sync_capture.main(['--out', str(Path(temp) / 'stage03'), '--max-seconds', '15', '--target-seconds', '15',
                    '--profile', str(common.MVSPLAT/'video_input/camera_scan_bow_conservative.json')])
            spawn.assert_not_called()

    def test_fast_scan_fits_15_second_capture_budget_without_hardware(self):
        with tempfile.TemporaryDirectory() as temp, patch('subprocess.Popen') as spawn, redirect_stdout(io.StringIO()):
            out = Path(temp) / 'stage03'
            stage03_sync_capture.main(['--out', str(out), '--max-seconds', '15', '--target-seconds', '15',
                '--profile', str(common.MVSPLAT/'video_input/camera_scan_bow_fast15.json')])
            spawn.assert_not_called()
            report = json.loads((out/'report.json').read_text(encoding='utf-8'))
            self.assertAlmostEqual(report['planned_video_seconds'], 12.66)
            self.assertAlmostEqual(report['capture_budget_seconds'], 14.66)
            self.assertEqual(report['target_video_seconds'], 15)
            self.assertFalse(report['complete'])

    def test_live_motor_commands_require_wiring_flag(self):
        with tempfile.TemporaryDirectory() as temp:
            for entry, extra in ((stage01_motor.main, ['--action', 'pan']), (stage02_trajectory.main, []),
                                 (stage03_sync_capture.main, [])):
                with self.assertRaisesRegex(ValueError, 'wiring'):
                    entry(['--out', str(Path(temp) / 'unused'), '--execute', *extra])

    def test_capture_precedes_motion_and_recording_stops_after_motion(self):
        clock, order = FakeClock(), []
        recorder = FakeRecorder(clock, order)
        def launch():
            order.append('motion_start')
            return FakeProcess(recorder, order)
        result = stage03_sync_capture.capture_and_scan(recorder, launch, clock=clock.now, sleep=clock.sleep)
        self.assertEqual(order, ['recording', 'motion_start', 'motion_end', 'stop_recording'])
        self.assertAlmostEqual(result['dispatch_after_recording_ready_seconds'], .5)
        self.assertFalse(result['exposure_hardware_synchronized'])

    def test_camera_failure_stops_running_motion_and_recording(self):
        clock, order = FakeClock(), []
        recorder = FakeRecorder(clock, order)
        with self.assertRaisesRegex(RuntimeError, 'disconnected'):
            stage03_sync_capture.capture_and_scan(recorder,
                lambda: FakeProcess(recorder, order, disconnect=True, forever=True),
                clock=clock.now, sleep=clock.sleep)
        self.assertIn('terminate_motion', order)
        self.assertEqual(order[-1], 'stop_recording')

    def test_camera_never_ready_does_not_move(self):
        clock, order = FakeClock(), []
        recorder = FakeRecorder(clock, order)
        with patch.object(recorder, 'snapshot', side_effect=RuntimeError('waiting')):
            with self.assertRaises(TimeoutError):
                stage03_sync_capture.capture_and_scan(recorder, lambda: self.fail('must not move'),
                    clock=clock.now, sleep=clock.sleep)
        self.assertNotIn('recording', order)

    def test_failed_native_execution_does_not_publish_capture_success(self):
        clock, order = FakeClock(), []
        recorder = FakeRecorder(clock, order)
        with self.assertRaisesRegex(RuntimeError, 'Native scan failed'):
            stage03_sync_capture.capture_and_scan(recorder,
                lambda: FakeProcess(recorder, order, fail=True), clock=clock.now, sleep=clock.sleep)
        self.assertEqual(order[-1], 'stop_recording')

    def test_motion_timeout_stops_motor(self):
        clock, order = FakeClock(), []
        recorder = FakeRecorder(clock, order)
        with self.assertRaises(TimeoutError):
            stage03_sync_capture.capture_and_scan(recorder,
                lambda: FakeProcess(recorder, order, forever=True), motion_timeout=.1,
                clock=clock.now, sleep=clock.sleep)
        self.assertIn('terminate_motion', order)

    def test_modified_source_video_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            receipt, video = self.capture_fixture(Path(temp))
            common.load_capture(receipt)
            video.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash-verified'):
                common.load_capture(receipt)

    def test_modified_handoff_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipt, _ = self.capture_fixture(root)
            handoff = root / 'handoff.json'
            common.write_json(handoff, dict(schema='camera_acceptance_handoff_v1', complete=True,
                              capture_receipt=str(receipt), capture_receipt_sha256=common.digest(receipt)))
            common.load_capture(handoff)
            receipt.write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed'):
                common.load_capture(handoff)

    def test_stage04_dry_run_preserves_npu_memory_guards_and_calibration(self):
        with tempfile.TemporaryDirectory() as temp, patch('subprocess.Popen') as spawn, redirect_stdout(io.StringIO()):
            root = Path(temp)
            receipt, video = self.capture_fixture(root)
            deployment = self.deployment_fixture(root)
            out = root / 'stage04'
            stage04_reconstruction.main(['--capture', str(receipt), '--deployment', str(deployment), '--out', str(out), '--scope', 'pair'])
            spawn.assert_not_called()
            report = json.loads((out / 'report.json').read_text(encoding='utf-8'))
            cmd = report['command']
            for flag, value in (('--backend', 'npu'), ('--pss-mib', '700'), ('--reserve-mib', '96'), ('--video', str(video))):
                self.assertEqual(cmd[cmd.index(flag)+1], value)
            self.assertIn('--calibration', cmd)
            self.assertEqual(cmd[cmd.index('--ring-pairs')+1], '1')
            self.assertFalse(report['complete'])

    def test_scan_scope_uses_actual_multi_pair_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings, backend, _ = stage04_reconstruction.load_deployment(self.deployment_fixture(root))
            cmd = stage04_reconstruction.pipeline_command(settings, backend, root / 'video.mkv', root / 'scene', scope='scan')
            self.assertIn('--ring-auto-coverage', cmd)
            self.assertEqual(cmd[cmd.index('--ring-candidates')+1], '64')
            self.assertEqual(cmd[cmd.index('--ring-max-gaussians')+1], '65536')

    def test_both_scopes_are_accepted_by_the_published_initialize_cli(self):
        from argparse import ArgumentParser
        from initialize.run import main as initialize_main

        class ParsedBeforeHardware(Exception):
            pass

        original_parse = ArgumentParser.parse_args

        def parse_only(parser, args=None, namespace=None):
            parsed = original_parse(parser, args, namespace)
            self.assertEqual(parsed.backend, 'npu')
            self.assertIn(parsed.ring_pairs, (1, 8))
            raise ParsedBeforeHardware

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings, backend, _ = stage04_reconstruction.load_deployment(self.deployment_fixture(root))
            for scope in ('pair', 'scan'):
                with self.subTest(scope=scope):
                    command = stage04_reconstruction.pipeline_command(
                        settings, backend, root / 'video.mkv', root / 'scene', scope=scope)
                    args = command[command.index('initialize') + 1:]
                    with patch.object(ArgumentParser, 'parse_args', parse_only), self.assertRaises(ParsedBeforeHardware):
                        initialize_main(args)

    def test_wrong_capture_calibration_mode_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipt, _ = self.capture_fixture(root)
            deployment = self.deployment_fixture(root)
            calibration = root / 'calibration.json'
            common.write_json(calibration, dict(schema='uvc_pinhole_v1', image_size=[640, 480]))
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                stage04_reconstruction.main(['--capture', str(receipt), '--deployment', str(deployment), '--out', str(root/'stage04')])

    def test_cpu_backend_is_not_labelled_npu(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = self.deployment_fixture(root)
            data = json.loads(config.read_text(encoding='utf-8'))
            data['backend'] = 'cpu'
            common.write_json(config, data)
            with self.assertRaisesRegex(ValueError, 'NPU backend'):
                stage04_reconstruction.load_deployment(config)

    def test_output_inside_repository_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            common.new_output(common.REPOSITORY / 'private_test_should_not_be_created')

    def test_frame_success_requires_correct_lifecycle_identity_and_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frame = root / 'board/first_frame/frame.bin'
            frame.parent.mkdir(parents=True)
            frame.write_bytes(bytes(128*128*3))
            events = [dict(event='READY', monotonic=1., backend='npu'),
                      dict(event='VIDEO_COMPLETE', monotonic=2., path=str(root/'capture.mkv')),
                      dict(event='SCENE_READY', monotonic=5.),
                      dict(event='FRAME_COMPLETE', monotonic=5.1, frame_sha256=common.digest(frame))]
            common.write_json(root/'events.json', dict(events=events))
            result = stage04_reconstruction.verified_first_frame(root, root/'capture.mkv')
            self.assertAlmostEqual(result['timing']['scene_preparation_seconds'], 3.1)
            events[-1]['frame_sha256'] = 'wrong'
            common.write_json(root/'events.json', dict(events=events))
            with self.assertRaisesRegex(ValueError, 'matching'):
                stage04_reconstruction.verified_first_frame(root, root/'capture.mkv')

    def test_motion_log_rejects_partial_or_watchdog_expired_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'motion.csv'
            path.write_text('event,elapsed_ms,status\nend,40,0\n', encoding='utf-8')
            self.assertEqual(stage03_sync_capture.validate_motion_log(path, 1)['samples'], 1)
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                stage03_sync_capture.validate_motion_log(path, 2)
            path.write_text('event,elapsed_ms,status\nend,40,256\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'watchdog'):
                stage03_sync_capture.validate_motion_log(path, 1)

    def test_real_recorder_state_machine_hands_off_its_original_receipt(self):
        from types import SimpleNamespace
        from video_input.recording import CameraRecorder
        import time
        class Preview:
            def __init__(self, *args, **kwargs):
                self.process = SimpleNamespace(poll=lambda: None, returncode=0)
                if kwargs:
                    Path(kwargs['recording']).write_bytes(b'original camera packet fixture')
            def snapshot(self):
                return b'jpeg', 60, 30.
            def packet(self):
                return self.snapshot() + (time.monotonic(),)
            def finish_recording(self):
                pass
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as temp, patch('video_input.recording.inspect_clip',
                return_value=(dict(codec_name='mjpeg', width=1280, height=720, fps=30.), [i/30 for i in range(46)])):
            recorder = CameraRecorder('/dev/fake', 1280, 720, 30., Path(temp), max_seconds=60, factory=Preview)
            try:
                order = []
                result = stage03_sync_capture.capture_and_scan(recorder,
                    lambda: FakeProcess(recorder, order), pre_roll=.01, post_roll=.01)
                receipt, record = common.load_capture(Path(recorder.clip['path']).with_name('capture.json'))
                self.assertEqual(receipt.name, 'capture.json')
                self.assertTrue(record['complete'])
                self.assertEqual(record['sha256'], common.digest(record['path']))
                self.assertIn('capture_validated_monotonic', result)
            finally:
                recorder.close()


if __name__ == '__main__':
    unittest.main()
