"""Stage03: wait for recording frames, execute motion, then hand off the verified clip."""
import csv
import math
from pathlib import Path
import subprocess
import time

from video_input.camera_scan import write_plan
from video_input.capture import validate_capture
from .common import (DEFAULT_PROFILE, MVSPLAT, base_report, digest, finish_cli,
                     load_capture, new_output, parser, require_board, require_motion,
                     scan_command, stop_process, write_json)


def capture_and_scan(recorder, launch_motion, *, pre_roll=.5, post_roll=.5,
                     motion_timeout=55., clock=time.monotonic, sleep=time.sleep):
    """Software ordering, never a claim of simultaneous sensor/PWM hardware triggering."""
    process = None
    evidence = dict(synchronization='recording_frames_ready_then_motion_dispatch',
                    exposure_hardware_synchronized=False)

    def recording_alive():
        status = recorder.status()
        if status['state'] != 'recording':
            raise RuntimeError('Recording ended during scan: ' + str(status.get('error') or status['state']))
        recorder.packet()  # Detect a stalled frame stream, not only a live FFmpeg process.

    def hold(seconds):
        deadline = clock() + seconds
        while clock() < deadline:
            recording_alive()
            sleep(min(.05, max(0., deadline - clock())))

    try:
        deadline = clock() + 15
        while True:
            try:
                recorder.snapshot()
                break
            except RuntimeError:
                if clock() >= deadline:
                    raise TimeoutError('No fresh USB camera preview; motion was not started')
                sleep(.05)
        recorder.start()
        deadline = clock() + 15
        while recorder.status()['state'] != 'recording':
            state = recorder.status()
            if state['state'] not in ('starting', 'recording'):
                raise RuntimeError('Recording did not start: ' + str(state.get('error') or state['state']))
            if clock() >= deadline:
                raise TimeoutError('Recording start timed out; motion was not started')
            sleep(.05)
        recording_alive()
        evidence['recording_ready_monotonic'] = clock()
        evidence['recording_ready_frame_received_monotonic'] = recorder.packet()[3]
        hold(pre_roll)
        recording_alive()
        evidence['motion_dispatch_monotonic'] = clock()
        process = launch_motion()
        deadline = clock() + motion_timeout
        while process.poll() is None:
            recording_alive()
            if clock() >= deadline:
                raise TimeoutError('Motion exceeded its capture-time budget; stopping motion and recording')
            sleep(.02)
        if process.returncode:
            raise RuntimeError('Native scan failed; see native.log')
        evidence['motion_exit_monotonic'] = clock()
        hold(post_roll)
        recording_alive()
        evidence['stop_recording_requested_monotonic'] = clock()
        recorder.stop()
        recorder.worker.join(45)
        if recorder.worker.is_alive():
            raise TimeoutError('Recording finalization did not finish')
        state = recorder.status()
        if state['state'] != 'recorded' or not recorder.clip:
            raise RuntimeError('Clip did not pass recorder checks: ' + str(state.get('error')))
        evidence['capture_validated_monotonic'] = clock()
        evidence['dispatch_after_recording_ready_seconds'] = (
            evidence['motion_dispatch_monotonic'] - evidence['recording_ready_monotonic'])
        evidence['motion_process_wall_seconds'] = evidence['motion_exit_monotonic'] - evidence['motion_dispatch_monotonic']
        evidence['note'] = ('Dispatch includes native identity/clock checks before first PWM update. '
                            'Frame receive times are not sensor exposure times; use an instrument for hardware timing.')
        return evidence
    finally:
        stop_process(process)
        if recorder.status()['state'] in ('starting', 'recording', 'stopping'):
            recorder.stop()


def validate_motion_log(path, samples):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != samples or rows[-1].get('event') != 'end':
        raise ValueError('Native motion log is incomplete')
    elapsed = [float(row['elapsed_ms']) for row in rows]
    if (not all(math.isfinite(value) for value in elapsed)
            or any(b < a for a, b in zip(elapsed, elapsed[1:]))
            or any(int(row['status']) & 0x100 for row in rows)):
        raise ValueError('Motion log has invalid time order or an expired watchdog')
    return dict(samples=len(rows), last_sample_elapsed_seconds=elapsed[-1] / 1000,
                register_log_only=True, physical_angles_measured=False)


def main(argv=None):
    p = parser(__doc__, motor=True)
    p.add_argument('--profile', type=Path, default=DEFAULT_PROFILE)
    p.add_argument('--device', default='/dev/video0')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--fps', type=float, default=30.)
    p.add_argument('--max-seconds', type=float, default=20., help='Safety ceiling, not a fixed recording length')
    p.add_argument('--target-seconds', type=float, default=20., help='Report the actual clip duration against this goal')
    p.add_argument('--pre-roll', type=float, default=.5)
    p.add_argument('--post-roll', type=float, default=.5)
    a = p.parse_args(argv)
    validate_capture(a.width, a.height, a.fps, a.max_seconds, 0)
    if not math.isfinite(a.target_seconds) or not 1 <= a.target_seconds <= a.max_seconds:
        p.error('Target duration must be finite and within 1..max-seconds')
    if not all(math.isfinite(x) and 0 <= x <= 5 for x in (a.pre_roll, a.post_roll)):
        p.error('Pre/post-roll must be finite and between 0 and 5 seconds')
    if a.execute:
        require_motion(a)
    out = new_output(a.out)
    report = base_report(3, a)
    recorder = None
    try:
        plan = out / 'trajectory.csv'
        summary = write_plan(a.profile, plan)
        required = summary['nominal_duration_seconds'] + a.pre_roll + a.post_roll + 2
        if required > a.max_seconds:
            raise ValueError('Capture is too short for this trajectory; need at least %.2f seconds' % required)
        command = scan_command(a.native, plan, out / 'motion.csv')
        report.update(plan=summary, command=command, target_video_seconds=a.target_seconds,
                      planned_video_seconds=summary['nominal_duration_seconds'] + a.pre_roll + a.post_roll,
                      capture_budget_seconds=required,
                      camera=dict(device=a.device, width=a.width,
                      height=a.height, fps=a.fps, max_seconds=a.max_seconds),
                      note='No model inference or reconstruction is started by stage03.')
        if a.execute:
            require_board()
            from video_input.recording import CameraRecorder
            recorder = CameraRecorder(a.device, a.width, a.height, a.fps, out / 'captures',
                                      max_seconds=a.max_seconds, auto_process=False)
            with (out / 'native.log').open('xb') as log:
                def launch():
                    return subprocess.Popen(command, cwd=MVSPLAT, stdin=subprocess.DEVNULL,
                                            stdout=log, stderr=subprocess.STDOUT)
                report['synchronization'] = capture_and_scan(recorder, launch,
                    pre_roll=a.pre_roll, post_roll=a.post_roll,
                    motion_timeout=a.max_seconds - a.pre_roll - a.post_roll - 1)
            report['motion'] = validate_motion_log(out / 'motion.csv', summary['samples'])
            receipt, clip = load_capture(Path(recorder.clip['path']).with_name('capture.json'))
            needed_duration = (report['synchronization']['motion_exit_monotonic'] -
                               report['synchronization']['recording_ready_monotonic'] + a.post_roll)
            if clip['duration_seconds'] < needed_duration - .5:
                raise ValueError('Saved clip does not cover the observed motion interval; handoff refused')
            recorder.close()
            recorder = None
            handoff = dict(schema='camera_acceptance_handoff_v1', complete=True,
                           capture_receipt=str(receipt), capture_receipt_sha256=digest(receipt),
                           video_path=clip['path'], video_sha256=clip['sha256'],
                           motion_log_sha256=digest(out / 'motion.csv'),
                           actual_video_seconds=clip['duration_seconds'],
                           target_video_seconds=a.target_seconds,
                           target_met=clip['duration_seconds'] <= a.target_seconds,
                           synchronization=report['synchronization'])
            write_json(out / 'capture_handoff.json', handoff)
            report.update(complete=True, capture_receipt=str(receipt), video_path=clip['path'],
                          actual_video_seconds=clip['duration_seconds'],
                          target_met=clip['duration_seconds'] <= a.target_seconds,
                          video_sha256=clip['sha256'])
        print('Stage03: ' + ('capture_handoff.json is ready for stage04' if a.execute else
                             'DRY RUN; no camera or motor opened; recording limit %.1f seconds' % a.max_seconds))
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        try:
            if recorder is not None:
                recorder.close()
        except BaseException as error:
            report.update(complete=False, cleanup_error=type(error).__name__ + ': ' + str(error))
            raise
        finally:
            write_json(out / 'report.json', report)


if __name__ == '__main__':
    finish_cli(main)
