"""Stage04: replay a verified camera clip through the formal NPU/FPGA pipeline."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from .common import (MVSPLAT, base_report, digest, finish_cli, load_capture,
                     new_output, parser, require_board, stop_process, write_json)

PATH_KEYS = ('python', 'weights', 'vendor', 'renderer', 'live_renderer', 'calibration',
             'partition_bundle', 'npu_library', 'npu_acceptance', 'oracle_graphs', 'worker_python')


def load_deployment(path):
    from gaussian_generation.npu_branch.config import BackendConfig
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    allowed = set(PATH_KEYS) | {'backend', 'partitions', 'buffer_policy', 'buffer_limit_mib'}
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError('Deployment JSON contains unknown settings')
    if data.get('backend') != 'npu':
        raise ValueError('Stage04 requires the formal NPU backend; CPU fallback is not accepted here')
    for key in PATH_KEYS:
        if data.get(key) is not None:
            value = Path(data[key]).expanduser()
            data[key] = (path.parent / value).resolve() if not value.is_absolute() else value.resolve()
    for key in ('weights', 'vendor', 'renderer', 'live_renderer', 'calibration'):
        if not data.get(key):
            raise ValueError('Missing deployment setting: ' + key)
    data['python'] = data.get('python') or Path(sys.executable)
    backend = BackendConfig.from_args(SimpleNamespace(**data))
    calibration = json.loads(data['calibration'].read_text(encoding='utf-8'))
    if calibration.get('schema') != 'uvc_pinhole_v1' or len(calibration.get('image_size', [])) != 2:
        raise ValueError('Require the camera calibration matching the capture mode')
    return data, backend, calibration


def pipeline_command(settings, backend, clip, run, *, scope='scan', host='0.0.0.0', port=18083, viewer=True):
    command = [str(settings['python']), str(MVSPLAT / 'initialize/measure.py'),
               '--out', str(run.parent / 'measurement.json'), '--rss-mib', '0',
               '--pss-mib', '700', '--reserve-mib', '96', '--timeout', '1800', '--',
               str(settings['python']), str(MVSPLAT.parent / 'pipeline.py'), 'initialize',
               '--video', str(clip), '--out', str(run), '--auto-select',
               '--size', '128', '--threads', '4', '--prepare-threads', '1',
               '--serial-prepare', '--render-profile', 'mainline']
    for key in ('weights', 'vendor', 'renderer', 'live_renderer', 'calibration'):
        command += ['--' + key.replace('_', '-'), str(settings[key])]
    if scope == 'scan':
        command += ['--ring-pairs', '8', '--ring-auto-coverage', '--ring-candidates', '64',
                    '--ring-max-gaussians', '65536']
    else:
        # The rig rotates at a fixed centre. Do not accidentally send its smoke
        # clip to the ordinary two-view path that requires translation/parallax.
        command += ['--ring-pairs', '1', '--ring-candidates', '16', '--ring-max-gaussians', '65536']
    if viewer:
        command += ['--interactive', '--viewer-host', host, '--viewer-port', str(port)]
    return command + backend.cli()


def verified_first_frame(run, expected_video):
    event_file = Path(run) / 'events.json'
    if not event_file.is_file():
        return None
    events = json.loads(event_file.read_text(encoding='utf-8'))['events']
    names = ('READY', 'VIDEO_COMPLETE', 'SCENE_READY', 'FRAME_COMPLETE')
    primary = [item for item in events if item.get('event') in names]
    if not any(item.get('event') == 'FRAME_COMPLETE' for item in primary):
        return None
    if [item['event'] for item in primary] != list(names):
        raise ValueError('Unexpected lifecycle ordering in reconstructed scene')
    ready, receipt, scene, frame = primary
    if ready.get('backend') != 'npu' or Path(receipt.get('path', '')).resolve() != Path(expected_video).resolve():
        raise ValueError('Lifecycle backend or input identity differs from this test')
    matches = [path for path in Path(run).glob('board/**/frame.bin')
               if path.is_file() and path.stat().st_size == 128 * 128 * 3
               and digest(path) == frame.get('frame_sha256')]
    if not matches:
        raise ValueError('FRAME_COMPLETE has no matching complete 128x128 RGB frame')
    timing = dict(scene_preparation_seconds=frame['monotonic'] - receipt['monotonic'],
                  gaussian_scene_seconds=scene['monotonic'] - receipt['monotonic'],
                  first_render_seconds=frame['monotonic'] - scene['monotonic'])
    if any(not math.isfinite(value) or value < 0 for value in timing.values()):
        raise ValueError('Negative lifecycle duration')
    return dict(frame=str(matches[0]), frame_sha256=frame['frame_sha256'], timing=timing,
                backend_from_ready='npu',
                timing_scope='manual replay: VIDEO_COMPLETE to FRAME_COMPLETE; excludes recording, warmup and manual gap')


def main(argv=None):
    p = parser(__doc__)
    p.add_argument('--capture', type=Path, required=True, help='Stage03 capture_handoff.json or original capture.json')
    p.add_argument('--deployment', type=Path, required=True, help='Use the existing qualified board deployment paths')
    p.add_argument('--scope', choices=('pair', 'scan'), default='scan')
    p.add_argument('--viewer-host', default='0.0.0.0')
    p.add_argument('--viewer-port', type=int, default=18083)
    p.add_argument('--no-viewer', action='store_true')
    a = p.parse_args(argv)
    if not 1 <= a.viewer_port <= 65535:
        p.error('Invalid viewer port')
    receipt, clip = load_capture(a.capture)
    settings, backend, calibration = load_deployment(a.deployment)
    requested = clip['requested']
    if calibration['image_size'] != [requested['width'], requested['height']]:
        p.error('Calibration resolution differs from the captured camera mode; do not reuse unrelated intrinsics')
    out = new_output(a.out)
    report = base_report(4, a)
    process = None
    run = out / 'scene'
    command = pipeline_command(settings, backend, clip['path'], run, scope=a.scope,
                               host=a.viewer_host, port=a.viewer_port, viewer=not a.no_viewer)
    report.update(command=command, scope=('one selected rotation pair, one target frame' if a.scope == 'pair' else
                  'existing sparse fixed-centre rotation reconstruction; multi-height quality requires visual checks'),
                  capture_receipt=str(receipt), source_sha256=clip['sha256'],
                  capture_mode={key: requested[key] for key in ('width', 'height', 'fps')},
                  calibration_sha256=digest(settings['calibration']),
                  viewer_port=None if a.no_viewer else a.viewer_port,
                  quality_measured=False, multirow_uniformity_verified=False)
    write_json(out / 'report.json', report)
    try:
        if not a.execute:
            print('Stage04: DRY RUN; verified camera source and deployment options; no NPU/FPGA opened')
            return
        require_board()
        for key in PATH_KEYS:
            value = settings.get(key)
            if value is not None and not value.exists():
                raise FileNotFoundError('Deployment asset not found: ' + str(value))
        env = dict(os.environ, PYTHONUNBUFFERED='1')
        if int(env.get('MGS_NPU_DDR_GUARD_MIB', '64')) < 64:
            raise ValueError('Do not weaken the existing 64 MiB NPU/FPGA device workspace separation')
        env.setdefault('MGS_NPU_DDR_GUARD_MIB', '64')
        with (out / 'pipeline.log').open('xb') as log:
            process = subprocess.Popen(command, cwd=MVSPLAT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if not report['complete']:
                    result = verified_first_frame(run, clip['path'])
                    if result:
                        report.update(complete=True, first_frame=result)
                        write_json(out / 'report.json', report)
                        print('FRAME_COMPLETE verified. %.3f s (%s). Viewer details: %s' %
                              (result['timing']['scene_preparation_seconds'], a.scope, out / 'pipeline.log'), flush=True)
                time.sleep(.2)
        if not report['complete']:
            result = verified_first_frame(run, clip['path'])
            if result:
                report.update(complete=True, first_frame=result)
        if process.returncode or not report['complete']:
            raise RuntimeError('Pipeline failed or no verified first frame; inspect pipeline.log and measurement.json')
        measurement = json.loads((out / 'measurement.json').read_text(encoding='utf-8'))
        if measurement.get('status') != 'passed':
            raise RuntimeError('Process-tree memory/runtime guard did not pass')
    except KeyboardInterrupt:
        report['operator_stopped_viewer'] = bool(report['complete'])
        if not report['complete']:
            report['error'] = 'Interrupted before first frame'
        print('Stopped; evidence retained in ' + str(out))
        if not report['complete']:
            raise SystemExit(130)
    except BaseException as error:
        report['complete'] = False
        report['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        stop_process(process, grace=12)
        if (out / 'measurement.json').is_file():
            report['measurement'] = json.loads((out / 'measurement.json').read_text(encoding='utf-8'))
        write_json(out / 'report.json', report)


if __name__ == '__main__':
    finish_cli(main)
