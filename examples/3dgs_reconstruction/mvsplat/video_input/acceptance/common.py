"""Evidence and process helpers shared by the four acceptance stages."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

MVSPLAT = Path(__file__).resolve().parents[2]
REPOSITORY = MVSPLAT.parents[2]
DEFAULT_PROFILE = MVSPLAT / 'video_input/camera_scan_bow.json'
DEFAULT_NATIVE = MVSPLAT / 'video_input/camera_servo_native'


def parser(description, *, motor=False):
    result = argparse.ArgumentParser(description=description)
    result.add_argument('--out', type=Path, required=True, help='Fresh evidence directory outside Git checkout')
    result.add_argument('--execute', action='store_true', help='Execute on ARM board; otherwise preview only')
    if motor:
        result.add_argument('--native', type=Path, default=DEFAULT_NATIVE)
        result.add_argument('--wiring-j17-23-21', action='store_true')
    return result


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def new_output(path):
    path = Path(path).resolve()
    if path == REPOSITORY or REPOSITORY in path.parents:
        raise ValueError('Keep camera media/evidence outside the Git checkout')
    path.mkdir(parents=True, exist_ok=False)
    return path


def require_board():
    if platform.system() != 'Linux' or platform.machine().lower() not in ('aarch64', 'arm64'):
        raise RuntimeError('--execute requires the Linux ARM board; no device was opened')


def require_motion(args):
    if not args.wiring_j17_23_21:
        raise ValueError('Motion requires --wiring-j17-23-21 (pan 23, tilt 21, ground 2)')


def stop_process(process, *, grace=3):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def run_logged(command, log_path, timeout):
    process = None
    with Path(log_path).open('xb') as log:
        try:
            process = subprocess.Popen([str(x) for x in command], cwd=MVSPLAT,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            code = process.wait(timeout=timeout)
            if code:
                raise RuntimeError('Command failed with exit code %s; see %s' % (code, log_path))
            return code
        finally:
            stop_process(process)


def scan_command(native, plan, log):
    return [str(Path(native).resolve()), 'scan', str(plan), '--execute',
            '--wiring-j17-23-21', '--log', str(log)]


def load_capture(path):
    """Accept the original completed recorder receipt, or stage03's bound handoff."""
    path = Path(path).resolve()
    record = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(record, dict):
        raise ValueError('Capture receipt must be a JSON object')
    if record.get('schema') == 'camera_acceptance_handoff_v1':
        if record.get('complete') is not True:
            raise ValueError('Stage03 did not finish successfully')
        receipt_path = Path(record['capture_receipt']).resolve()
        if digest(receipt_path) != record['capture_receipt_sha256']:
            raise ValueError('Capture receipt changed after stage03')
        path = receipt_path
        record = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(record, dict):
            raise ValueError('Capture receipt must be a JSON object')
    video = Path(record.get('path', '')).resolve()
    if (path.name != 'capture.json' or record.get('complete') is not True
            or video.parent != path.parent or video.name != 'capture.mkv'
            or not video.is_file() or digest(video) != record.get('sha256')):
        raise ValueError('Require the original complete, hash-verified capture.json and capture.mkv')
    mode = record.get('requested', {})
    if not all(key in mode for key in ('width', 'height', 'fps')):
        raise ValueError('Capture receipt is missing the negotiated camera mode')
    return path, record


def finish_cli(function, argv=None):
    try:
        return function(argv)
    except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error)) from error


def base_report(stage, args):
    return dict(schema='camera_acceptance_stage_v1', stage=stage, complete=False,
                dry_run=not args.execute, started_epoch=time.time(),
                physical_motion_observed=False, operator_visual_acceptance='pending')
