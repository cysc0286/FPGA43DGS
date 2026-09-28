"""Portable ARM entry point for the frozen 3DGS renderer v1 (Python 3.8+)."""
import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for relative, record in manifest['files'].items():
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('Package checksum mismatch: ' + relative)
    return manifest


def write_ppm(source, output):
    raw = source.read_bytes()
    if raw[:8] != b'GSSOUT01':
        raise ValueError('Framebuffer ABI')
    w, h = struct.unpack_from('<II', raw, 8)
    if len(raw) != 16 + w * h * 20:
        raise ValueError('Framebuffer size')
    rgb = bytearray(w * h * 3)
    for i, values in enumerate(struct.iter_unpack('<4fI', raw[16:])):
        if not all(math.isfinite(v) for v in values[:4]):
            raise ValueError('Nonfinite framebuffer')
        for c in range(3):
            rgb[i * 3 + c] = round(max(0, min(1, values[c])) * 255)
    output.write_bytes(('P6\n%d %d\n255\n' % (w, h)).encode() + rgb)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=ROOT / 'data/model.ply')
    p.add_argument('--camera', type=Path, default=ROOT / 'data/v0.bin')
    p.add_argument('--out', type=Path)
    p.add_argument('--backend', choices=['fpga', 'cpu_dense', 'cpu_base'], default='fpga')
    p.add_argument('--work-root', type=Path, default=Path('/dev/shm'))
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    manifest = verify()
    print('Package checksums verified', flush=True)
    if a.verify_only:
        return
    if platform.machine() not in ('aarch64', 'arm64'):
        p.error('Runtime binaries require the ARM64 board')
    if a.out is None:
        p.error('--out must name a new output directory')
    model, camera, dest = a.model.resolve(), a.camera.resolve(), a.out.resolve()
    if not model.is_file() or not camera.is_file() or not a.work_root.is_dir():
        p.error('Model, camera or work-root is absent')
    camera_raw = camera.read_bytes()
    if len(camera_raw) != 136 or camera_raw[:8] != b'FLCAM001':
        p.error('Camera must use FLCAM001, see INTERFACE.md')
    dest.mkdir(parents=True, exist_ok=False)
    record = dict(package=manifest['name'], backend=a.backend, model_sha256=sha(model),
                  camera_sha256=sha(camera), complete=False,
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    env = dict(os.environ)
    sdk = env.get('ICRAFT_SDK_ROOT', '/root/heterogs_npu/sdk_3.36.1/usr')
    env['LD_LIBRARY_PATH'] = sdk + '/lib/aarch64-linux-gnu:' + env.get('LD_LIBRARY_PATH', '')
    try:
        with tempfile.TemporaryDirectory(prefix='gs-render-', dir=str(a.work_root)) as temporary:
            work = Path(temporary)
            with (dest / 'run.log').open('w') as log:
                begin = time.monotonic_ns()
                subprocess.run([str(ROOT/'bin/attributes'), str(model), str(camera), str(work/'attr.bin')],
                               check=True, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
                projected = time.monotonic_ns()
                subprocess.run([str(ROOT/'bin/group_sort'), str(work/'attr.bin'), str(work/'scene.bin')],
                               check=True, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
                sorted_time = time.monotonic_ns()
                args = (['mode', '2', 'scene.bin', 'frame', 'pipeline', '8', '1', '0']
                        if a.backend == 'fpga' else
                        ['cpu', 'scene.bin', 'frame', 'dense' if a.backend == 'cpu_dense' else 'base', '4', '1', '0', '0'])
                subprocess.run([str(ROOT/'bin/render')] + args, cwd=str(work), check=True,
                               env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
                finished = time.monotonic_ns()
            record.update(attributes_ms=(projected-begin)/1e6, group_sort_ms=(sorted_time-projected)/1e6,
                          render_ms=(finished-sorted_time)/1e6, total_ms=(finished-begin)/1e6,
                          scene_sha256=sha(work/'scene.bin'), frame_sha256=sha(work/'frame.bin'))
            for name in ('frame.bin', 'frame_timing.csv'):
                shutil.copy2(str(work/name), str(dest/name))
            write_ppm(dest/'frame.bin', dest/'frame.ppm')
        record['complete'] = True
    finally:
        (dest/'result.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
