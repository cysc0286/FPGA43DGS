"""Same-input ARM board comparison of native/compiler variants, including errors."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from initialize.session import render_environment
from rendering.runtime import LiveRenderer
from rendering.benchmark import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--build', action='append', required=True, help='name=/absolute/binary')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--compact-build', action='append', default=[])
    p.add_argument('--depth-build', action='append', default=[])
    p.add_argument('--cpu-build', action='append', default=[])
    a = p.parse_args()
    if platform.machine() != 'aarch64' or not 1 <= a.repeats <= 100 or not 1 <= a.rounds <= 10:
        p.error('Bounded real-board comparison required')
    a.out.mkdir(parents=True, exist_ok=False)
    meta = json.loads((a.scene / 'manifest.json').read_text())
    cameras = [c['file'] for c in meta['cameras'] if c['role'] == 'target']
    builds = [entry.split('=', 1) for entry in a.build]
    result = dict(scene=str(a.scene), resolution='128x128', builds={}, rounds=a.rounds,
                  scope='camera to complete RGB, initialized scene, no archive or display',
                  scene_sha256=hashlib.sha256((a.scene/'model.ply').read_bytes()).hexdigest())
    baseline = {}
    for round_id in range(a.rounds):
        # Reverse order in alternate rounds to expose thermal/order drift.
        for name, binary in builds[::1 if round_id % 2 == 0 else -1]:
            folder = a.out / f'{name}_r{round_id}'
            folder.mkdir()
            records = []
            checks = []
            with LiveRenderer(binary, render_environment(), folder/'native.log', threads=4, batch=2,
                              compact_payload=name in a.compact_build,
                              depth_layout=name in a.depth_build,
                              cpu=name in a.cpu_build) as runtime:
                scene = runtime.load_scene(a.scene/'model.ply')
                first = runtime.render_camera((a.scene/cameras[0]).read_bytes()).metadata
                timed_hashes = []
                for camera in cameras*a.repeats:
                    frame = runtime.render_camera((a.scene/camera).read_bytes())
                    records.append(dict(frame.metadata, camera=camera))
                    timed_hashes.append((camera, hashlib.sha256(frame.rgb.tobytes()).hexdigest()))
                for camera in cameras:
                    frame = runtime.render_camera((a.scene/camera).read_bytes(), include_raw=True)
                    check = dict(frame.archive(folder/Path(camera).stem), camera=camera)
                    if round_id == 0 and name == builds[0][0]:
                        baseline[camera] = (frame.rgb.copy(), frame.raw)
                    reference, raw = baseline[camera]
                    error = frame.rgb.astype(np.float64) - reference
                    mse = np.mean(error**2)
                    check.update(raw_equals_baseline=raw == frame.raw,
                                 rgb_equals_baseline=np.array_equal(frame.rgb, reference),
                                 rgb_max_error=float(np.abs(error).max()),
                                 rgb_mean_error=float(np.abs(error).mean()),
                                 psnr_vs_baseline_db=None if mse == 0 else float(10*np.log10(255**2/mse)))
                    checks.append(check)
            hashes = {c['camera']: c['rgb_sha256'] for c in checks}
            if not all(hashes[c] == value for c, value in timed_hashes):
                raise RuntimeError('Timed output differs from archived output')
            entry = dict(round=round_id, scene=scene, first=first, records=records, checks=checks,
                         summary=summarize(records), all_timed_frames_match=True)
            build = result['builds'].setdefault(name, dict(binary=binary,
                binary_sha256=hashlib.sha256(Path(binary).read_bytes()).hexdigest(), runs=[]))
            build['runs'].append(entry)
            build['summary'] = summarize([r for run in build['runs'] for r in run['records']])
            (a.out/'results.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(dict(build=name, round=round_id, summary=entry['summary'],
                    exact=[c['raw_equals_baseline'] for c in checks])), flush=True)
    result['complete'] = True
    (a.out/'results.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
