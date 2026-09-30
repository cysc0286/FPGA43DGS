"""Compare HLS C models on deterministic sparse/dense/edge Gaussian streams.

This verifies arithmetic/ordering only, not RTL, routing or board performance.
Both executables must be built by the existing pipeline/tb.cpp CLI.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import subprocess


def vectors():
    rng = random.Random(303064)
    records = []
    shapes = [(1, 1), (5, 16), (8, 8), (9, 9), (16, 2), (16, 16)]
    for tile in range(36):
        w, h = shapes[tile % len(shapes)]
        n = [0, 1, 2, 7, 24, 48][tile // 6]
        ox, oy = (tile % 6) * 16, (tile // 6) * 16
        header = bytearray(64)
        struct.pack_into('<IHHH', header, 0, n, ox, oy, w | (h << 5))
        struct.pack_into('<eee', header, 10, .1, .2, .3)
        records.append(header)
        for j in range(n):
            a = 10 ** rng.uniform(-3, .5)
            c = 10 ** rng.uniform(-3, .5)
            b = rng.uniform(-.6, .6) * (a * c) ** .5
            opacity = [0., .001, .02, .5, .99, 1.2][j % 6]
            # Includes Gaussian groups with absent subtiles, mixed contribution
            # colors, early termination, anisotropy and edge-tile clipping.
            x, y = ox + rng.uniform(-12, 28), oy + rng.uniform(-12, 28)
            values = (x, y, a, b, c, opacity, rng.random(), rng.random(), rng.random())
            word = bytearray(64)
            struct.pack_into('<9f', word, 0, *values)
            records.append(word)
    return b''.join(records), 36


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    data, tiles = vectors()
    source = (a.out / 'input.flk').resolve()
    source.write_bytes(data)
    env = dict(os.environ)
    vivado = Path('D:/Xilinx/Vivado/2018.3')
    runtime = [vivado/p for p in ['win64/lib/csim', 'win64/tools/fpo_v7_0', 'bin', 'lib/win64.o', 'bin/unwrapped/win64.o',
                                  'msys64/mingw64/bin']]
    env['PATH'] = ';'.join(str(p) for p in runtime) + ';' + env.get('PATH', '')
    result = dict(scope='HLS C arithmetic/ordering only; not RTL or board timing',
                  tiles=tiles, records=len(data)//64, input_sha256=hashlib.sha256(data).hexdigest(),
                  reference=str(a.reference.resolve()), candidate=str(a.candidate.resolve()),
                  executable_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest()
                                     for k,v in [('reference',a.reference),('candidate',a.candidate)]},
                  modes=[], passed=False)
    try:
        for mode in range(6):
            outputs = []
            for label, exe in [('reference', a.reference), ('candidate', a.candidate)]:
                dest = (a.out / f'{label}_mode{mode}.raw').resolve()
                run = subprocess.run([str(exe.resolve()), str(source), str(dest), str(tiles), str(mode)],
                                     cwd=exe.resolve().parent, env=env, capture_output=True,
                                     timeout=180, text=True)
                (a.out / f'{label}_mode{mode}.log').write_text(run.stdout+run.stderr)
                if run.returncode:
                    raise RuntimeError(f'{label} mode {mode}: exit {run.returncode}')
                payload = dest.read_bytes()
                if len(payload) != tiles*4096:
                    raise RuntimeError('Output size mismatch')
                outputs.append(payload)
            equal = outputs[0] == outputs[1]
            result['modes'].append(dict(mode=mode, exact=equal,
                                        sha256=[hashlib.sha256(v).hexdigest() for v in outputs]))
            if not equal:
                raise RuntimeError(f'Mode {mode}: candidate differs from frozen C reference')
        result['passed'] = True
    finally:
        (a.out/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
