"""Check the timed framebuffer against the matching frozen HLS numerical contract."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--frame', type=Path, required=True)
    p.add_argument('--raw-board', type=Path, required=True)
    a = p.parse_args()
    meta = json.loads((a.reference / 'input.json').read_text())
    execution = json.loads((a.reference / 'execution.json').read_text())
    board = json.loads((a.raw_board / 'result.json').read_text())
    source = a.reference / 'expected_mode2.raw'
    if not execution['complete'] or not board['passed'] or board['mode'] != 2:
        raise ValueError('Incomplete reference/board check')
    if sha(source) != execution['modes']['2']['output_sha256'] or sha(source) != board['output_sha256']:
        raise ValueError('Golden differs from hardware')
    if sha(a.raw_board / 'output.raw') != sha(source) or board['input_sha256'] != meta['input_sha256']:
        raise ValueError('Input/output provenance differs')
    w, h = meta['width'], meta['height']
    if meta['tiles'] != list(range(((w+15)//16)*((h+15)//16))):
        raise ValueError('Full-frame reference required')
    data = np.fromfile(source, dtype=[('rgba', '<f2', (4,)), ('last', '<u4'), ('tag', '<u4')])
    if not np.array_equal(data['tag'], np.arange(len(meta['tiles'])*256, dtype=np.uint32)):
        raise ValueError('Output tags')
    frame = np.zeros(w*h, dtype=[('rgba', '<f4', (4,)), ('last', '<u4')])
    for t in meta['tiles']:
        x = t % ((w+15)//16)*16
        for py in range(16):
            y = t // ((w+15)//16)*16+py
            if y >= h:
                continue
            width = min(16, w-x)
            src = data[t*256+py*16:t*256+py*16+width]
            frame[y*w+x:y*w+x+width]['rgba'] = src['rgba']
            frame[y*w+x:y*w+x+width]['last'] = src['last']
    expected = b'GSSOUT01'+struct.pack('<II', w, h)+frame.tobytes()
    actual = a.frame.read_bytes()
    result = dict(reference=str(a.reference.resolve()), frame=str(a.frame.resolve()),
                  raw_board=str(a.raw_board.resolve()), scene_sha256=meta['scene_sha256'],
                  expected_frame_sha256=hashlib.sha256(expected).hexdigest(),
                  actual_frame_sha256=sha(a.frame), bitwise_equal=actual == expected,
                  raw_output_bytes=len(data)*16,
                  differing_bytes=sum(x != y for x, y in zip(actual, expected))+abs(len(actual)-len(expected)))
    path = a.reference / 'timed_frame_check.json'
    if path.exists():
        raise ValueError('Preserve existing verification')
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result['bitwise_equal']:
        raise ValueError('Timed framebuffer does not match HLS numerical contract')


if __name__ == '__main__':
    main()
