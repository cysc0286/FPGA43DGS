"""Pack a camera pose and intrinsics for the board-side FLICKER front end."""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cameras', type=Path, required=True)
    p.add_argument('--view', type=int, required=True)
    p.add_argument('--width', type=int, required=True)
    p.add_argument('--height', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Preserve existing camera evidence')
    cameras = json.loads(a.cameras.read_text())
    camera = cameras[a.view]
    if camera['id'] != a.view or not all(0 < x <= 2048 for x in (a.width, a.height)):
        raise ValueError('Unexpected camera or output dimensions')
    values = [float(x) for row in camera['rotation'] for x in row]
    values += [float(x) for x in camera['position']]
    values += [float(camera['fx']), float(camera['fy'])]
    if len(values) != 14:
        raise ValueError('Invalid camera schema')
    blob = b'FLCAM001' + struct.pack('<4I14d', a.width, a.height,
                                   camera['width'], camera['height'], *values)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_bytes(blob)
    record = {'view': a.view, 'width': a.width, 'height': a.height,
              'source_sha256': hashlib.sha256(a.cameras.read_bytes()).hexdigest(),
              'camera_sha256': hashlib.sha256(blob).hexdigest(),
              'source_camera': camera, 'schema': 'FLCAM001-v1'}
    a.output.with_suffix('.json').write_text(json.dumps(record, indent=2))
    print(json.dumps({k: record[k] for k in ('view', 'width', 'height', 'camera_sha256')}))


if __name__ == '__main__':
    main()
