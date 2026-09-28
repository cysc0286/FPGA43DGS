"""Compare board-projected active attributes against the locked CUDA export."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
ATTR = np.dtype([('q', '<f4', (10,)), ('rect', '<u2', (4,)), ('valid', '<u4')])
GAUSSIAN = np.dtype([('id', '<u4'), ('q', '<f4', (10,))])
NAMES = ['x', 'y', 'a', 'b', 'c', 'opacity', 'r', 'g', 'blue', 'depth']


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evidence', type=Path, required=True)
    a = p.parse_args()
    stage = json.loads((a.evidence/'result.json').read_text())
    if not stage['complete']:
        raise ValueError('Incomplete board preprocess')
    report = {'stage': str(a.evidence.resolve()), 'cases': []}
    for item in stage['camera_runs']:
        camera = json.loads(Path(item['camera']).with_suffix('.json').read_text())
        name = f'n559263_v{camera["view"]}_w{camera["width"]}'
        scene = REPO/'examples/3dgs_scene/data/20260926T135717'/name/'scene.bin'
        raw = scene.read_bytes()
        if raw[:8] != b'GSSCN001':
            raise ValueError('Official scene ABI')
        w,h,tile,n,active,entries,tiles = struct.unpack_from('<7I', raw, 8)
        if (w,h,n) != (camera['width'],camera['height'],559263):
            raise ValueError('Scene/camera mismatch')
        official = np.frombuffer(raw, GAUSSIAN, active, 48)
        attr_path = a.evidence/item['output']
        if sha(attr_path) != item['output_sha256']:
            raise ValueError('Board result drift')
        attr = np.memmap(attr_path, mode='r', dtype=ATTR, offset=24, shape=(n,))
        with attr_path.open('rb') as f:
            header = f.read(24)
        if header[:8] != b'FLKATR01' or struct.unpack_from('<4I',header,8)[:3] != (n,w,h):
            raise ValueError('Board attribute ABI')
        ids = official['id']
        missing = ids[attr['valid'][ids] != 1]
        if not np.isfinite(attr['q'][ids]).all():
            raise ValueError('Nonfinite active board attributes')
        stats = {}
        for i, field in enumerate(NAMES):
            error = np.abs(attr['q'][ids, i].astype(np.float64)
                           - official['q'][:, i].astype(np.float64))
            stats[field] = {'mean_abs': float(error.mean()),
                            'p99_abs': float(np.percentile(error, 99)),
                            'max_abs': float(error.max()),
                            'exact_count': int(np.count_nonzero(error == 0))}
        case = {'case': name, 'official_active': active,
                'board_valid': int(struct.unpack_from('<I',header,20)[0]),
                'missing_official_active': len(missing),
                'missing_ids_first': missing[:20].tolist(),
                'fields': stats, 'scene_sha256': sha(scene),
                'attribute_sha256': item['output_sha256'],
                'board_timing': item['timing']}
        report['cases'].append(case)
        print(json.dumps({'case': name, 'missing': len(missing),
                          'board_timing': item['timing'],
                          'max_fields': {k: v['max_abs'] for k,v in stats.items()}}, indent=2))
    (a.evidence/'validation.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
