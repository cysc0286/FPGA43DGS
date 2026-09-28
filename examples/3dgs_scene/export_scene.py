"""Export complete prepared frames from the locked official CUDA renderer.

No Gaussian replication, training, or edited upstream kernels. CPU/FPGA inputs
contain attributes and ordered indices only; reference pixels stay on the PC.
"""
import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--source-hashes', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    frozen = json.loads(args.source_hashes.read_text())
    for name in ('cuda_rasterizer/rasterizer_impl.cu',
                 'cuda_rasterizer/forward.cu', 'rasterize_points.cu'):
        name = 'submodules/diff-gaussian-rasterization/' + name
        if sha(args.source / name) != frozen[name]:
            raise ValueError('Official source drift: ' + name)
    sys.path.insert(0, str(args.source.resolve()))
    import numpy as np
    import torch
    from gaussian_renderer import GaussianModel, render
    from diff_gaussian_rasterization import _C
    from scene.cameras import MiniCam
    from utils.graphics_utils import focal2fov, getProjectionMatrix

    args.output.mkdir(parents=True, exist_ok=False)
    ply = args.model / 'point_cloud/iteration_7000/point_cloud.ply'
    model = GaussianModel(3)
    model.load_ply(str(ply))
    names = ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation')
    original = {name: getattr(model, name).detach() for name in names}
    total = len(original['_xyz'])
    permutation = np.random.default_rng(20260926).permutation(total)
    cameras = json.loads((args.model / 'cameras.json').read_text())
    cases = [(n, 0, 320) for n in (10000, 50000, 100000, total)]
    cases.append((total, 10, 320))
    # Larger image for a CPU scaling check; FPGA testing is selected after timing.
    cases.append((total, 0, 640))
    manifest = dict(schema='gs-prepared-scene-v1', source_gaussians=total,
        model_sha256=sha(ply), cameras_sha256=sha(args.model / 'cameras.json'),
        source_hashes={name: frozen[name] for name in frozen},
        selection='nested RNG seed 20260926 permutation prefixes, original PLY order retained',
        training=False, replication=False, preprocessing='official GPU, outside board timings',
        thresholds=dict(rgb_max_abs=1e-4, rgb_mean_abs=1e-5,
                        transmittance_max_abs=1e-5, last_contributor='exact'), cases=[])

    class Buffer:
        def __init__(self, tensor):
            assert tensor.data_ptr() % 128 == 0
            self.raw = tensor.cpu().numpy()
            self.offset = 0

        def take(self, dtype, shape):
            self.offset = (self.offset + 127) & ~127
            length = int(np.prod(shape))
            size = length * np.dtype(dtype).itemsize
            assert self.offset + size <= self.raw.nbytes
            result = np.frombuffer(self.raw, dtype=dtype, count=length,
                                   offset=self.offset).reshape(shape)
            self.offset += size
            return result

    for n, view, w in cases:
        selected = np.sort(permutation[:n])
        index = torch.as_tensor(selected, dtype=torch.long, device='cuda')
        for name in names:
            setattr(model, name, torch.nn.Parameter(original[name][index], requires_grad=False))
        entry = cameras[view]
        h = round(w * entry['height'] / entry['width'])
        c2w = np.eye(4)
        c2w[:3, :3] = entry['rotation']
        c2w[:3, 3] = entry['position']
        matrix = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device='cuda').T
        fx, fy = focal2fov(entry['fx'], entry['width']), focal2fov(entry['fy'], entry['height'])
        projection = getProjectionMatrix(.01, 100., fx, fy).T.cuda()
        cam = MiniCam(w, h, fy, fx, .01, 100., matrix, matrix @ projection)
        bg = torch.zeros(3, device='cuda')
        empty = torch.empty(0, device='cuda')
        with torch.no_grad():
            count, color, radii, geometry, binning, image, _ = _C.rasterize_gaussians(
                bg, model.get_xyz, empty, model.get_opacity, model.get_scaling,
                model.get_rotation, 1.0, empty, cam.world_view_transform,
                cam.full_proj_transform, math.tan(fx * .5), math.tan(fy * .5),
                h, w, model.get_features, 3, cam.camera_center, False, False, False)
            reference = render(cam, model, SimpleNamespace(compute_cov3D_python=False,
                convert_SHs_python=False, debug=False, antialiasing=False), bg)['render']
            torch.cuda.synchronize()
            assert torch.equal(color.clamp(0, 1), reference)
        geom = Buffer(geometry)
        depth = geom.take('<f4', (n,))
        geom.take('u1', (n, 3))
        geom.take('<i4', (n,))
        xy = geom.take('<f4', (n, 2))
        geom.take('<f4', (n, 6))
        conic = geom.take('<f4', (n, 4))
        rgb = geom.take('<f4', (n, 3))
        bins = Buffer(binning)
        point_list = bins.take('<u4', (count,))
        bins.take('<u4', (count,))
        keys = bins.take('<u8', (count,))
        img = Buffer(image)
        trans = img.take('<f4', (h, w))
        contrib = img.take('<u4', (h, w))
        tiles = ((w + 15) // 16) * ((h + 15) // 16)
        ranges = img.take('<u4', (h * w, 2))[:tiles]
        assert count > 0 and point_list.max() < n
        assert np.all(keys[1:] >= keys[:-1])
        assert int((ranges[:, 1] - ranges[:, 0]).sum()) == count
        active, remapped = np.unique(point_list, return_inverse=True)
        assert np.all(radii.cpu().numpy()[active] > 0)
        np.testing.assert_array_equal((keys & 0xffffffff).astype('<u4'), depth[point_list].view('<u4'))
        for tile, (begin, end) in enumerate(ranges):
            if end > begin:
                assert np.all((keys[begin:end] >> 32) == tile)
        attributes = np.column_stack([xy[active], conic[active], rgb[active], depth[active]])
        assert np.isfinite(attributes).all()
        assert len(np.unique(selected[active])) == len(active)
        rows = np.empty(len(active), dtype=[('id', '<u4'), ('values', '<f4', (10,))])
        rows['id'], rows['values'] = selected[active], attributes
        name = f'n{n}_v{view}_w{w}'
        case = args.output / name
        case.mkdir()
        inp = case / 'scene.bin'
        with inp.open('wb') as f:
            f.write(b'GSSCN001')
            f.write(struct.pack('<7I3f', w, h, 16, n, len(active), count, tiles, 0., 0., 0.))
            f.write(rows.tobytes())
            f.write(ranges.astype('<u4').tobytes())
            f.write(remapped.astype('<u4').tobytes())
        expected = np.empty(w * h, dtype=[('rgba', '<f4', (4,)), ('last', '<u4')])
        expected['rgba'][:, :3] = color.permute(1, 2, 0).cpu().numpy().reshape(-1, 3)
        expected['rgba'][:, 3] = trans.reshape(-1)
        expected['last'] = contrib.reshape(-1)
        out = case / 'official.bin'
        out.write_bytes(b'GSSOUT01' + struct.pack('<2I', w, h) + expected.tobytes())
        counts = ranges[:, 1] - ranges[:, 0]
        record = dict(name=name, camera=view, width=w, height=h, pixels=w*h,
            scene_gaussians=n, projected_unique_gaussians=len(active), tile_entries=count,
            tiles=tiles, max_tile_candidates=int(counts.max()),
            mean_tile_candidates=float(counts.mean()), input_bytes=inp.stat().st_size,
            input_sha256=sha(inp), official_sha256=sha(out))
        (case / 'case.json').write_text(json.dumps(record, indent=2))
        manifest['cases'].append(record)
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
