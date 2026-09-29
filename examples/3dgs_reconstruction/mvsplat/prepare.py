"""Prepare calibrated video frames from an existing CPU PoseSet for MVSplat."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pycolmap

from common import new_directory, save, sha


def centered_affine(width, height, cx, cy, output_size):
    scale = output_size / min(width, height)
    return np.array([[scale, 0, (output_size - 1) / 2 - scale * cx],
                     [0, scale, (output_size - 1) / 2 - scale * cy]], dtype=np.float64)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose-run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--context", type=int, nargs=2, default=[0, 29])
    p.add_argument("--targets", type=int, nargs="+", default=[7, 15, 22])
    p.add_argument("--size", type=int, default=256)
    a = p.parse_args()
    if a.size < 64 or a.size % 64 or set(a.context) & set(a.targets) or len(set(a.context)) != 2:
        p.error("Use a multiple of 64 and distinct context/held-out target frames")
    start = time.perf_counter()
    # Reuse the original undistorted resolution where available. PoseSet.project
    # is already reduced for the earlier 160-pixel training candidate.
    source = a.pose_run / "undistorted"
    sparse = source / "sparse"
    if not (sparse / "cameras.bin").exists():
        source = a.pose_run / "project"
        sparse = source / "sparse/0"
    rec = pycolmap.Reconstruction(sparse)
    images = sorted(rec.images.values(), key=lambda im: im.name)
    selected = a.context + a.targets
    if min(selected) < 0 or max(selected) >= len(images):
        p.error("Selected frame index outside registered sequence")
    poses = [im.cam_from_world().inverse().matrix() for im in images]
    poses = [np.vstack([c, [0, 0, 0, 1]]) for c in poses]
    points = np.array([pt.xyz for pt in rec.points3D.values()])
    depths = []
    for idx in a.context:
        z = ((points - poses[idx][:3, 3]) @ poses[idx][:3, :3])[:, 2]
        depths.extend(z[z > 0].tolist())
    if not depths:
        raise ValueError("No positive sparse depths for scale normalization")
    # Monocular SfM has arbitrary scale. Keep author's near=1/far=100 and
    # place the median observed depth at 10, recording the exact similarity.
    scale = 10.0 / float(np.median(depths))
    origin = poses[a.context[0]][:3, 3].copy()
    rotation = poses[a.context[0]][:3, :3].T.copy()
    out = new_directory(a.out)
    (out / "images").mkdir()
    views = []
    context_pixels, context_poses, context_k = [], [], []
    for idx in selected:
        im, c2w = images[idx], poses[idx].copy()
        cam = rec.cameras[im.camera_id]
        if cam.model_name != "PINHOLE":
            raise ValueError("MVSplat requires undistorted PINHOLE cameras")
        path = source / "images" / im.name
        rgb = cv2.imread(str(path))
        if rgb is None or rgb.shape[:2] != (cam.height, cam.width):
            raise ValueError("Image/camera dimensions disagree")
        affine = centered_affine(cam.width, cam.height, cam.principal_point_x,
                                 cam.principal_point_y, a.size)
        image = cv2.warpAffine(rgb, affine, (a.size, a.size), flags=cv2.INTER_LINEAR)
        name = Path(im.name).name
        cv2.imwrite(str(out / "images" / name), image)
        fx, fy = cam.focal_length_x * affine[0, 0], cam.focal_length_y * affine[1, 1]
        # MVSplat samples (pixel_index + 0.5) / size. Frozen FLCAM001 uses
        # integer pixel indices and principal point (size - 1)/2.
        k = np.array([[fx / a.size, 0, .5], [0, fy / a.size, .5], [0, 0, 1]])
        c2w[:3, :3] = rotation @ c2w[:3, :3]
        c2w[:3, 3] = scale * rotation @ (c2w[:3, 3] - origin)
        role = "context" if idx in a.context else "target"
        views.append(dict(name=name, index=idx, role=role, c2w=c2w.tolist(),
                          intrinsics_normalized=k.tolist(), width=a.size, height=a.size,
                          affine_source_to_output=affine.tolist(), source_sha256=sha(path),
                          prepared_sha256=sha(out / "images" / name)))
        if role == "context":
            context_pixels.append(image[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255)
            context_poses.append(c2w)
            context_k.append(k)
    np.savez(out / "context.npz", image=np.array(context_pixels, dtype=np.float32),
             extrinsics=np.array(context_poses, dtype=np.float32),
             intrinsics=np.array(context_k, dtype=np.float32),
             near=np.ones(2, np.float32), far=np.full(2, 100, np.float32))
    save(out / "input.json", dict(schema="mvsplat_context_v1", pose_run=str(a.pose_run.resolve()),
         pose_source=str(source.resolve()), context_indices=a.context, target_indices=a.targets,
         views=views, source_geometry_sha256={f.name: sha(f) for f in sparse.glob("*.bin")},
         context_sha256=sha(out / "context.npz"), near=1, far=100,
         normalization=dict(scale=scale, rotation=rotation.tolist(), origin=origin.tolist(),
                            positive_depth_percentiles_normalized=(np.percentile(depths, [0, 1, 50, 99, 100])*scale).tolist()),
         target_scope="Targets excluded from network context; used by the earlier SfM reconstruction. No scene training.",
         seconds=time.perf_counter()-start))
    print(json.dumps({"prepared": str(out), "context": a.context, "targets": a.targets, "size": a.size}))


if __name__ == "__main__":
    main()
