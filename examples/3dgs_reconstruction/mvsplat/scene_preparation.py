"""Orchestrate video decoding and poses; numerical implementations live in their modules."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from common import new_directory, save, sha


from video_input.decode import frame_indices, decode_selected
from pose_estimation.geometry import mutual_ratio_matches, estimate_pair, estimate_target


def main(argv=None, on_context_ready=None, on_first_target_ready=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--focal-ratio", type=float, default=0.9,
                   help="Approximate focal length / decoded frame width; calibrate for a real camera")
    p.add_argument("--context", type=int, nargs=2, default=[0, -1])
    p.add_argument("--targets", type=int, nargs="+",
                   help="Default: quarter, middle and three-quarter video positions")
    p.add_argument("--min-inliers", type=int, default=30)
    a = p.parse_args(argv)
    if (a.size < 64 or a.size % 64 or a.width < 64 or a.threads < 1
            or not np.isfinite(a.focal_ratio) or a.focal_ratio <= 0 or a.min_inliers < 8):
        p.error("Invalid image size, width, threads or focal ratio")
    started = time.perf_counter()
    timing = {}
    decode_started = time.perf_counter()
    cv2.setNumThreads(a.threads)
    frames, count, fps, context, targets = decode_selected(
        a.video, a.context, a.targets, a.width, a.threads)
    timing["decode_seconds"] = time.perf_counter()-decode_started
    h, w = frames[context[0]].shape[:2]
    if any(frame.shape[:2] != (h, w) for frame in frames.values()):
        raise ValueError("Video frame dimensions changed")
    focal = a.focal_ratio * w
    intrinsic = np.array([[focal, 0, (w-1)/2], [0, focal, (h-1)/2], [0, 0, 1]], np.float64)
    camera1, landmarks, keys0, descriptors0, pair_info = estimate_pair(
        frames[context[0]], frames[context[1]], intrinsic, a.min_inliers, timing)
    poses = {context[0]: np.eye(4), context[1]: camera1}
    out = new_directory(a.out)
    (out / "images").mkdir()
    views, pixels, extrinsics, intrinsics = [], [], [], []
    scale = a.size / min(w, h)
    affine = np.array([[scale, 0, (a.size-1)/2-scale*(w-1)/2],
                       [0, scale, (a.size-1)/2-scale*(h-1)/2]], np.float64)
    k = np.array([[focal*scale/a.size, 0, .5],
                  [0, focal*scale/a.size, .5], [0, 0, 1]], np.float64)
    for i in context:
        name = f"frame_{i:06d}.png"
        image = cv2.warpAffine(frames[i], affine, (a.size, a.size), flags=cv2.INTER_LINEAR)
        if not cv2.imwrite(str(out / "images" / name), image):
            raise RuntimeError("Could not write prepared image")
        views.append(dict(name=name, index=i, role="context", c2w=poses[i].tolist(),
                          intrinsics_normalized=k.tolist(), width=a.size, height=a.size,
                          affine_source_to_output=affine.tolist(),
                          prepared_sha256=sha(out / "images" / name)))
        pixels.append(image[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)/255)
        extrinsics.append(poses[i])
        intrinsics.append(k)
    np.savez(out / "context.npz", image=np.array(pixels, np.float32),
             extrinsics=np.array(extrinsics, np.float32),
             intrinsics=np.array(intrinsics, np.float32),
             near=np.ones(2, np.float32), far=np.full(2, 100, np.float32))
    context_hash = sha(out / "context.npz")
    if on_context_ready is not None:
        on_context_ready(out, context_hash)
    targets_info = {}
    first_target = targets[0]
    deferred_targets = targets[1:]
    first_target_published = False
    for position, i in enumerate(targets):
        poses[i], targets_info[str(i)] = estimate_target(
            frames[i], keys0, descriptors0, landmarks, intrinsic, a.min_inliers,
            timing, label=f"target_{i}")
        name = f"frame_{i:06d}.png"
        image = cv2.warpAffine(frames[i], affine, (a.size, a.size), flags=cv2.INTER_LINEAR)
        if not cv2.imwrite(str(out / "images" / name), image):
            raise RuntimeError("Could not write prepared image")
        views.append(dict(name=name, index=i, role="target", c2w=poses[i].tolist(),
                          intrinsics_normalized=k.tolist(), width=a.size, height=a.size,
                          affine_source_to_output=affine.tolist(),
                          prepared_sha256=sha(out / "images" / name)))
        if position == 0:
            initial_meta = dict(schema="mvsplat_context_v1", pose_source="two_view_sift_essential_pnp",
                input_video=str(a.video.resolve()), input_video_sha256=sha(a.video),
                source_frames=count, fps=fps, context_indices=context, target_indices=[first_target],
                deferred_target_indices=deferred_targets,
                calibration=dict(kind="approximate", focal_ratio=a.focal_ratio,
                                 focal_px=focal, intrinsic=intrinsic.tolist(), distortion="assumed zero"),
                geometry=dict(pair=pair_info, targets={str(first_target): targets_info[str(first_target)]}),
                views=list(views), context_sha256=context_hash, near=1, far=100,
                normalization=dict(median_triangulated_depth=10., scale=pair_info["translation_scale"]),
                target_scope="Initial snapshot contains the first target only; deferred targets remain in input.json.",
                timing=dict(timing), seconds=time.perf_counter()-started)
            save(out / "input.initial.json", initial_meta)
            first_target_published = True
            if on_first_target_ready is not None:
                on_first_target_ready(out, context_hash, first_target)
    timing["remaining_target_pose_seconds"] = sum(
        timing.get(f"target_{i}_total_seconds", 0.) for i in deferred_targets)
    timing["remaining_target_pnp_seconds"] = sum(
        timing.get(f"target_{i}_pnp_seconds", 0.) for i in deferred_targets)
    save(out / "input.json", dict(schema="mvsplat_context_v1", pose_source="two_view_sift_essential_pnp",
         input_video=str(a.video.resolve()), input_video_sha256=sha(a.video),
         source_frames=count, fps=fps, context_indices=context, target_indices=targets,
         calibration=dict(kind="approximate", focal_ratio=a.focal_ratio,
                          focal_px=focal, intrinsic=intrinsic.tolist(), distortion="assumed zero"),
         geometry=dict(pair=pair_info, targets=targets_info), views=views,
          context_sha256=context_hash, near=1, far=100,
         normalization=dict(median_triangulated_depth=10., scale=pair_info["translation_scale"]),
         target_scope="Targets excluded from network context; positioned with two-view landmarks and PnP, not full SfM.",
         timing=timing, initial_target_published=first_target_published,
         seconds=time.perf_counter()-started))
    print(json.dumps(dict(prepared=str(out), seconds=time.perf_counter()-started,
                          context=context, targets=targets, geometry=pair_info)))


if __name__ == "__main__":
    main()
