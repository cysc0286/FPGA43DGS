"""Prepare a short video for MVSplat with bounded two-view CPU geometry."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from common import new_directory, save, sha


def mutual_ratio_matches(a, b, ratio=0.75):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(a, b, k=2)
    backward = matcher.knnMatch(b, a, k=2)
    reverse = {m.queryIdx: m.trainIdx for pair in backward
               if len(pair) == 2 for m, other in [pair] if m.distance < ratio * other.distance}
    return [m for pair in forward if len(pair) == 2 for m, other in [pair]
            if m.distance < ratio * other.distance and reverse.get(m.trainIdx) == m.queryIdx]


def frame_indices(count, context, targets):
    context = [i if i >= 0 else count + i for i in context]
    targets = ([count // 4, count // 2, 3 * count // 4] if targets is None
               else [i if i >= 0 else count + i for i in targets])
    selected = context + targets
    if len(context) != 2 or not targets or len(set(selected)) != len(selected):
        raise ValueError("Use two distinct context frames and at least one distinct target")
    if min(selected) < 0 or max(selected) >= count:
        raise ValueError("Selected frame outside video")
    return context, targets


def decode_selected(video, context, targets, width, threads):
    if not hasattr(cv2, "CAP_PROP_N_THREADS"):
        raise RuntimeError("OpenCV FFmpeg decoder thread control is required")
    cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, threads])
    if not cap.isOpened():
        raise ValueError("Cannot open video")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if count < 3 or fps <= 0:
        raise ValueError("Video must contain at least three timed frames")
    context, targets = frame_indices(count, context, targets)
    selected = set(context + targets)
    frames = {}
    try:
        for i in range(max(selected) + 1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Video decode failed at frame {i}")
            if i in selected:
                h, w = frame.shape[:2]
                scale = min(1., width / w)
                frames[i] = cv2.resize(frame, (round(w * scale), round(h * scale)),
                                       interpolation=cv2.INTER_AREA)
    finally:
        cap.release()
    return frames, count, fps, context, targets


def estimate_pair(first, second, intrinsic, min_inliers):
    sift = cv2.SIFT_create(nfeatures=2048)
    k0, d0 = sift.detectAndCompute(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), None)
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(second, cv2.COLOR_BGR2GRAY), None)
    if d0 is None or d1 is None:
        raise ValueError("No SIFT descriptors in context frames")
    matches = mutual_ratio_matches(d0, d1)
    if len(matches) < min_inliers:
        raise ValueError(f"Only {len(matches)} mutual matches")
    p0 = np.float64([k0[m.queryIdx].pt for m in matches])
    p1 = np.float64([k1[m.trainIdx].pt for m in matches])
    essential, mask = cv2.findEssentialMat(p0, p1, intrinsic, cv2.RANSAC,
                                          prob=0.999, threshold=1.0)
    if essential is None:
        raise ValueError("Essential matrix estimation failed")
    inliers, rotation, translation, mask = cv2.recoverPose(essential, p0, p1,
                                                           intrinsic, mask=mask)
    if inliers < min_inliers:
        raise ValueError(f"Only {inliers} cheirality-consistent matches")
    chosen = mask.ravel() != 0
    p0, p1 = p0[chosen], p1[chosen]
    chosen_matches = [m for m, good in zip(matches, chosen) if good]
    projection0 = intrinsic @ np.column_stack((np.eye(3), np.zeros(3)))
    projection1 = intrinsic @ np.column_stack((rotation, translation))
    homogeneous = cv2.triangulatePoints(projection0, projection1, p0.T, p1.T).T
    points = homogeneous[:, :3] / homogeneous[:, 3:]
    depth1 = (rotation @ points.T + translation).T[:, 2]
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0) & (depth1 > 0)
    if valid.sum() < min_inliers:
        raise ValueError("Too few positive triangulated points")
    median_depth = float(np.median(points[valid, 2]))
    scale = 10.0 / median_depth
    points *= scale
    camera1 = np.eye(4)
    camera1[:3, :3] = rotation.T
    camera1[:3, 3] = (-rotation.T @ translation).ravel() * scale
    landmark_by_keypoint = {m.queryIdx: xyz for m, xyz, good in
                            zip(chosen_matches, points, valid) if good}
    diagnostics = dict(context_features=[len(k0), len(k1)], mutual_matches=len(matches),
                       essential_pose_inliers=int(inliers), positive_points=int(valid.sum()),
                       median_depth_before_normalization=median_depth,
                       translation_scale=scale, baseline_after_normalization=float(np.linalg.norm(camera1[:3, 3])),
                       median_context_disparity_px=float(np.median(np.linalg.norm(p0-p1, axis=1))))
    return camera1, landmark_by_keypoint, k0, d0, diagnostics


def estimate_target(frame, keypoints0, descriptors0, landmarks, intrinsic, min_inliers):
    sift = cv2.SIFT_create(nfeatures=2048)
    keys, descriptors = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
    if descriptors is None:
        raise ValueError("No target descriptors")
    matches = [m for m in mutual_ratio_matches(descriptors0, descriptors)
               if m.queryIdx in landmarks]
    if len(matches) < min_inliers:
        raise ValueError(f"Only {len(matches)} landmark-to-target matches")
    world = np.float64([landmarks[m.queryIdx] for m in matches])
    pixels = np.float64([keys[m.trainIdx].pt for m in matches])
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(world, pixels, intrinsic, None,
                                                 iterationsCount=500, reprojectionError=4.,
                                                 confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < min_inliers:
        raise ValueError("Target PnP did not obtain enough inliers")
    cv2.solvePnPRefineLM(world[inliers[:, 0]], pixels[inliers[:, 0]], intrinsic,
                         None, rvec, tvec)
    projected, _ = cv2.projectPoints(world[inliers[:, 0]], rvec, tvec, intrinsic, None)
    error = np.linalg.norm(projected.reshape(-1, 2)-pixels[inliers[:, 0]], axis=1)
    camera = np.eye(4)
    rotation = cv2.Rodrigues(rvec)[0]
    camera[:3, :3] = rotation.T
    camera[:3, 3] = (-rotation.T @ tvec).ravel()
    return camera, dict(target_features=len(keys), landmark_matches=len(matches),
                        pnp_inliers=len(inliers), median_reprojection_px=float(np.median(error)))


def main(argv=None, on_context_ready=None):
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
    cv2.setNumThreads(a.threads)
    frames, count, fps, context, targets = decode_selected(
        a.video, a.context, a.targets, a.width, a.threads)
    h, w = frames[context[0]].shape[:2]
    if any(frame.shape[:2] != (h, w) for frame in frames.values()):
        raise ValueError("Video frame dimensions changed")
    focal = a.focal_ratio * w
    intrinsic = np.array([[focal, 0, (w-1)/2], [0, focal, (h-1)/2], [0, 0, 1]], np.float64)
    camera1, landmarks, keys0, descriptors0, pair_info = estimate_pair(
        frames[context[0]], frames[context[1]], intrinsic, a.min_inliers)
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
    for i in targets:
        poses[i], targets_info[str(i)] = estimate_target(
            frames[i], keys0, descriptors0, landmarks, intrinsic, a.min_inliers)
        name = f"frame_{i:06d}.png"
        image = cv2.warpAffine(frames[i], affine, (a.size, a.size), flags=cv2.INTER_LINEAR)
        if not cv2.imwrite(str(out / "images" / name), image):
            raise RuntimeError("Could not write prepared image")
        views.append(dict(name=name, index=i, role="target", c2w=poses[i].tolist(),
                          intrinsics_normalized=k.tolist(), width=a.size, height=a.size,
                          affine_source_to_output=affine.tolist(),
                          prepared_sha256=sha(out / "images" / name)))
    save(out / "input.json", dict(schema="mvsplat_context_v1", pose_source="two_view_sift_essential_pnp",
         input_video=str(a.video.resolve()), input_video_sha256=sha(a.video),
         source_frames=count, fps=fps, context_indices=context, target_indices=targets,
         calibration=dict(kind="approximate", focal_ratio=a.focal_ratio,
                          focal_px=focal, intrinsic=intrinsic.tolist(), distortion="assumed zero"),
         geometry=dict(pair=pair_info, targets=targets_info), views=views,
          context_sha256=context_hash, near=1, far=100,
         normalization=dict(median_triangulated_depth=10., scale=pair_info["translation_scale"]),
         target_scope="Targets excluded from network context; positioned with two-view landmarks and PnP, not full SfM.",
         seconds=time.perf_counter()-started))
    print(json.dumps(dict(prepared=str(out), seconds=time.perf_counter()-started,
                          context=context, targets=targets, geometry=pair_info)))


if __name__ == "__main__":
    main()
