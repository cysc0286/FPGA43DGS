"""Two-view geometry and held-out target PnP, in one shared scene frame."""
import time
import cv2
import numpy as np


def mutual_ratio_matches(a, b, ratio=0.75):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(a, b, k=2)
    backward = matcher.knnMatch(b, a, k=2)
    reverse = {m.queryIdx: m.trainIdx for pair in backward
               if len(pair) == 2 for m, other in [pair] if m.distance < ratio * other.distance}
    return [m for pair in forward if len(pair) == 2 for m, other in [pair]
            if m.distance < ratio * other.distance and reverse.get(m.trainIdx) == m.queryIdx]


def estimate_pair(first, second, intrinsic, min_inliers, timing=None):
    started = time.perf_counter()
    sift = cv2.SIFT_create(nfeatures=2048)
    k0, d0 = sift.detectAndCompute(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), None)
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(second, cv2.COLOR_BGR2GRAY), None)
    if d0 is None or d1 is None:
        raise ValueError("No SIFT descriptors in context frames")
    if timing is not None:
        timing["context_sift_seconds"] = time.perf_counter()-started
    matching_started = time.perf_counter()
    matches = mutual_ratio_matches(d0, d1)
    if timing is not None:
        timing["context_matching_seconds"] = time.perf_counter()-matching_started
    if len(matches) < min_inliers:
        raise ValueError(f"Only {len(matches)} mutual matches")
    p0 = np.float64([k0[m.queryIdx].pt for m in matches])
    p1 = np.float64([k1[m.trainIdx].pt for m in matches])
    pose_started = time.perf_counter()
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
    if timing is not None:
        timing["context_pose_seconds"] = time.perf_counter()-pose_started
        timing["context_total_seconds"] = time.perf_counter()-started
    return camera1, landmark_by_keypoint, k0, d0, diagnostics


def estimate_target(frame, keypoints0, descriptors0, landmarks, intrinsic, min_inliers,
                    timing=None, label="target"):
    started = time.perf_counter()
    sift = cv2.SIFT_create(nfeatures=2048)
    keys, descriptors = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
    if descriptors is None:
        raise ValueError("No target descriptors")
    if timing is not None:
        timing[f"{label}_sift_seconds"] = time.perf_counter()-started
    matching_started = time.perf_counter()
    matches = [m for m in mutual_ratio_matches(descriptors0, descriptors)
               if m.queryIdx in landmarks]
    if timing is not None:
        timing[f"{label}_matching_seconds"] = time.perf_counter()-matching_started
    if len(matches) < min_inliers:
        raise ValueError(f"Only {len(matches)} landmark-to-target matches")
    world = np.float64([landmarks[m.queryIdx] for m in matches])
    pixels = np.float64([keys[m.trainIdx].pt for m in matches])
    pnp_started = time.perf_counter()
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(world, pixels, intrinsic, None,
                                                 iterationsCount=500, reprojectionError=4.,
                                                 confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < min_inliers:
        raise ValueError("Target PnP did not obtain enough inliers")
    cv2.solvePnPRefineLM(world[inliers[:, 0]], pixels[inliers[:, 0]], intrinsic,
                         None, rvec, tvec)
    projected, _ = cv2.projectPoints(world[inliers[:, 0]], rvec, tvec, intrinsic, None)
    error = np.linalg.norm(projected.reshape(-1, 2)-pixels[inliers[:, 0]], axis=1)
    if timing is not None:
        timing[f"{label}_pnp_seconds"] = time.perf_counter()-pnp_started
        timing[f"{label}_total_seconds"] = time.perf_counter()-started
    camera = np.eye(4)
    rotation = cv2.Rodrigues(rvec)[0]
    camera[:3, :3] = rotation.T
    camera[:3, 3] = (-rotation.T @ tvec).ravel()
    return camera, dict(target_features=len(keys), landmark_matches=len(matches),
                        pnp_inliers=len(inliers), median_reprojection_px=float(np.median(error)))
