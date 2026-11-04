"""Image-estimated fixed-centre rotations; pure rotation cannot observe depth.

Frames must already be rectified for the supplied pixel intrinsic matrix.
This is an explicit approximation, not a translation/SfM or free-view pose path.
"""
from collections import OrderedDict
import heapq
import time

import cv2
import numpy as np

from pose_estimation.geometry import image_features, mutual_ratio_matches
from pose_estimation.rotation_graph import refine_observed_rotations


def _degrees(rotation):
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1))))


def _nearest_rotation(matrix):
    u, _, vt = np.linalg.svd(matrix)
    return u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt


def _relative(first, second, intrinsic, min_inliers, max_reprojection):
    keys0, desc0 = first
    keys1, desc1 = second
    if desc0 is None or desc1 is None or len(keys0) < min_inliers or len(keys1) < min_inliers:
        raise ValueError('Insufficient texture for rotation registration')
    matches = mutual_ratio_matches(desc0, desc1)
    if len(matches) < min_inliers:
        raise ValueError(f'Only {len(matches)} mutual rotation matches')
    p0 = np.array([keys0[m.queryIdx].pt for m in matches], np.float64)
    p1 = np.array([keys1[m.trainIdx].pt for m in matches], np.float64)
    if not np.isfinite(p0).all() or not np.isfinite(p1).all():
        raise ValueError('Nonfinite feature coordinates')
    homography, mask = cv2.findHomography(p0, p1, cv2.RANSAC, max_reprojection,
                                         maxIters=1500, confidence=.999)
    if homography is None or mask is None or not np.isfinite(homography).all():
        raise ValueError('Rotation homography estimation failed')
    chosen = mask.ravel() != 0
    if int(chosen.sum()) < min_inliers:
        raise ValueError('Too few homography inliers')
    # H-RANSAC selects actual correspondences, but its eight-parameter matrix
    # can be ill-conditioned for locally clustered points. Projecting K^-1 H K
    # onto SO(3) can then reject all good points before bearing refit is reached.
    # Fit the three-parameter rotation directly on its unit-bearing consensus.
    kinv = np.linalg.inv(intrinsic)
    determinant = np.linalg.det(homography)
    if not np.isfinite(determinant) or abs(determinant) < 1e-12:
        raise ValueError('Degenerate rotation homography')
    rays0 = np.column_stack((p0, np.ones(len(p0)))) @ kinv.T
    rays1 = np.column_stack((p1, np.ones(len(p1)))) @ kinv.T
    unit0 = rays0/np.linalg.norm(rays0, axis=1, keepdims=True)
    unit1 = rays1/np.linalg.norm(rays1, axis=1, keepdims=True)
    rotation = _nearest_rotation(unit1[chosen].T @ unit0[chosen])

    def errors(candidate):
        rays = rays0 @ candidate.T
        projected = rays @ intrinsic.T
        valid = (rays[:, 2] > 1e-9) & np.isfinite(projected).all(axis=1)
        result = np.full(len(rays), np.inf)
        result[valid] = np.linalg.norm(projected[valid, :2]/projected[valid, 2:]-p1[valid], axis=1)
        return result

    residual = errors(rotation)
    chosen &= residual <= max_reprojection
    if int(chosen.sum()) < min_inliers:
        raise ValueError('Homography is inconsistent with a pure rotation')
    # Refit on unit bearings to avoid retaining projective/shear components.
    rotation = _nearest_rotation(unit1[chosen].T @ unit0[chosen])
    residual = errors(rotation)
    chosen &= residual <= max_reprojection
    inliers = int(chosen.sum())
    if inliers < min_inliers:
        raise ValueError('Rotation bearing fit failed pixel reprojection validation')
    return rotation, dict(matches=len(matches), inliers=inliers,
                          initialization='homography_inlier_unit_bearings',
                          median_reprojection_px=float(np.median(residual[chosen])),
                          max_reprojection_px=float(np.max(residual[chosen])),
                          degrees=_degrees(rotation))


def estimate_rotation_sequence(frames, intrinsic, min_inliers=20, max_reprojection=3.,
                               *, feature_contrast=.04):
    """Return image-estimated c2w rotations and diagnostics, with every t = 0.

    At most 64 chronological candidates are accepted. Each adjacent or one-gap
    edge is image-validated independently. Only the largest connected component
    is returned, relative to its earliest frame, so a bad opening frame does not
    discard a later valid sweep. Disconnected components are never merged. No
    uniform turn rate or invented baseline is introduced. Near a completed
    sweep, multiple observed end-to-start edges can correct accumulated drift;
    an assumed 360-degree endpoint is never supplied to the estimator.
    """
    started = time.perf_counter()
    if not isinstance(frames, dict) or not 2 <= len(frames) <= 64:
        raise ValueError('Rotation sequence requires 2 to 64 candidate frames')
    if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0 for i in frames):
        raise ValueError('Frame indices must be nonnegative integers')
    if isinstance(min_inliers, bool) or not isinstance(min_inliers, (int, np.integer)) or min_inliers < 4:
        raise ValueError('At least four rotation inliers are required')
    if not np.isfinite(max_reprojection) or max_reprojection <= 0:
        raise ValueError('Reprojection threshold must be positive and finite')
    if not np.isfinite(feature_contrast) or not 0 < feature_contrast <= .04:
        raise ValueError('SIFT feature contrast must be finite and in (0, .04]')
    k = np.asarray(intrinsic, np.float64)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1], atol=1e-10) or abs(np.linalg.det(k)) < 1e-12):
        raise ValueError('Expected a finite calibrated pixel intrinsic matrix')
    indices = sorted(frames)
    dimensions = None
    for image in frames.values():
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
                or image.shape[2] != 3 or min(image.shape[:2]) < 16):
            raise ValueError('Expected uint8 BGR images with at least 16 pixels per side')
        if dimensions is not None and image.shape != dimensions:
            raise ValueError('Rotation frame dimensions changed')
        dimensions = image.shape
    cache = OrderedDict()
    peak_cache = 0
    detector = (cv2.SIFT_create(nfeatures=2048, contrastThreshold=feature_contrast)
                if feature_contrast != .04 else None)

    def features(index):
        nonlocal peak_cache
        if index not in cache:
            # Evict before inserting, including the optional closure check.
            while len(cache) >= 4:
                cache.popitem(last=False)
            cache[index] = (image_features(frames[index]) if detector is None else
                            detector.detectAndCompute(cv2.cvtColor(frames[index], cv2.COLOR_BGR2GRAY), None))
            peak_cache = max(peak_cache, len(cache))
        cache.move_to_end(index)
        return cache[index]

    graph = {index: [] for index in indices}
    edges, rejected, relative_edges = [], [], []
    for position, current in enumerate(indices[1:], 1):
        for previous in reversed(indices[max(0, position-2):position]):
            try:
                relative, details = _relative(features(previous), features(current), k,
                                              min_inliers, max_reprojection)
            except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
                rejected.append(dict(source=int(previous), target=int(current), reason=str(error)))
                continue
            edges.append(dict(source=int(previous), target=int(current), **details))
            relative_edges.append((previous, current, relative))
            graph[previous].append((current, relative))
            graph[current].append((previous, relative.T))
    if not relative_edges:
        raise ValueError('No valid rotation edge: insufficient texture or inconsistent image motion')
    components, remaining = [], set(indices)
    while remaining:
        todo, component = [min(remaining)], set()
        while todo:
            node = todo.pop()
            if node in component:
                continue
            component.add(node)
            todo.extend(neighbour for neighbour, _ in graph[node] if neighbour not in component)
        remaining -= component
        components.append(sorted(component))
    components.sort(key=lambda component: (-len(component), component[0]))
    selected = set(components[0])
    anchor = min(selected)
    poses = {anchor: np.eye(4, dtype=np.float64)}
    # Earliest reachable frame first, preferring its nearest temporal neighbour.
    # A fully valid adjacent chain retains the old frame-by-frame composition.
    positions = {index: n for n, index in enumerate(indices)}
    pending, tree_edges = [], []
    def enqueue(source):
        for target, relative in graph[source]:
            if target not in poses:
                heapq.heappush(pending, (target, abs(positions[target]-positions[source]),
                                        source, relative))
    enqueue(anchor)
    while pending:
        target, _, source, relative = heapq.heappop(pending)
        if target in poses:
            continue
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = _nearest_rotation(poses[source][:3, :3] @ relative.T)
        poses[target] = pose
        tree_edges.append(dict(source=int(source), target=int(target)))
        enqueue(target)
    registered = sorted(poses)
    forward = np.array([poses[i][:3, 2] for i in registered])
    yaw = np.degrees(np.unwrap(np.arctan2(forward[:, 0], forward[:, 2])))
    span = float(np.ptp(yaw))
    closure, closure_edge = None, None
    loop_edges, loop_details = [], []
    refinement = dict(applied=False, reason='sweep_too_short', loop_count=0)
    if span >= 180 and len(registered) >= 8:
        for first in registered[:2]:
            for last in registered[-2:]:
                try:
                    relative, details = _relative(features(first), features(last), k,
                                                  max(40, min_inliers), max_reprojection)
                    weight = min(200., details['inliers'] / (1. + details['median_reprojection_px']**2))
                    loop_edges.append((first, last, relative, weight))
                    loop_details.append(dict(source=int(first), target=int(last), **details))
                except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
                    rejected.append(dict(source=int(first), target=int(last),
                                         closure=True, reason=str(error)))
        local_edges = [(i, j, r, min(200., d['inliers'] / (1. + d['median_reprojection_px']**2)))
                       for (i, j, r), d in zip(relative_edges, edges) if i in poses and j in poses]
        poses, refinement = refine_observed_rotations(poses, local_edges, loop_edges)
        forward = np.array([poses[i][:3, 2] for i in registered])
        yaw = np.degrees(np.unwrap(np.arctan2(forward[:, 0], forward[:, 2])))
        span = float(np.ptp(yaw))
    if span >= 270:
        last = registered[-1]
        try:
            relative, details = _relative(features(anchor), features(last), k,
                                          min_inliers, max_reprojection)
            closure = _degrees(relative @ poses[last][:3, :3])
            closure_edge = dict(source=int(anchor), target=int(last), **details)
        except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
            rejected.append(dict(source=int(anchor), target=int(last),
                                 closure=True, reason=str(error)))
    diagnostics = dict(geometry='pure_rotation_approx', translation='zero_unobserved',
        feature_contrast=float(feature_contrast), minimum_rotation_inliers=int(min_inliers),
        maximum_reprojection_px=float(max_reprojection),
        depth_observable=False, free_translation_supported=False,
        frame_indices=[int(i) for i in indices], registered_frame_indices=[int(i) for i in registered],
        missing_frame_indices=[int(i) for i in indices if i not in poses],
        excluded_frame_indices=[int(i) for i in indices if i not in poses],
        anchor_index=int(anchor), component_sizes=[len(component) for component in components],
        components=[[int(i) for i in component] for component in components],
        component_selection='largest_image_validated_connected_component_then_earliest_anchor',
        coordinate_reference='selected_anchor_camera_with_exact_zero_translation',
        per_edge=[edge for edge in edges if edge['source'] in selected],
        validated_edge_count=len(relative_edges), selected_tree_edges=tree_edges,
        rejected_edges=rejected,
        yaw_degrees_by_index={str(i): float(angle) for i, angle in zip(registered, yaw)},
        observed_yaw_span_degrees=span, closure_error_degrees=closure, closure_edge=closure_edge,
        closure_reference_degrees=3., closure_adjusted=refinement['applied'],
        loop_refinement=refinement, loop_edges=loop_details,
        feature_cache_peak_frames=peak_cache,
        seconds=time.perf_counter()-started,
        limitations='Image-validated fixed-centre rotation approximation; arbitrary depth is unobservable. '
                    'No invented baseline, forced 360-degree coverage or free camera motion.')
    return poses, diagnostics
