"""Bounded video samples and overlapping pairs for a fixed-centre panorama.

Rotation-only capture supplies directions, not triangulated depth. This explicit
candidate never adds a fictitious camera baseline or claims a metric 3D map.
"""
import time
from pathlib import Path

import numpy as np

from common import new_directory, save, sha


def decode_ring(video, width=480, candidates=24, threads=1, calibration=None):
    frames, count, actual, fps = _decode_ring_pass(
        video, width, candidates, threads, calibration)
    if actual < candidates:
        raise ValueError('Too few timed frames for rotation sampling')
    if actual != count:
        # Some containers only expose duration * nominal FPS as FRAME_COUNT.
        # USB MJPEG capture can have irregular PTS: sample by the observed
        # presentation count, then verify that count on a second complete pass.
        # Keep the normal exact-count path single-pass and memory bounded.
        frames.clear()
        frames, count, verified, second_fps = _decode_ring_pass(
            video, width, candidates, threads, calibration, count=actual)
        if verified != actual or not np.isclose(second_fps, fps):
            raise ValueError('Decoded frame count differs between panorama sampling passes')
        fps = second_fps
    selected = set(np.linspace(0, count-1, candidates, dtype=int).tolist())
    if set(frames) != selected:
        raise ValueError('Decoded frame count differs from panorama sampling metadata')
    if len({f.shape for f in frames.values()}) != 1:
        raise ValueError('Capture dimensions changed')
    return frames, count, fps


def _decode_ring_pass(video, width, candidates, threads, calibration, *, count=None):
    import cv2

    if not 8 <= candidates <= 64 or width < 128 or threads < 1:
        raise ValueError('Invalid rotation sampling budget')
    cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG,
                          [cv2.CAP_PROP_N_THREADS, threads])
    if not cap.isOpened():
        raise ValueError('Cannot open panorama video')
    try:
        if count is None:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if count < candidates or not np.isfinite(fps) or fps <= 0:
            raise ValueError('Too few timed frames for rotation sampling')
        selected = set(np.linspace(0, count-1, candidates, dtype=int).tolist())
        frames, actual = {}, 0
        # Grab still decodes each presentation-order frame, preserving interframe
        # dependencies. Retrieve only selected frames to avoid BGR conversion and
        # copies for discarded images. No seeking or packet-order indices;
        # Observe the true presentation count through the final grab at EOF.
        while cap.grab():
            if actual in selected:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    raise ValueError('Cannot retrieve selected panorama frame')
                if calibration:
                    frame = calibration.rectify(frame, width)
                else:
                    h, w = frame.shape[:2]
                    scale = min(1., width/w)
                    frame = cv2.resize(frame, (round(w*scale), round(h*scale)),
                                       interpolation=cv2.INTER_AREA)
                frames[actual] = frame
            actual += 1
        return frames, count, actual, fps
    finally:
        cap.release()


def select_ring_pairs(poses, groups=2, *, intrinsic_normalized=None, size=128,
                      max_pairs=8, min_pairs=1, coverage_target=.95):
    """Select a fixed count or greedily cover actual cropped horizontal frusta.

    Uniform time is only a sampling proposal: actual image-estimated rotations
    determine placement. A video that fails to cover a full circle stays partial.
    Explicit 1/2/4 groups retain yaw-anchor selection. ``auto`` requires final
    cropped K and stops at the sampled coverage target or its bounded budget.
    """
    if groups == 'auto':
        return _select_coverage_pairs(poses, intrinsic_normalized, size, max_pairs,
                                      min_pairs, coverage_target)
    if min_pairs != 1:
        raise ValueError('Minimum pair count is only valid with automatic coverage selection')
    if type(groups) is not int or groups not in (1, 2, 4):
        raise ValueError('Use one, two or four panorama pairs')
    indices = sorted(poses)
    if not 3*groups <= len(indices) <= 64:
        raise ValueError('Insufficient registered rotation views')
    cameras = _fixed_centre_cameras([poses[i] for i in indices])
    forward = cameras[:, :3, 2]
    if np.any(np.linalg.norm(forward[:, [0, 2]], axis=1) < 1e-8):
        raise ValueError('Horizontal heading is undefined for a vertical camera axis')
    yaw = np.unwrap(np.arctan2(forward[:, 0], forward[:, 2]))
    direction = 1 if yaw[-1] >= yaw[0] else -1
    progress = direction*(yaw-yaw[0])
    extent = float(np.ptp(progress))
    if extent < np.radians(20):
        raise ValueError('Too little observed rotation for sparse panorama')
    frusta = (_selection_frusta(cameras, intrinsic_normalized, size)
              if intrinsic_normalized is not None else None)
    candidates = []
    for n in range(len(indices)-2):
        separation = np.degrees(np.arccos(np.clip(np.dot(forward[n], forward[n+2]), -1, 1)))
        if not 5 <= separation <= 55:
            continue
        if frusta is not None and not _cropped_pair_overlap(frusta, n, n+2, separation):
            continue
        candidates.append(dict(position=n, mask=7 << n,
                               centre=float((progress[n]+progress[n+2])/2)))
    if len(candidates) < groups:
        raise ValueError('No unused overlapping panorama pair at requested coverage')
    circular = extent >= np.radians(330)
    if circular:
        # Keep a real candidate as the phase; do not stretch a 330-degree sweep
        # to 360. Only the selection distance wraps around the measured circle.
        anchors = candidates[0]['centre']+np.arange(groups)*2*np.pi/groups
    else:
        low, high = min(c['centre'] for c in candidates), max(c['centre'] for c in candidates)
        anchors = np.linspace(low, high, groups) if groups > 1 else np.array([(low+high)/2])
    choices = []
    for anchor in anchors:
        ranked = []
        for candidate in candidates:
            delta = candidate['centre']-anchor
            error = abs(np.arctan2(np.sin(delta), np.cos(delta))) if circular else abs(delta)
            ranked.append((float(error), candidate['position'], candidate))
        choices.append(sorted(ranked, key=lambda row: row[:2]))
    # A greedy nearest choice can consume the sole valid triple for a later
    # group. At most 30 triples and four groups permit bounded exact selection.
    best, best_score = None, None
    def choose(level, used, selected, squared_error, maximum_error):
        nonlocal best, best_score
        if level == groups:
            score = (squared_error, maximum_error, tuple(c['position'] for c in selected))
            if best_score is None or score < best_score:
                best, best_score = selected[:], score
            return
        for error, _, candidate in choices[level]:
            score = squared_error+error*error
            if best_score is not None and score > best_score[0]+1e-12:
                break
            if candidate['mask'] & used:
                continue
            choose(level+1, used | candidate['mask'], selected+[candidate],
                   score, max(maximum_error, error))
    choose(0, 0, [], 0., 0.)
    if best is None:
        raise ValueError('No unused overlapping panorama pair at requested coverage')
    pairs = []
    for candidate in best:
        a, target, b = indices[candidate['position']:candidate['position']+3]
        pairs.append(dict(context=[a, b], target=target))
    return pairs, dict(observed_yaw_span_degrees=float(np.degrees(extent)),
                       pair_groups=groups, sampling='image_rotation_not_uniform_time',
                       anchor_mode='circular_yaw' if circular else 'observed_interval',
                       near_full_yaw_threshold_degrees=330.,
                       anchor_yaw_degrees=np.degrees(yaw[0]+direction*anchors).tolist(),
                       pair_center_yaw_degrees=[float(np.degrees(yaw[0]+direction*c['centre'])) for c in best],
                       full_360_geometry=False)


def _selection_frusta(cameras, intrinsic_normalized, size):
    """Real cropped-image coverage plus a conservative optical-axis limit."""
    if intrinsic_normalized is None:
        raise ValueError('Auto panorama selection requires final cropped intrinsics')
    intrinsic = np.asarray(intrinsic_normalized, np.float64)
    if intrinsic.shape == (3, 3):
        intrinsic = np.repeat(intrinsic[None], len(cameras), axis=0)
    # The same public projector validates all K matrices, including each view's
    # final principal point and both FOV dimensions, before pair selection.
    horizontal_frustum_coverage(cameras, intrinsic, size)
    masks = np.zeros((len(cameras), 360), dtype=bool)
    fovs = []
    for index, (pose, k) in enumerate(zip(cameras, intrinsic)):
        measured = horizontal_frustum_coverage([pose], k, size)
        masks[index, measured['covered_yaw_sample_degrees']] = True
        horizontal = np.arctan2(k[0, 2], k[0, 0])+np.arctan2(1-k[0, 2], k[0, 0])
        vertical = np.arctan2(k[1, 2], k[1, 1])+np.arctan2(1-k[1, 2], k[1, 1])
        fovs.append(float(np.degrees(min(horizontal, vertical))))
    return dict(masks=masks, min_fov_degrees=np.asarray(fovs), intrinsic=intrinsic)


def _cropped_pair_overlap(frusta, first, second, separation):
    limit = .85*min(frusta['min_fov_degrees'][first], frusta['min_fov_degrees'][second])
    # Axis proximity alone is insufficient when pitch, roll or a narrow crop
    # removes the shared reference-horizontal region. Require real ray overlap.
    return separation <= limit+1e-10 and bool(np.any(frusta['masks'][first] & frusta['masks'][second]))


def _select_coverage_pairs(poses, intrinsic_normalized, size, max_pairs, min_pairs,
                           coverage_target):
    """Bounded greedy coverage, not an exponential eight-group exact search."""
    if (type(max_pairs) is not int or not 1 <= max_pairs <= 8
            or type(min_pairs) is not int or not 1 <= min_pairs <= max_pairs
            or type(coverage_target) not in (int, float) or not np.isfinite(coverage_target)
            or not 0 < coverage_target <= 1):
        raise ValueError('Invalid automatic panorama coverage budget or target')
    indices = sorted(poses)
    if not 3 <= len(indices) <= 64:
        raise ValueError('Insufficient registered rotation views')
    cameras = _fixed_centre_cameras([poses[i] for i in indices])
    forward = cameras[:, :3, 2]
    if np.any(np.linalg.norm(forward[:, [0, 2]], axis=1) < 1e-8):
        raise ValueError('Horizontal heading is undefined for a vertical camera axis')
    yaw = np.unwrap(np.arctan2(forward[:, 0], forward[:, 2]))
    extent = float(np.ptp(yaw))
    circular = extent >= np.radians(330)
    if extent < np.radians(20):
        raise ValueError('Too little observed rotation for sparse panorama')
    frusta = _selection_frusta(cameras, intrinsic_normalized, size)
    candidates = []
    for n in range(len(indices)-2):
        separation = float(np.degrees(np.arccos(np.clip(np.dot(forward[n], forward[n+2]), -1, 1))))
        if not 5 <= separation <= 55 or not _cropped_pair_overlap(frusta, n, n+2, separation):
            continue
        candidates.append(dict(position=n, mask=7 << n, separation=separation,
                               centre=float((yaw[n]+yaw[n+2])/2),
                               visible=frusta['masks'][n] | frusta['masks'][n+2]))
    if not candidates:
        raise ValueError('No panorama pair with real cropped-frustum overlap; increase candidate sampling or check intrinsics')
    covered = np.zeros(360, dtype=bool)
    selected, used, gains = [], 0, []
    target_samples = max(1, int(np.ceil(360*coverage_target)))
    reason = 'pair_budget'
    while len(selected) < max_pairs:
        target_met = int(covered.sum()) >= target_samples
        if target_met and len(selected) >= min_pairs:
            reason = 'coverage_target';break
        available = [c for c in candidates if not c['mask'] & used]
        if not available:
            reason = 'no_disjoint_triple';break
        def novelty(candidate):
            distances = [abs(candidate['centre']-chosen['centre']) for chosen in selected]
            if circular:
                distances = [abs(np.arctan2(np.sin(delta), np.cos(delta))) for delta in distances]
            return min(distances) if distances else np.pi

        if target_met or (len(selected) < min_pairs and
                          not any(np.count_nonzero(c['visible'] & ~covered) for c in available)):
            candidate = max(available, key=lambda c: (novelty(c),
                int(np.count_nonzero(c['visible'] & ~covered)), -c['position']))
        else:
            candidate = max(available, key=lambda c: (int(np.count_nonzero(c['visible'] & ~covered)),
                                                    -c['position']))
        gain = int(np.count_nonzero(candidate['visible'] & ~covered))
        if gain == 0 and len(selected) >= min_pairs and not target_met:
            reason = 'no_positive_coverage_gain';break
        selected.append(candidate);used |= candidate['mask'];covered |= candidate['visible'];gains.append(gain)
    if (reason == 'pair_budget' and len(selected) >= max_pairs
            and int(covered.sum()) >= target_samples and len(selected) >= min_pairs):
        reason = 'pair_budget_after_minimum'
    if len(selected) < min_pairs:
        raise ValueError(f'Panorama selection found {len(selected)} usable disjoint pairs; '
                         f'minimum required is {min_pairs}. '
                         'Record a slower, sharper full rotation or explicitly select a smaller pair budget')
    pairs = []
    for candidate in selected:
        first, target, second = indices[candidate['position']:candidate['position']+3]
        pairs.append(dict(context=[first, second], target=target))
    contexts = [i for c in selected for i in (c['position'], c['position']+2)]
    measured = horizontal_frustum_coverage(cameras[contexts], frusta['intrinsic'][contexts], size)
    return pairs, dict(measured, observed_yaw_span_degrees=float(np.degrees(extent)),
        pair_groups=len(pairs), pair_count=len(pairs), max_pairs=max_pairs,
        min_pairs=min_pairs, minimum_pair_count_met=len(pairs) >= min_pairs,
        coverage_target=float(coverage_target), achieved_fraction=measured['sample_covered_fraction'],
        target_met=measured['sample_covered_degrees'] >= target_samples,
        uncovered_yaw_sample_degrees=np.flatnonzero(~covered).tolist(),
        selection_mode='auto', selection_method=('bounded_greedy_cropped_frustum_gain_then_farthest_yaw'
            if min_pairs > 1 else 'bounded_greedy_cropped_frustum_coverage_gain'),
        selection_stop_reason=reason, coverage_gain_degrees_per_pair=gains,
        eligible_triples=len(candidates), cropped_overlap_fov_ratio=.85,
        selected_context_axis_separation_degrees=[c['separation'] for c in selected],
        selected_context_axis_limit_degrees=[.85*float(min(frusta['min_fov_degrees'][c['position']],
            frusta['min_fov_degrees'][c['position']+2])) for c in selected],
        sampling='image_rotation_not_uniform_time', full_360_geometry=False,
        selection_guarantee='Greedy sampled angular coverage only; no optimality, visibility, spherical or image-quality guarantee')


def _fixed_centre_cameras(poses):
    cameras = np.asarray(poses, np.float64)
    if (cameras.ndim != 3 or cameras.shape[1:] != (4, 4) or not len(cameras)
            or not np.isfinite(cameras).all()
            or not np.allclose(cameras[:, 3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(cameras[:, :3, 3], 0., atol=1e-8)):
        raise ValueError('Coverage requires finite fixed-centre c2w matrices with zero translation')
    rotations = cameras[:, :3, :3]
    if (not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-5)
            or not np.allclose(np.linalg.det(rotations), 1., atol=1e-5)):
        raise ValueError('Coverage c2w rotations must be proper orthonormal matrices')
    return cameras


def horizontal_frustum_coverage(context_poses, intrinsic_normalized, size=128):
    """Sample the world y=0 direction circle against actual camera frusta.

    This is a 1-degree angular sampling estimate, not spherical coverage, scene
    geometry, visibility, Gaussian support or a rendered-image quality measure.
    Positive camera Z and both image coordinates are checked, so pitch/roll and
    principal point affect the result instead of assuming yaw +/- horizontal FOV.
    """
    cameras = _fixed_centre_cameras(context_poses)
    if type(size) is not int or size < 1:
        raise ValueError('Coverage image size must be a positive integer')
    k = np.asarray(intrinsic_normalized, np.float64)
    if k.shape == (3, 3):
        k = np.repeat(k[None], len(cameras), axis=0)
    if (k.shape != (len(cameras), 3, 3) or not np.isfinite(k).all()
            or np.any(k[:, [0, 1], [0, 1]] <= 0)
            or not np.allclose(k[:, 2], [0, 0, 1], atol=1e-10)):
        raise ValueError('Expected finite normalized pinhole intrinsics for every context')
    yaw = np.radians(np.arange(360, dtype=np.float64))
    rays = np.column_stack((np.sin(yaw), np.zeros(360), np.cos(yaw)))
    covered = np.zeros(360, dtype=bool)
    per_view = []
    for pose, intrinsic in zip(cameras, k):
        camera_rays = rays @ pose[:3, :3]
        projected = camera_rays @ intrinsic.T
        front = camera_rays[:, 2] > 1e-9
        uv = np.full((360, 2), np.inf)
        uv[front] = projected[front, :2]/projected[front, 2:]
        # Normalized coordinates cover pixel support [-.5, size-.5): all 128
        # columns AND rows, not merely the source optical-axis yaw interval.
        pixels = uv*size-.5
        visible = front & (pixels[:, 0] >= -.5) & (pixels[:, 0] < size-.5)
        visible &= (pixels[:, 1] >= -.5) & (pixels[:, 1] < size-.5)
        per_view.append(int(visible.sum()))
        covered |= visible
    count = int(covered.sum())
    if count == 0:
        gap = 360
    else:
        # Cut the circular scan on an observed direction, preserving the gap
        # that crosses 359/0 rather than counting it as two short holes.
        rotated = np.roll(covered, -int(np.flatnonzero(covered)[0]))
        gap, current = 0, 0
        for visible in rotated:
            current = 0 if visible else current+1
            gap = max(gap, current)
    return dict(coverage_scope='reference_horizontal_world_y_zero_circle_in_context_frusta_only',
                sample_step_degrees=1., sample_count=360, image_size=[size, size],
                sample_covered_degrees=count, sample_covered_fraction=count/360.,
                sample_uncovered_degrees=360-count,
                max_continuous_gap_degrees=int(gap),
                per_context_sample_covered_degrees=per_view,
                covered_yaw_sample_degrees=np.flatnonzero(covered).tolist(),
                horizontal_samples_complete=count == 360,
                spherical_coverage_measured=False,
                coverage_estimate='1-degree ray sampling; boundary angles are discretized, no visibility or quality guarantee')


def prepare_ring(video, out, *, groups=2, candidates=24, size=128,
                 width=480, threads=1, focal_ratio=.9, calibration=None,
                 max_pairs=8, min_pairs=1, coverage_target=.95, feature_contrast=.04):
    import cv2

    from pose_estimation.rotation import estimate_rotation_sequence
    from video_input.calibration import Calibration
    if size != 128 or (not calibration and
                       (focal_ratio is None or not np.isfinite(focal_ratio) or focal_ratio <= 0)):
        raise ValueError('Sparse NPU ring currently requires 128x128 and valid intrinsics')
    started = time.monotonic()
    calibration = Calibration(calibration) if calibration else None
    frames, count, fps = decode_ring(video, width, candidates, threads, calibration)
    decoded = time.monotonic()
    h, w = next(iter(frames.values())).shape[:2]
    intrinsic = (calibration.intrinsic if calibration else
                 np.array([[focal_ratio*w, 0, (w-1)/2],
                           [0, focal_ratio*w, (h-1)/2], [0, 0, 1]], np.float64))
    scale = size/min(h, w)
    affine = np.array([[scale, 0, (size-1)/2-scale*(w-1)/2],
                       [0, scale, (size-1)/2-scale*(h-1)/2]], np.float64)
    k = np.vstack((affine, [0, 0, 1])) @ intrinsic
    k[:2] /= size
    k[:2, 2] += .5/size
    if not np.allclose(k[:2, 2], .5, atol=1e-8):
        raise ValueError('Panorama requires centred rectified renderer intrinsics')
    poses, geometry = estimate_rotation_sequence(frames, intrinsic, feature_contrast=feature_contrast)
    pairs, coverage = select_ring_pairs(poses, groups, intrinsic_normalized=k, size=size,
                                         max_pairs=max_pairs, min_pairs=min_pairs,
                                         coverage_target=coverage_target)
    out = new_directory(out)
    (out/'images').mkdir()
    contexts = [i for pair in pairs for i in pair['context']]
    targets = list(dict.fromkeys(p['target'] for p in pairs))
    views, arrays = [], {}
    for i in dict.fromkeys(contexts+targets):
        image = cv2.warpAffine(frames[i], affine, (size, size), flags=cv2.INTER_LINEAR)
        name = f'frame_{i:06d}.png'
        if not cv2.imwrite(str(out/'images'/name), image):
            raise RuntimeError('Cannot save panorama sample')
        arrays[i] = image[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)/255
        views.append(dict(name=name, index=i, role='context' if i in contexts else 'target',
                          c2w=poses[i].tolist(), intrinsics_normalized=k.tolist(),
                          width=size, height=size, prepared_sha256=sha(out/'images'/name)))
    def write_context(folder, indices):
        np.savez(folder/'context.npz', image=np.stack([arrays[i] for i in indices]),
                 extrinsics=np.array([poses[i] for i in indices], np.float32),
                 intrinsics=np.repeat(k[None], len(indices), axis=0).astype(np.float32),
                 near=np.ones(len(indices), np.float32),
                 far=np.full(len(indices), 100, np.float32))
        return sha(folder/'context.npz')
    pair_records = []
    for n, pair in enumerate(pairs):
        folder = new_directory(out/f'pair_{n:02d}')
        digest = write_context(folder, pair['context'])
        pair_records.append(dict(**pair, directory=str(folder), context_sha256=digest))
    digest = write_context(out, contexts)
    headings = [np.asarray(poses[i])[:3, 2].tolist() for i in contexts]
    sampled_coverage = horizontal_frustum_coverage([poses[i] for i in contexts], k, size)
    record = dict(schema='mvsplat_rotation_ring_v1', context_sha256=digest,
        pose_source='rotation_only_image_registration', input_video_sha256=sha(video),
        source_frames=count, fps=fps, context_indices=contexts, target_indices=targets,
        views=views, pairs=pair_records, near=1, far=100,
        geometry=geometry, calibration=calibration.metadata() if calibration else
        dict(kind='approximate', focal_ratio=focal_ratio, distortion='assumed zero'),
        capture_center=[0., 0., 0.], default_view_mode='look',
        coverage=dict({**coverage, **sampled_coverage}, directions=headings,
                      crop_horizontal_fov_degrees=float(np.degrees(2*np.arctan(.5/k[0, 0])))),
        depth_observable=False, full_3d_reconstruction=False,
        quality_scope='Fixed-centre angular preview only; learned depth is unconstrained by pure rotation',
        timing=dict(decode_seconds=decoded-started, rotation_and_prepare_seconds=time.monotonic()-decoded),
        seconds=time.monotonic()-started)
    save(out/'input.json', record)
    return record
