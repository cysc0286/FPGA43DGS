"""Bounded direction-aware pair allocation, using observed cameras only.

Coverage is sampled on the reference horizontal circle. Interior support is a
geometric proxy, not a prediction of depth accuracy or rendered image quality.
"""
import numpy as np


def interior_support(cameras, intrinsics):
    """Distance from the closest crop edge, zero outside each actual frustum."""
    angles = np.radians(np.arange(360))
    rays = np.column_stack((np.sin(angles), np.zeros(360), np.cos(angles)))
    result = np.zeros((len(cameras), 360), np.float64)
    for index, (pose, k) in enumerate(zip(cameras, intrinsics)):
        local = rays @ pose[:3, :3]
        projected = local @ k.T
        front = local[:, 2] > 1e-9
        uv = np.zeros((360, 2))
        uv[front] = projected[front, :2]/projected[front, 2:]
        edge = np.minimum(uv, 1.-uv).min(axis=1)
        result[index] = np.where(front, np.clip(2.*edge, 0., 1.), 0.)
    return result


def circular_gap(mask):
    if not mask.any():
        return 360
    missing = np.flatnonzero(~np.roll(mask, -int(np.flatnonzero(mask)[0])))
    if not len(missing):
        return 0
    return int(max(len(run) for run in np.split(missing, np.flatnonzero(np.diff(missing) > 1)+1)))


def allocate(candidates, legacy, max_pairs, min_pairs, target_samples):
    """Try eight angular seeds, then bounded one-pair exchanges.

    The previous valid selection is always a contender. Successful sets prefer
    fewer calls, then interior coverage and balanced support across 12 sectors.
    Failed sets maximize real sampled coverage. No global optimum is claimed.
    """
    def state(items):
        visible = np.zeros(360, bool)
        support = np.zeros(360)
        used = 0
        for item in items:
            visible |= item['visible']
            support = np.maximum(support, item['support'])
            used |= item['mask']
        return visible, support, used

    def quality(visible, support):
        central = support >= .2
        sectors = support.reshape(12, 30).mean(axis=1)
        return (int(visible.sum()), int(central.sum()), float(sectors.min()), float(support.sum()))

    reference_visible, reference_support, _ = state(legacy)
    reference_quality = quality(reference_visible, reference_support)
    reference_gap = circular_gap(reference_support >= .2)
    reference_complete = reference_visible.sum() >= target_samples and len(legacy) >= min_pairs

    def score(items):
        visible, support, _ = state(items)
        reached = visible.sum() >= target_samples and len(items) >= min_pairs
        q = quality(visible, support)
        # Fewer calls must not come from weakening the original interior cover
        # or its least-supported sector. These are geometry checks, not PSNR.
        retains_support = (not reference_complete or
                           q[1] >= reference_quality[1] and q[2]+1e-12 >= reference_quality[2]
                           and circular_gap(support >= .2) <= reference_gap)
        return (int(reached), int(retains_support), -len(items) if reached else q[0],
                *q, -len(items), tuple(-c['position'] for c in items))

    possibilities = [legacy] if len(legacy) >= min_pairs else []
    origin = candidates[0]['centre']
    seeds = []
    for angle in origin+np.arange(8)*2*np.pi/8:
        seed = min(candidates, key=lambda c: (
            abs(np.arctan2(np.sin(c['centre']-angle), np.cos(c['centre']-angle))), c['position']))
        if seed['position'] not in [c['position'] for c in seeds]:
            seeds.append(seed)
    # Empty start also gives the coverage-greedy path with interior tie breaks.
    for seed in [None, *seeds]:
        chosen = [] if seed is None else [seed]
        visible, support, used = state(chosen)
        while len(chosen) < max_pairs:
            if visible.sum() >= target_samples and len(chosen) >= min_pairs:
                break
            available = [c for c in candidates if not c['mask'] & used]
            if not available:
                break
            candidate = max(available, key=lambda c: (
                *quality(visible | c['visible'], np.maximum(support, c['support'])),
                -c['position']))
            if not np.any(candidate['visible'] & ~visible) and len(chosen) >= min_pairs:
                break
            chosen.append(candidate)
            visible |= candidate['visible']
            support = np.maximum(support, candidate['support'])
            used |= candidate['mask']
        if len(chosen) >= min_pairs:
            possibilities.append(chosen)
    if not possibilities:
        raise ValueError(f'Panorama selection cannot satisfy the minimum required is {min_pairs}; '
                         'record a slower, sharper rotation or reduce the explicit minimum')
    best = max(possibilities, key=score)
    # Improve spatial balance without adding another NPU forward call.
    for _ in range(2):
        previous = score(best)
        improved = best
        improved_score = previous
        for index in range(len(best)):
            others = best[:index]+best[index+1:]
            _, _, used = state(others)
            for candidate in candidates:
                if candidate['mask'] & used:
                    continue
                proposal = others+[candidate]
                value = score(proposal)
                if value > improved_score:
                    improved, improved_score = proposal, value
        best = improved
        if improved_score <= previous:
            break
    # A deterministic gain-first order makes incremental coverage auditable.
    ordered, remaining, covered = [], best[:], np.zeros(360, bool)
    while remaining:
        item = max(remaining, key=lambda c: (int(np.count_nonzero(c['visible'] & ~covered)),
                                            -c['position']))
        remaining = [c for c in remaining if c['position'] != item['position']]
        ordered.append(item)
        covered |= item['visible']
    visible, support, _ = state(ordered)
    return ordered, dict(
        balance_search='eight_yaw_seeds_and_two_single_pair_exchange_passes',
        central_support_threshold=.2,
        central_sample_covered_fraction=float(np.mean(support >= .2)),
        central_max_continuous_gap_degrees=circular_gap(support >= .2),
        sector_width_degrees=30,
        sector_support_mean=support.reshape(12, 30).mean(axis=1).tolist(),
        support_scope='geometric_crop_edge_margin_only_not_rendered_quality')
