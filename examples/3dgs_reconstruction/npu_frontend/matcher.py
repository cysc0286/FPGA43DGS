"""Blockwise SIFT matching candidate, following COLMAP 4.2 brute-force semantics.

Only the descriptor dot products cross the backend boundary. Geometry, SIFT
extraction, Gaussian training and rendering are outside this module.
"""
import time
import numpy as np

DIM = 128
SCALE = np.float32(512)
SCORE_ATOL = 2e-4  # Candidate acceptance; final feature-index matches must ALSO agree.


def descriptors(value):
    a = np.asarray(value)
    if a.ndim != 2 or a.shape[1] != DIM or a.dtype != np.uint8:
        raise ValueError("Expected COLMAP uint8 SIFT descriptors [N,128]")
    return np.ascontiguousarray(a)


def blocks(a, b, tile):
    if tile not in (32, 64, 128, 256):
        raise ValueError("Supported block sizes: 32/64/128/256")
    for i in range(0, len(a), tile):
        for j in range(0, len(b), tile):
            ni, nj = min(tile, len(a)-i), min(tile, len(b)-j)
            left = np.zeros((1, 1, tile, DIM), dtype=np.float32)
            right = np.zeros((1, 1, DIM, tile), dtype=np.float32)
            left[0, 0, :ni] = a[i:i+ni].astype(np.float32)/SCALE
            right[0, 0, :, :nj] = b[j:j+nj].astype(np.float32).T/SCALE
            yield i, j, ni, nj, left, right


class TopTwo:
    def __init__(self, n):
        self.scores = np.zeros((n, 2), np.float32)
        self.ids = np.full((n, 2), -1, np.int64)

    def update(self, start, scores, candidate_start):
        n, m = scores.shape
        if m == 0 or n == 0:
            return
        row = np.arange(n)
        best = np.argmax(scores, axis=1)
        top = scores[row, best]
        copy = scores.copy()
        copy[row, best] = -1
        second = np.argmax(copy, axis=1)
        second_score = np.maximum(copy[row, second], 0)
        second_id = second+candidate_start if m > 1 else np.full(n, -1)
        merged_scores = np.column_stack((self.scores[start:start+n], top, second_score))
        merged_ids = np.column_stack((self.ids[start:start+n], best+candidate_start, second_id))
        # On equal scores retain the smaller global feature index, independent of tiling.
        sort_ids = np.where(merged_ids < 0, np.iinfo(np.int64).max, merged_ids)
        order = np.lexsort((sort_ids, -merged_scores), axis=1)[:, :2]
        self.scores[start:start+n] = np.take_along_axis(merged_scores, order, axis=1)
        self.ids[start:start+n] = np.take_along_axis(merged_ids, order, axis=1)

    def accepted(self, ratio, distance):
        angles = np.arccos(np.minimum(self.scores, np.float32(1)))
        keep = ((self.scores[:, 0] > 0) & (self.ids[:, 0] >= 0)
                & (angles[:, 0] <= np.float32(distance))
                & (angles[:, 0] < np.float32(ratio)*angles[:, 1]))
        return np.where(keep, self.ids[:, 0], -1)


def finish(forward, reverse, ratio, distance, cross_check):
    if not 0 < ratio <= 1 or not 0 < distance <= np.pi/2:
        raise ValueError("Invalid SIFT angular matching thresholds")
    f = forward.accepted(ratio, distance)
    ii = np.flatnonzero(f >= 0)
    if cross_check:
        rev = reverse.accepted(ratio, distance)
        ii = ii[rev[f[ii]] == ii]
    return np.column_stack((ii, f[ii])).astype(np.uint32).reshape(-1, 2)


def match(a, b, backend, tile=128, ratio=.8, distance=.7, cross_check=True):
    start = time.perf_counter()
    a, b = descriptors(a), descriptors(b)
    forward, reverse = TopTwo(len(a)), TopTwo(len(b))
    stats = {"backend": backend.name, "npu_executed": bool(backend.npu_executed),
             "rows_a": len(a), "rows_b": len(b), "tile": tile, "calls": 0,
             "prepare_s": 0., "backend_s": 0., "merge_s": 0.,
             "logical_input_bytes": 0, "logical_output_bytes": 0,
             "actual_dma_bytes": None, "valid_macs": len(a)*len(b)*DIM}
    previous = time.perf_counter()
    for i, j, ni, nj, left, right in blocks(a, b, tile):
        packed = time.perf_counter()
        scores = backend.dot(left, right)
        done = time.perf_counter()
        scores = np.asarray(scores)
        if scores.shape != (1, 1, tile, tile) or scores.dtype != np.float32:
            raise ValueError("Backend score shape/dtype mismatch")
        if not np.isfinite(scores).all() or np.any(scores < 0):
            raise ValueError("Non-finite or negative descriptor dot product")
        valid = scores[0, 0, :ni, :nj]
        forward.update(i, valid, j)
        reverse.update(j, valid.T, i)
        merged = time.perf_counter()
        stats["calls"] += 1
        stats["prepare_s"] += packed-previous
        stats["backend_s"] += done-packed
        stats["merge_s"] += merged-done
        stats["logical_input_bytes"] += left.nbytes+right.nbytes
        stats["logical_output_bytes"] += scores.nbytes
        previous = merged
    pairs = finish(forward, reverse, ratio, distance, cross_check)
    stats.update(total_s=time.perf_counter()-start, matched=len(pairs),
                 padded_macs=stats["calls"]*tile*tile*DIM,
                 memory_boundary="two input blocks, one output block and O(N+M) top-two state; no full score matrix",
                 time_boundary="host packing/backend call/merging; backend/session construction excluded")
    return pairs, stats


def scalar_reference(a, b, ratio=.8, distance=.7, cross_check=True):
    """Independent full-matrix int64 oracle for tests, not the low-memory runtime."""
    a, b = descriptors(a), descriptors(b)
    dots = a.astype(np.int64) @ b.astype(np.int64).T
    def one_way(scores):
        result = np.full(len(scores), -1, dtype=np.int64)
        for i, row in enumerate(scores):
            order = sorted(range(len(row)), key=lambda j: (-int(row[j]), j))
            if not order or row[order[0]] == 0:
                continue
            v1 = np.float32(row[order[0]])/np.float32(512**2)
            v2 = np.float32(row[order[1]] if len(order) > 1 else 0)/np.float32(512**2)
            d1, d2 = np.arccos(min(v1, np.float32(1))), np.arccos(min(v2, np.float32(1)))
            if d1 <= np.float32(distance) and d1 < np.float32(ratio)*d2:
                result[i] = order[0]
        return result
    f = one_way(dots)
    r = one_way(dots.T) if cross_check else None
    pairs = [(i, int(j)) for i, j in enumerate(f) if j >= 0 and (not cross_check or r[j] == i)]
    return np.asarray(pairs, dtype=np.uint32).reshape(-1, 2)
