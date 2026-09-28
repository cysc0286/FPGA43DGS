"""Experimental NPU coarse scores followed by exact sparse CPU dot products.

Top-two preservation is conditional on every raw score error being <= bound.
The bound must be measured on the evaluated inputs; it is not a vendor guarantee.
Full raw-score validation and full CPU matching remain separate acceptance gates.
"""
import time
import numpy as np
from matcher import DIM, SCALE


class RefinedBackend:
    name = "coarse_scores_plus_cpu_exact_candidates"

    def __init__(self, backend, bound=.003):
        if not np.isfinite(bound) or bound < 0:
            raise ValueError("Invalid raw-score error budget")
        self.backend, self.bound = backend, float(bound)
        self.npu_executed = backend.npu_executed
        self.candidates, self.sparse_dot_s = 0, 0.

    def dot(self, left, right):
        raw = self.backend.dot(left, right)
        if not np.isfinite(raw).all() or np.any(raw < 0):
            raise ValueError("Invalid coarse scores")
        s = raw[0, 0]
        # Includes both approximate top-two sets and every competitor that can
        # displace them under an absolute error budget. The factor two covers
        # opposite-sign errors in the candidate and the second-best score.
        row_second = np.partition(s, -2, axis=1)[:, -2]
        col_second = np.partition(s, -2, axis=0)[-2, :]
        keep = (s >= row_second[:, None]-2*self.bound) | (s >= col_second[None, :]-2*self.bound)
        # Padded zero descriptors cannot affect a strictly positive match.
        valid_a = np.any(left[0, 0] != 0, axis=1)
        valid_b = np.any(right[0, 0] != 0, axis=0)
        keep &= valid_a[:, None] & valid_b[None, :]
        rr, cc = np.nonzero(keep)
        self.candidates += len(rr)
        exact = np.zeros_like(s)
        start = time.perf_counter()
        aa = np.rint(left[0, 0]*SCALE).astype(np.int32)
        bb = np.rint(right[0, 0].T*SCALE).astype(np.int32)
        # Bound gather memory even when identical descriptors make all candidates
        # ambiguous; never materialize N*M*128 temporary values.
        for k in range(0, len(rr), 256):
            rows, cols = rr[k:k+256], cc[k:k+256]
            dots = np.sum(aa[rows]*bb[cols], axis=1, dtype=np.int64)
            exact[rows, cols] = dots.astype(np.float32)/np.float32(512**2)
        self.sparse_dot_s += time.perf_counter()-start
        return exact.reshape(raw.shape)
