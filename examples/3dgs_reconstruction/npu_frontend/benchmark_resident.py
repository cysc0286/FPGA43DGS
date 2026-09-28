"""Retained SDK session, complete in-memory matching, independent raw-score audit."""
import argparse
import json
import os
import platform
import resource
from pathlib import Path
import numpy as np
from matcher import blocks, match
from backends import NumpyBackend
from native_backend import NativeNPUBackend
from refine import RefinedBackend


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--bound", type=float, default=.003)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    npu = NativeNPUBackend(a.library, a.model, a.raw, a.tile)
    report = {"mode": "persistent_session", "tile": a.tile, "npu_executed": False,
              "init_s": npu.init_s, "library_sha256": npu.library_sha256, "placement_audit": npu.audit,
              "platform": platform.platform(), "numpy": np.__version__,
              "thread_settings": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
              "raw_score_error_budget": a.bound, "cases": [], "passed": False,
              "boundary": "descriptor arrays to matches; packing/refinement/merge included, shared setup and validation separate; empirical error budget only"}
    def save():
        (a.output/"result.json").write_text(json.dumps(report, indent=2))
    save()
    try:
        for source in sorted(a.cases.glob("*.npz")):
            with np.load(source, allow_pickle=False) as data:
                left, right = data["a"], data["b"]
            expected, _ = match(left, right, NumpyBackend(), a.tile)
            # Validate every raw score on these actual inputs before using the
            # empirical error bound in sparse candidate refinement.
            maximum = 0.
            for i, j, ni, nj, aa, bb in blocks(left, right, a.tile):
                raw = npu.dot(aa, bb)
                reference = np.zeros((a.tile, a.tile), np.float32)
                reference[:ni, :nj] = (left[i:i+ni].astype(np.int64)@right[j:j+nj].astype(np.int64).T).astype(np.float32)/np.float32(512**2)
                if not np.isfinite(raw).all():
                    raise ValueError("Non-finite NPU score")
                maximum = max(maximum, float(np.abs(raw[0, 0]-reference).max()))
            case = {"name": source.stem, "rows": [len(left), len(right)], "raw_max_abs_error": maximum,
                    "raw_v1_score_gate_passed": maximum <= .0002, "raw_budget_passed": maximum <= a.bound,
                    "cpu": [], "npu_refined": []}
            report["cases"].append(case)
            save()
            if maximum > a.bound:
                raise ValueError("Raw NPU score exceeds refinement budget; retain CPU path")
            for rep in range(a.repeats):
                _, cpu_stats = match(left, right, NumpyBackend(), a.tile)
                refined = RefinedBackend(npu, a.bound)
                before = npu.total_times.copy()
                actual, npu_stats = match(left, right, refined, a.tile)
                exact = bool(np.array_equal(actual, expected))
                npu_stats.update(indices_exact=exact, raw_budget_validated_for_these_inputs=True,
                                 exact_cpu_dots=refined.candidates, cpu_sparse_dot_s=refined.sparse_dot_s,
                                 sdk_write_forward_read_s=(npu.total_times-before).tolist())
                case["cpu"].append(cpu_stats)
                case["npu_refined"].append(npu_stats)
                np.save(a.output/(source.stem+"_matches.npy"), actual)
                report["npu_executed"] = npu.calls > 0
                save()
                if not exact:
                    raise ValueError("Refined matching differs from exact CPU")
                print(source.stem, rep, "cpu_ms", cpu_stats["total_s"]*1000, "npu_refined_ms", npu_stats["total_s"]*1000,
                      "raw_error", maximum, "matches", len(actual), flush=True)
        report["passed"] = True
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save()
    finally:
        npu.close()


if __name__ == "__main__":
    main()
