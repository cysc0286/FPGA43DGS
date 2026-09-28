"""Same-board CPU versus experimental NPU/CPU file-pipeline comparison.

The measured NPU path includes packing, SDK process/session startup, file I/O,
output checks and sparse exact refinement. Numerical oracle runs are separate.
"""
import argparse
import contextlib
import io
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from backends import NumpyBackend
from block_io import pack, RecordedBackend
from matcher import match
from refine import RefinedBackend

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--runner", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tile", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--bound", type=float, default=.003)
    a = p.parse_args()
    if a.repeats < 1:
        p.error("Need at least one repeat")
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with np.load(a.case, allow_pickle=False) as d:
        left, right = d["a"], d["b"]
    expected, _ = match(left, right, NumpyBackend(), a.tile)  # CPU warmup
    np.save(out/"cpu_matches.npy", expected)
    report = {"schema": 2, "tile": a.tile, "rows": [len(left), len(right)], "matching_policy": "angular .8/.7 both directions",
              "cpu": [], "npu_cpu": [], "npu_executed": False, "raw_bound": a.bound,
              "raw_bound_validated": False, "numpy": np.__version__,
              "threads_environment": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
              "scope": "descriptor arrays to matches, separate cold NPU processes with 3 internal warmups; includes file ABI and hashes; raw-score bound requires external validation"}
    def save():
        (out/"result.json").write_text(json.dumps(report, indent=2))
    save()
    for rep in range(a.repeats):
        cpu, stat = match(left, right, NumpyBackend(), a.tile)
        if not np.array_equal(cpu, expected):
            raise ValueError("CPU reference changed across repetitions")
        report["cpu"].append(stat)
        start = time.perf_counter()
        bundle = out/("job_"+str(rep))
        with contextlib.redirect_stdout(io.StringIO()):
            pack(a.case, bundle, a.tile)
        packed = time.perf_counter()
        runtime = out/("npu_"+str(rep))
        cmd = [sys.executable, str(ROOT/"run_board.py"), "--runner", str(a.runner.resolve()),
               "--model", str(a.model.resolve()), "--raw", str(a.raw.resolve()), "--bundle", str(bundle),
               "--output", str(runtime), "--execute"]
        with (out/("launch_"+str(rep)+".log")).open("wb") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=200)
        executed = time.perf_counter()
        execution = json.loads((runtime/"result.json").read_text())
        if execution["status"] != "executed_precision_pending" or not execution["npu_executed"]:
            raise ValueError("Missing NPU execution evidence")
        replay = RecordedBackend(bundle, runtime/"runner/scores.bin", validate_scores=False)
        backend = RefinedBackend(replay, a.bound)
        try:
            candidate, refined_stat = match(left, right, backend, a.tile)
            replay.finish()
        finally:
            replay.file.close()
        end = time.perf_counter()
        exact = bool(np.array_equal(candidate, expected))
        np.save(out/("refined_"+str(rep)+"_matches.npy"), candidate)
        report["npu_cpu"].append({"rep": rep, "indices_exact": exact, "matches": len(candidate),
            "total_s": end-start, "pack_file_s": packed-start, "launch_sdk_file_s": executed-packed,
            "refine_file_s": end-executed, "sparse_candidates": backend.candidates,
            "fraction_rescored": backend.candidates/(len(left)*len(right)), "cpu_refinement": refined_stat,
            "runtime": execution, "raw_precision_gate": "pending offline check"})
        report["npu_executed"] = True
        report["cpu_driver_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["child_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        save()
        if not exact:
            raise ValueError("Refined feature indices differ from exact CPU")
        print("rep", rep, "cpu_ms", stat["total_s"]*1000, "npu_cpu_ms", (end-start)*1000, "matches", len(candidate), flush=True)


if __name__ == "__main__":
    main()
