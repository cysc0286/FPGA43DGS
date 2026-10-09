"""Verify partition oracles and full Gaussian inference using one retained worker.

ONNX mode is a host transport/numerics check, never an NPU timing result.
NPU mode must be run on ARM with a placement-validated bundle and SDK bridge.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from common import new_directory, save, sha, load_gaussians
from gaussian_generation.runtime import ModelRuntime
from gaussian_generation.npu_branch.catalog import PARTITIONS
from gaussian_generation.npu_branch.runtime import PartitionRuntime


def errors(actual, expected, rtol, atol):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        return dict(passed=False, reason="shape or finite contract")
    diff = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    return dict(passed=bool(np.allclose(actual, expected, rtol=rtol, atol=atol)),
                max_abs=float(diff.max()), mean_abs=float(diff.mean()),
                rmse=float(np.sqrt(np.mean(diff*diff))))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("onnx_reference", "npu"), required=True)
    p.add_argument("--graphs", type=Path, required=True, help="Export bundle containing real-input oracles")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--vendor", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--library", type=Path)
    p.add_argument("--worker-python", type=Path)
    p.add_argument("--buffer-policy", choices=("shared", "per_partition"), default="shared")
    p.add_argument("--buffer-limit-mib", type=int, default=128)
    p.add_argument("--partitions", nargs="+", choices=list(PARTITIONS), required=True)
    a = p.parse_args(argv)
    out = new_directory(a.out)
    record = dict(complete=False, backend=a.backend, npu_executed=False,
                  scope="partition/Gaussian checks; image quality and first frame not measured",
                  tolerances=dict(partition=dict(rtol=2e-3, atol=2e-4),
                                  gaussian=dict(rtol=5e-3, atol=1e-3)))
    try:
        runtime = ModelRuntime(a.weights, a.vendor, 2, 16, out / "runtime")
        baseline = runtime.infer(a.input, out / "cpu", sha(a.input / "context.npz"))
        meta = json.loads((a.graphs / "manifest.json").read_text())
        if meta["weight_sha256"] != sha(a.weights) or meta["context_sha256"] != baseline["input_sha256"]:
            raise ValueError("Oracle input/checkpoint differs from the verification job")
        from initialize.session import worker_environment
        with PartitionRuntime(a.backend, a.bundle, a.partitions, out / "worker", library=a.library,
                worker_python=a.worker_python, environment=worker_environment() if a.backend == "npu" else None,
                buffer_policy=a.buffer_policy, buffer_limit_mib=a.buffer_limit_mib) as worker:
            try:
                oracle_results = {}
                for name in a.partitions:
                    path = a.graphs / name / "oracle.npz"
                    if sha(path) != meta["partitions"][name]["oracle_sha256"]:
                        raise ValueError("Oracle hash mismatch")
                    with np.load(path, allow_pickle=False) as data:
                        actual = worker.execute(name, data["input"])
                        oracle_results[name] = errors(actual, data["expected"], 2e-3, 2e-4)
                    save(out / "oracles.json", oracle_results)
                    if not oracle_results[name]["passed"]:
                        raise ValueError("Partition numerical contract failed: " + name)
                runtime.attach_partitions(worker)
                hybrid = runtime.infer(a.input, out / "hybrid", baseline["input_sha256"])
                cpu_values = load_gaussians(out / "cpu/gaussians.npz")
                hybrid_values = load_gaussians(out / "hybrid/gaussians.npz")
                comparisons = {key: errors(hybrid_values[key], value, 5e-3, 1e-3)
                               for key, value in cpu_values.items()}
                record.update(oracle_results=oracle_results, gaussian_errors=comparisons,
                              cpu=baseline, hybrid=hybrid)
                if not all(v["passed"] for v in comparisons.values()):
                    raise ValueError("Full Gaussian numerical contract failed")
                # Repeat same retained model/worker; catches stale buffer/task reuse.
                repeat = runtime.infer(a.input, out / "hybrid_repeat", baseline["input_sha256"])
                record["repeat_same_hash"] = repeat["gaussian_sha256"] == hybrid["gaussian_sha256"]
                if not record["repeat_same_hash"]:
                    raise ValueError("Retained worker produced inconsistent repeat output")
                record["complete"] = True
            finally:
                record["worker"] = worker.report()
                record["npu_executed"] = record["worker"]["npu_executed"]
    except Exception as exc:
        record["error"] = type(exc).__name__+": "+str(exc)
        raise
    finally:
        save(out / "verification.json", record)
    print(json.dumps(dict(complete=record["complete"], backend=a.backend,
                         npu_executed=record["npu_executed"], gaussian_errors=record["gaussian_errors"])), flush=True)


if __name__ == "__main__":
    main()
