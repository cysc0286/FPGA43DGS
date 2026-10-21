"""DIAGNOSTIC ONLY: dump NPU Gaussians even when numerical gates fail.

This is a copy of verify.py with the two hard gates removed, so that
`hybrid/gaussians.npz` is produced for image-level evaluation. It is NOT an
acceptance artifact: `verification_diagnostic.json` records `diagnostic_only`.

Everything else - CPU baseline, per-partition oracle comparison, the retained
worker, the buffer plan, the input SHA binding - is identical to verify.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import new_directory, save, sha, load_gaussians
from initialize.model_runtime import ModelRuntime
from npu.catalog import PARTITIONS
from npu.runtime import PartitionRuntime


def errors(actual, expected, rtol, atol):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        return dict(passed=False, reason="shape or finite contract")
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return dict(passed=bool(np.allclose(actual, expected, rtol=rtol, atol=atol)),
                max_abs=float(diff.max()), mean_abs=float(diff.mean()),
                rmse=float(np.sqrt(np.mean(diff * diff))))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("onnx_reference", "npu"), required=True)
    p.add_argument("--graphs", type=Path, required=True)
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
    p.add_argument("--threads", type=int, default=2,
                   help="CPU 前向线程数；verify.py 写死为 2，仓库默认入口用 4")
    a = p.parse_args(argv)

    out = new_directory(a.out)
    record = dict(complete=False, diagnostic_only=True, backend=a.backend,
                  npu_executed=False, dumped=False,
                  scope="DIAGNOSTIC: gates not enforced; Gaussians dumped for image evaluation only",
                  tolerances=dict(partition=dict(rtol=2e-3, atol=2e-4),
                                  gaussian=dict(rtol=5e-3, atol=1e-3)),
                  threads=a.threads)
    try:
        runtime = ModelRuntime(a.weights, a.vendor, a.threads, 16, out / "runtime")
        baseline = runtime.infer(a.input, out / "cpu", sha(a.input / "context.npz"))
        meta = json.loads((a.graphs / "manifest.json").read_text())
        if (meta["weight_sha256"] != sha(a.weights)
                or meta["context_sha256"] != baseline["input_sha256"]):
            raise ValueError("Oracle input/checkpoint differs from the verification job")
        from initialize.session import worker_environment
        with PartitionRuntime(a.backend, a.bundle, a.partitions, out / "worker",
                              library=a.library, worker_python=a.worker_python,
                              environment=worker_environment() if a.backend == "npu" else None,
                              buffer_policy=a.buffer_policy,
                              buffer_limit_mib=a.buffer_limit_mib) as worker:
            try:
                oracle_results, hard_errors = {}, {}
                for name in a.partitions:
                    path = a.graphs / name / "oracle.npz"
                    if sha(path) != meta["partitions"][name]["oracle_sha256"]:
                        hard_errors[name] = "oracle hash mismatch"
                        continue
                    with np.load(path, allow_pickle=False) as data:
                        try:
                            actual = worker.execute(name, data["input"])
                            oracle_results[name] = errors(actual, data["expected"], 2e-3, 2e-4)
                        except Exception as exc:
                            hard_errors[name] = type(exc).__name__ + ": " + str(exc)
                    save(out / "oracles.json", dict(results=oracle_results, hard_errors=hard_errors))
                    print("oracle", name, json.dumps(oracle_results.get(name, hard_errors.get(name))),
                          flush=True)
                runtime.attach_partitions(worker)
                hybrid = runtime.infer(a.input, out / "hybrid", baseline["input_sha256"])
                cpu_values = load_gaussians(out / "cpu/gaussians.npz")
                hybrid_values = load_gaussians(out / "hybrid/gaussians.npz")
                comparisons = {key: errors(hybrid_values[key], value, 5e-3, 1e-3)
                               for key, value in cpu_values.items()}
                record.update(oracle_results=oracle_results, hard_errors=hard_errors,
                              gaussian_errors=comparisons, cpu=baseline, hybrid=hybrid,
                              dumped=True)
                print("gaussians", json.dumps(comparisons), flush=True)
            finally:
                record["worker"] = worker.report()
                record["npu_executed"] = record["worker"]["npu_executed"]
    except Exception as exc:
        record["error"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        save(out / "verification_diagnostic.json", record)
    print(json.dumps(dict(diagnostic_only=True, dumped=record.get("dumped", False),
                          gaussian_errors=record.get("gaussian_errors"))), flush=True)


if __name__ == "__main__":
    main()
