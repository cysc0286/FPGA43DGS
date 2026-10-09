"""Paired real-input partition benchmarks, including every IPC/SDK/layout cost.

Use gaussian_generation.npu_branch.verify afterwards for full-model correctness. Module times cannot be
added up to claim video-to-frame acceleration. ONNX mode measures host CPUs only.
"""
import argparse
import json
from pathlib import Path
import platform
import time
import numpy as np
from common import new_directory, save, sha
from gaussian_generation.runtime import ModelRuntime
from gaussian_generation.npu_branch.catalog import PARTITIONS, locate
from gaussian_generation.npu_branch.runtime import PartitionRuntime
from gaussian_generation.npu_branch.verify import errors


def statistics(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(samples=len(a), mean_seconds=float(a.mean()), median_seconds=float(np.median(a)),
                min_seconds=float(a.min()), max_seconds=float(a.max()),
                p95_seconds=float(np.percentile(a, 95)), p99_seconds=float(np.percentile(a, 99)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("onnx_reference", "npu"), required=True)
    for arg in ("graphs", "bundle", "weights", "vendor", "out"):
        p.add_argument("--"+arg, type=Path, required=True)
    p.add_argument("--library", type=Path)
    p.add_argument("--worker-python", type=Path)
    p.add_argument("--partitions", nargs="+", choices=list(PARTITIONS), required=True)
    p.add_argument("--buffer-policy", choices=("shared", "per_partition"), default="shared")
    p.add_argument("--buffer-limit-mib", type=int, default=128)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args(argv)
    if a.repeats < 3 or a.warmup < 1:
        p.error("At least 3 measured repetitions and 1 warmup required")
    out = new_directory(a.out)
    record = dict(complete=False, backend=a.backend, npu_executed=False,
        machine=platform.machine(), threads=a.threads, warmup_per_partition=a.warmup,
        scope="paired repeated input, individual partitions; excludes startup, whole model and rendering",
        tail_note="empirical quantiles; few repeated samples do not establish production tail latency",
        partitions={}, scene_preparation_seconds=None)
    try:
        model = ModelRuntime(a.weights, a.vendor, a.threads, 16, out/"runtime")
        from gaussian_generation.npu_branch.artifacts import validate_bundle
        meta, _ = validate_bundle(a.graphs, a.partitions, "onnx_reference")
        if meta["weight_sha256"] != sha(a.weights):
            raise ValueError("Oracle/checkpoint mismatch")
        record.update(weight_sha256=sha(a.weights), context_sha256=meta["context_sha256"])
        from initialize.session import worker_environment
        with PartitionRuntime(a.backend, a.bundle, a.partitions, out/"worker", library=a.library,
                worker_python=a.worker_python, buffer_policy=a.buffer_policy, buffer_limit_mib=a.buffer_limit_mib,
                environment=worker_environment() if a.backend == "npu" else None) as worker:
            try:
                for name in a.partitions:
                    path = a.graphs/name/"oracle.npz"
                    if sha(path) != meta["partitions"][name]["oracle_sha256"]:
                        raise ValueError("Oracle changed: "+name)
                    with np.load(path, allow_pickle=False) as data:
                        value, expected = data["input"], data["expected"]
                    module = locate(model.model, name)
                    results = dict(cpu=[], candidate=[], comparisons=[], passed=False)
                    record["partitions"][name] = results
                    first_call = len(worker.calls)
                    with model.torch.inference_mode():
                        for i in range(a.warmup+a.repeats):
                            outputs, durations = {}, {}
                            # Alternate order to limit systematic cache/order bias.
                            for kind in (("cpu", "candidate") if i % 2 == 0 else ("candidate", "cpu")):
                                start = time.perf_counter()
                                if kind == "cpu":
                                    output = module(model.torch.from_numpy(value))
                                    if isinstance(output, (list, tuple)):
                                        output = output[0]
                                    outputs[kind] = output.detach().numpy()
                                else:
                                    outputs[kind] = worker.execute(name, value)
                                durations[kind] = time.perf_counter()-start
                            comparison = {kind: errors(result, expected, 2e-3, 2e-4)
                                          for kind, result in outputs.items()}
                            results["comparisons"].append(comparison)
                            if not all(v["passed"] for v in comparison.values()):
                                raise ValueError("Benchmark correctness failed: "+name)
                            if i >= a.warmup:
                                for kind in durations:
                                    results[kind].append(durations[kind])
                    results.update(passed=True, cpu_summary=statistics(results["cpu"]),
                        candidate_summary=statistics(results["candidate"]),
                        measured_worker_calls=worker.calls[first_call+a.warmup:])
                    results["cpu_over_candidate_median"] = (results["cpu_summary"]["median_seconds"] /
                                                              results["candidate_summary"]["median_seconds"])
                    results["npu_partition_gain_measured"] = (a.backend == "npu" and
                                                                results["cpu_over_candidate_median"] > 1)
                    save(out/"benchmark.json", record)
                    print(name, json.dumps({k:results[k] for k in ("cpu_summary", "candidate_summary")} ), flush=True)
                record["complete"] = True
            finally:
                record["worker"] = worker.report()
                record["npu_executed"] = record["worker"]["npu_executed"]
    except BaseException as exc:
        record["error"] = type(exc).__name__+": "+str(exc)
        raise
    finally:
        save(out/"benchmark.json", record)


if __name__ == "__main__":
    main()
