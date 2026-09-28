"""Hash-bound, explicit local ARM execution. This script never opens SSH/serial."""
import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from audit_graph import audit


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, q):
    values = sorted(values)
    at = (len(values)-1)*q
    lo = int(at)
    return values[lo]+(values[min(lo+1, len(values)-1)]-values[lo])*(at-lo)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runner", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true", help="Run locally on ARM64 Linux; default writes a plan")
    p.add_argument("--timeout", type=float, default=180)
    a = p.parse_args()
    if a.timeout <= 0:
        p.error("--timeout must be positive")
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((a.bundle/"manifest.json").read_text())
    for filename, key in [("jobs.bin", "jobs_sha256"), ("descriptors.npz", "descriptors_sha256")]:
        if sha(a.bundle/filename) != manifest[key]:
            raise ValueError("Input bundle hash mismatch")
    mapping = audit(a.model, a.raw, manifest["tile"])
    out = a.output/"runner"
    cmd = [str(a.runner.resolve()), str(a.model.resolve()), str(a.raw.resolve()), str((a.bundle/"jobs.bin").resolve()), str(out)]
    files = {"model": a.model, "raw": a.raw, "jobs": a.bundle/"jobs.bin"}
    record = {"status": "plan_only", "command": cmd, "placement_audit": mapping,
              "input_hashes": {k: sha(v) for k, v in files.items()}, "npu_executed": False,
              "precision_verified": False, "actual_dma_bytes": None, "power_w": None}
    def save():
        (a.output/"result.json").write_text(json.dumps(record, indent=2))
    save()
    if not a.execute:
        print(a.output/"result.json")
        return
    try:
        if platform.system() != "Linux" or platform.machine().lower() not in ("aarch64", "arm64"):
            raise RuntimeError("Execute this runner only on the intended ARM64 Linux board")
        if not a.runner.is_file():
            raise ValueError("Build the NPU runner on ARM first")
        record.update(status="running", runner_sha256=sha(a.runner))
        save()
        start = time.perf_counter()
        with (a.output/"run.log").open("wb") as log:
            run = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout)
        record.update(process_wall_s=time.perf_counter()-start, exit_code=run.returncode)
        if run.returncode:
            raise RuntimeError("Runner failed; partial output is not accepted")
        status = json.loads((out/"status.json").read_text())
        record["runner_status"] = status
        if not status["completed"] or not status["npu_executed"] or status["matmul_ops"] != 1:
            raise ValueError("Missing NPU execution evidence")
        if status["blocks"] != manifest["blocks"]:
            raise ValueError("Runner block count differs from input")
        if any(sha(v) != record["input_hashes"][k] for k, v in files.items()) or sha(a.runner) != record["runner_sha256"]:
            raise ValueError("Runner or input changed during execution")
        with (out/"timing.csv").open() as f:
            rows = list(csv.DictReader(f))
        if len(rows) != manifest["blocks"]:
            raise ValueError("Incomplete block timing records")
        record["per_block_us"] = {}
        for column in rows[0]:
            if column != "sample":
                values = [float(r[column]) for r in rows]
                record["per_block_us"][column] = {"mean": statistics.mean(values), "median": statistics.median(values),
                        "p95": percentile(values, .95), "p99": percentile(values, .99)}
        record.update(status="executed_precision_pending", npu_executed=True,
                      output_hashes={f.name: sha(f) for f in out.iterdir() if f.is_file()})
    except Exception as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        save()
    print(a.output/"result.json")


if __name__ == "__main__":
    main()
