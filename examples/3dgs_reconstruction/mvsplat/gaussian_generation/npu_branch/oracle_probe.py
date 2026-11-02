"""Low-memory hardware numerical gate before loading the full Torch model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from common import new_directory, save, sha
from gaussian_generation.npu_branch.artifacts import validate_bundle, relative_file
from gaussian_generation.npu_branch.runtime import PartitionRuntime
from gaussian_generation.npu_branch.verify import errors
from initialize.session import worker_environment


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--graphs", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--partitions", nargs="+", required=True)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--interleave", action="store_true",
                   help="Run A,B,...,A,B,... to detect retained-session interference")
    p.add_argument("--no-save-outputs", action="store_true",
                   help="Keep hashes and numerical diagnostics without large NPY payloads")
    p.add_argument("--diagnostic-continue", action="store_true",
                   help="Collect numerical failures for every partition; still exit nonzero")
    a = p.parse_args(argv)
    if a.repeats < 1:
        p.error("At least one repetition required")
    meta, _ = validate_bundle(a.graphs, a.partitions, "onnx_reference")
    compiled, _ = validate_bundle(a.bundle, a.partitions, "npu")
    if compiled["source_manifest_sha256"] != sha(a.graphs/"manifest.json"):
        raise ValueError("Compiled/oracle provenance mismatch")
    out = new_directory(a.out)
    record = dict(complete=False, npu_executed=False, partitions={},
                  interleaved=a.interleave, output_payloads_saved=not a.no_save_outputs,
                  scope="real NPU partition oracles only; not full model or image quality")
    try:
        with PartitionRuntime("npu", a.bundle, a.partitions, out/"worker", library=a.library,
                              environment=worker_environment()) as worker:
            try:
                record["readiness"] = worker.readiness
                order = ([(name, repeat) for repeat in range(a.repeats) for name in a.partitions]
                         if a.interleave else
                         [(name, repeat) for name in a.partitions for repeat in range(a.repeats)])
                first_hashes = {}
                for name, repeat in order:
                    path = relative_file(a.graphs, name+"/oracle.npz")
                    if sha(path) != meta["partitions"][name]["oracle_sha256"]:
                        raise ValueError("Oracle changed")
                    with np.load(path, allow_pickle=False) as data:
                        value, expected = data["input"], data["expected"]
                    results = record["partitions"].setdefault(name, [])
                    actual = worker.execute(name, value)
                    comparison = errors(actual, expected, 2e-3, 2e-4)
                    output_hash = hashlib.sha256(actual.tobytes()).hexdigest()
                    first_hashes.setdefault(name, output_hash)
                    comparison.update(round=repeat, output_sha256=output_hash,
                        repeat_exact=output_hash == first_hashes[name],
                        finite=bool(np.isfinite(actual).all()),
                        expected_mean=float(expected.mean()), actual_mean=float(actual.mean()),
                        call=worker.report(worker.sequence-1)["calls"][0])
                    results.append(comparison)
                    if not a.no_save_outputs:
                        np.save(out/(name+"_"+str(repeat)+".npy"), actual, allow_pickle=False)
                    save(out/"probe.json", record)
                    print(name, repeat, json.dumps(comparison), flush=True)
                    if not comparison["passed"] and not a.diagnostic_continue:
                        raise ValueError("NPU partition numerical gate failed: "+name)
                    del value, expected, actual
                record['finite_repeat_exact'] = all(c['finite'] and c['repeat_exact']
                    for values in record['partitions'].values() for c in values)
                record["complete"] = record['finite_repeat_exact'] and all(
                    c["passed"] for values in record["partitions"].values() for c in values)
                if not record["complete"]:
                    raise ValueError("One or more NPU partitions failed the unchanged numerical gate")
            finally:
                record["worker"] = worker.report()
                record["npu_executed"] = record["worker"]["npu_executed"]
    except BaseException as exc:
        record["error"] = type(exc).__name__+": "+str(exc)
        raise
    finally:
        save(out/"probe.json", record)


if __name__ == "__main__":
    main()
