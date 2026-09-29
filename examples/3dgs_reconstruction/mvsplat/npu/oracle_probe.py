"""Low-memory hardware numerical gate before loading the full Torch model."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import new_directory, save, sha
from npu.artifacts import validate_bundle, relative_file
from npu.runtime import PartitionRuntime
from npu.verify import errors
from initialize.session import worker_environment


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--graphs", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--partitions", nargs="+", required=True)
    p.add_argument("--repeats", type=int, default=2)
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
                  scope="real NPU partition oracles only; not full model or image quality")
    try:
        with PartitionRuntime("npu", a.bundle, a.partitions, out/"worker", library=a.library,
                              environment=worker_environment()) as worker:
            try:
                record["readiness"] = worker.readiness
                for name in a.partitions:
                    path = relative_file(a.graphs, name+"/oracle.npz")
                    if sha(path) != meta["partitions"][name]["oracle_sha256"]:
                        raise ValueError("Oracle changed")
                    with np.load(path, allow_pickle=False) as data:
                        value, expected = data["input"], data["expected"]
                    results = []
                    record["partitions"][name] = results
                    for repeat in range(a.repeats):
                        actual = worker.execute(name, value)
                        comparison = errors(actual, expected, 2e-3, 2e-4)
                        results.append(comparison)
                        np.save(out/(name+"_"+str(repeat)+".npy"), actual, allow_pickle=False)
                        print(name, repeat, json.dumps(comparison), flush=True)
                        if not comparison["passed"] and not a.diagnostic_continue:
                            raise ValueError("NPU partition numerical gate failed: "+name)
                record["complete"] = all(c["passed"] for values in record["partitions"].values() for c in values)
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
