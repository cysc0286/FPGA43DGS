"""Test actual compiled NHWC layouts with real captured MVSplat inputs on the host."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from common import new_directory, save, sha
from gaussian_generation.npu_branch.artifacts import validate_bundle, to_wire, from_wire
from gaussian_generation.npu_branch.buffers import MappedBuffers, buffer_plan
from gaussian_generation.npu_branch.benchmark import statistics


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compiled", required=True, type=Path)
    p.add_argument("--graphs", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--partitions", required=True, nargs="+")
    p.add_argument("--repeats", type=int, default=20)
    a = p.parse_args(argv)
    if a.repeats < 3:
        p.error("At least three measured repeats required")
    compiled, parts = validate_bundle(a.compiled, a.partitions, "npu")
    meta, _ = validate_bundle(a.graphs, a.partitions, "onnx_reference")
    if compiled["source_manifest_sha256"] != sha(a.graphs/"manifest.json"):
        raise ValueError("Compiled/source provenance mismatch")
    out = new_directory(a.out)
    record = dict(complete=False, npu_executed=False,
        scope="host real-input layout packing only, no model execution or board traffic",
        plan=buffer_plan(parts), partitions={})
    arena = MappedBuffers(out, parts, record["plan"], create=True)
    try:
        for name, part in parts.items():
            oracle = a.graphs/name/"oracle.npz"
            if sha(oracle) != meta["partitions"][name]["oracle_sha256"]:
                raise ValueError("Oracle hash mismatch")
            with np.load(oracle, allow_pickle=False) as data:
                value = data["input"]
            spec, dest = part["input"], arena.views[name][0]
            legacy, direct = [], []
            for i in range(a.repeats+1):
                for kind in (("legacy", "direct") if i % 2 == 0 else ("direct", "legacy")):
                    start = time.perf_counter()
                    if kind == "legacy":
                        temporary = to_wire(value, spec)
                        dest[:] = temporary
                        del temporary
                    else:
                        to_wire(value, spec, out=dest)
                    elapsed = time.perf_counter()-start
                    if i:
                        (legacy if kind == "legacy" else direct).append(elapsed)
                    # Numeric checks are outside the measured packing interval.
                    np.testing.assert_array_equal(from_wire(dest, spec), value)
            record["partitions"][name] = dict(exact_roundtrip=True, layout=spec["layout"],
                input_bytes=value.nbytes, legacy_seconds=legacy, direct_seconds=direct,
                legacy=statistics(legacy), direct=statistics(direct),
                removed_float_temporary_bytes=value.nbytes if spec["layout"] == "NHWC" else 0)
        record["complete"] = True
    finally:
        arena.close()
        save(out/"transport.json", record)
    print(json.dumps({name:{k:v[k] for k in ("legacy", "direct")} for name,v in record["partitions"].items()}, indent=2))


if __name__ == "__main__":
    main()
