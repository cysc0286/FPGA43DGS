"""ICraft per-layer reference for the saved first-convolution NPU diagnostic.

Host execution is an oracle only, never a deployed inference benchmark.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from common import sha
from gaussian_generation.npu_branch.precision_probe import compare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--actual", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--runner", type=Path, default=Path(__file__).resolve().parents[5]/
                   "npu_3dgs/.vendor/icraft-3.36.1/bin/icraft-run.exe")
    a = p.parse_args()
    network = json.loads(a.graph.read_text())
    if sum(op["_type_key"] == "icraft::xir::Conv2d" for op in network["ops"]) != 1:
        raise ValueError("This probe requires exactly one reference convolution")
    a.out.mkdir(parents=True, exist_ok=False)
    with np.load(a.oracle, allow_pickle=False) as data:
        x, expected = data["input"], data["expected"]
    actual = np.load(a.actual, allow_pickle=False)
    if actual.shape != expected.shape:
        raise ValueError("NPU output shape mismatch")
    wire = x.transpose(0, 2, 3, 1).copy()
    wire.tofile(a.out/"input.ftmp")
    command = [str(a.runner.resolve()), "--json", str(a.graph.resolve()),
        "--raw", str(a.raw.resolve()), "--input", str((a.out/"input.ftmp").resolve()),
        "--backends", "Host", "--dump_format", "SFB", "--log_path", str((a.out/"logs").resolve())]
    with (a.out/"run.log").open("wb") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=60, check=True)
    result = dict(scope="ICraft host layer oracle; NPU output was collected on board",
                  command=command, layers={}, hashes={str(path):sha(path)
                    for path in (a.graph, a.raw, a.oracle, a.actual)})
    dumps = a.out/"logs"/network["name"]/"ftmpSFB"
    for op in network["ops"]:
        for value in op.get("outputs", []):
            dtype = value["dtype"]
            if dtype["layout"] != "@layout(NHWC)":
                raise ValueError("Unexpected reference layout")
            dump = np.fromfile(dumps/(str(value["v_id"])+".ftmp"), dtype=np.float32)
            dump = dump.reshape(dtype["shape"]).transpose(0, 3, 1, 2)
            if op["_type_key"] == "icraft::xir::Input" and not np.array_equal(dump, x):
                raise ValueError("ICraft reference input differs from the fixed input")
            if dump.shape == expected.shape:
                result["layers"][str(value["v_id"])] = dict(
                    host_vs_oracle=compare(dump, expected), npu_vs_host=compare(actual, dump))
    if not result["layers"]:
        raise ValueError("No convolution output dump collected")
    (a.out/"comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result["layers"], indent=2))


if __name__ == "__main__":
    main()
