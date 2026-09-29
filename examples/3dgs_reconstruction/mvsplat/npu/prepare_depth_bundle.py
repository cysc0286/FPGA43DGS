"""Validate real depth-prefix ONNX exports and prepare a diagnostic ICraft bundle."""
import argparse
from pathlib import Path
import shutil

import numpy as np
import onnxruntime as ort

from common import new_directory, save, sha


NAMES = ("corr_refine_net_prefix_3", "refine_unet_prefix_3")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coverage", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--names", nargs="+", default=NAMES)
    a = p.parse_args(argv)
    import json
    coverage = json.loads((a.coverage / "result.json").read_text(encoding="utf-8"))
    out = new_directory(a.out)
    result = dict(schema="mvsplat_partitions_v1", weight_sha256=coverage["weight_sha256"],
                  input_shape=[1, 2, 3, 128, 128], npu_executed=False, partitions={})
    for name in a.names:
        module = name.rsplit("_prefix_", 1)[0]
        info = coverage["modules"][module]["prefixes"][name]
        folder = new_directory(out / name)
        source = a.coverage / (name + ".onnx")
        oracle = a.coverage / (name + "_oracle.npz")
        record = dict(complete=False)
        try:
            if not info["exported"] or sha(source) != info["graph_sha256"] or sha(oracle) != info["oracle_sha256"]:
                raise ValueError("Depth export provenance mismatch")
            with np.load(oracle, allow_pickle=False) as tensors:
                x, expected = tensors["input"], tensors["expected"]
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            session = ort.InferenceSession(str(source), sess_options=options,
                                           providers=["CPUExecutionProvider"])
            actual = session.run(None, {"features": x})[0]
            error = np.abs(actual - expected)
            passed = bool(np.isfinite(actual).all() and np.allclose(actual, expected,
                                                                   rtol=2e-3, atol=2e-4))
            record.update(complete=passed, input_shape=list(x.shape), output_shape=list(expected.shape),
                          logical_layout="NCHW", dtype="float32", graph_sha256=sha(source),
                          oracle_sha256=sha(oracle), host_onnx_passed=passed,
                          max_abs_error=float(error.max()), mean_abs_error=float(error.mean()),
                          rmse=float(np.sqrt(np.mean(np.square(error)))), npu_executed=False)
            if passed:
                shutil.copy2(source, folder / "model.onnx")
                shutil.copy2(oracle, folder / "oracle.npz")
        except Exception as exc:
            record["error"] = type(exc).__name__ + ": " + str(exc)
        result["partitions"][name] = record
        save(out / "manifest.json", result)
        print(name, record, flush=True)
    if not all(item["complete"] for item in result["partitions"].values()):
        raise RuntimeError("At least one depth prefix failed ONNX Runtime validation")


if __name__ == "__main__":
    main()
