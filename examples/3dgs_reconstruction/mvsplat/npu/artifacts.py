"""Validate deployment assets and translate logical NCHW to audited host layouts."""
import json
from pathlib import Path
import numpy as np
from common import FULL_WEIGHT_SHA256, sha
from npu.catalog import PARTITIONS


def relative_file(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError("Artifact outside bundle or missing: " + str(name))
    return path


def validate_bundle(root, names, backend):
    root = Path(root).resolve()
    data = json.loads((root / "manifest.json").read_text())
    schema = "mvsplat_compiled_v1" if backend == "npu" else "mvsplat_partitions_v1"
    if data["schema"] != schema or data["weight_sha256"] != FULL_WEIGHT_SHA256:
        raise ValueError("Bundle schema/checkpoint mismatch")
    if not names or len(set(names)) != len(names) or any(name not in PARTITIONS for name in names):
        raise ValueError("Select unique, known partitions explicitly")
    result = {}
    for name in names:
        item = data["partitions"][name]
        if not item["complete"]:
            raise ValueError("Incomplete partition: " + name)
        if backend == "npu":
            from npu.compile_graphs import audit
            graph, raw = relative_file(root, item["graph"]), relative_file(root, item["raw"])
            actual = audit(graph, raw, item["export"])
            if actual != item["audit"]:
                raise ValueError("Compiled placement/ABI/hash changed")
            meta = item["export"]
            result[name] = dict(meta=meta, input=actual["input"], output=actual["output"],
                                graph=str(graph), raw=str(raw), audit=actual)
        else:
            graph = relative_file(root, name + "/model.onnx")
            if sha(graph) != item["graph_sha256"] or not item["host_onnx_passed"]:
                raise ValueError("Unvalidated ONNX graph")
            result[name] = dict(meta=item, graph=str(graph),
                input=dict(layout="NCHW", shape=item["input_shape"]),
                output=dict(layout="NCHW", shape=item["output_shape"]))
    return data, result


def to_wire(array, spec):
    if spec["layout"] == "NHWC":
        array = array.transpose(0, 2, 3, 1)
    elif spec["layout"] != "NCHW":
        raise ValueError("Unsupported layout")
    if list(array.shape) != spec["shape"] or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("Input dtype/shape/finite contract failed")
    return np.ascontiguousarray(array)


def from_wire(array, spec):
    if list(array.shape) != spec["shape"] or not np.isfinite(array).all():
        raise ValueError("Output shape/finite contract failed")
    if spec["layout"] == "NHWC":
        array = array.transpose(0, 3, 1, 2)
    elif spec["layout"] != "NCHW":
        raise ValueError("Unsupported output layout")
    # Copy before the reusable output buffer is assigned to the next job.
    return np.array(array, dtype=np.float32, order="C", copy=True)
