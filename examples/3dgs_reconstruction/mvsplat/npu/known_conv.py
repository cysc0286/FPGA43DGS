"""Small deterministic SDK diagnostic; never a MVSplat performance substitute.

Export on the compiler host, then run the compiled graph on the ARM board.
The exact binary-fraction case separates transport/weights from rounding.
"""
import argparse
import ctypes as ct
import json
from pathlib import Path
import platform
import time

import numpy as np
from common import new_directory, save, sha


def export(out):
    import onnx
    import torch
    out = new_directory(out)
    manifest = dict(schema="known_conv_diagnostic_v1", weight_sha256="known_not_mvsplat",
                    input_shape=[1, 16, 4, 4], partitions={})
    for name in ("identity", "dyadic", "fractional"):
        folder = new_directory(out/name)
        x = (np.arange(16*32*32, dtype=np.float32).reshape(1, 16, 32, 32) % 31 - 15)/16
        weight = np.eye(16, dtype=np.float32) if name == "identity" else (
            np.arange(256, dtype=np.float32).reshape(16, 16) % 9-4)/32
        bias = np.zeros(16, np.float32)
        if name == "fractional":
            x = x/1.37
            weight = weight/1.19
        expected = np.einsum("oc,nchw->nohw", weight, x, optimize=False)
        layer = torch.nn.Conv2d(16, 16, 1, bias=True).eval()
        with torch.no_grad():
            layer.weight.copy_(torch.from_numpy(weight[:, :, None, None]))
            layer.bias.copy_(torch.from_numpy(bias))
        torch.onnx.export(layer, (torch.from_numpy(x),), str(folder/"model.onnx"),
                          input_names=["features"], output_names=["result"],
                          opset_version=17, dynamo=False)
        onnx.checker.check_model(str(folder/"model.onnx"))
        np.savez(folder/"oracle.npz", input=x, expected=expected, weight=weight)
        manifest["partitions"][name] = dict(complete=True, input_shape=list(x.shape),
            output_shape=list(expected.shape), graph_sha256=sha(folder/"model.onnx"),
            oracle_sha256=sha(folder/"oracle.npz"), exact_required=name != "fractional")
    save(out/"manifest.json", manifest)


def run(graphs, bundle, library, out, repeats):
    if platform.machine().lower() not in ("aarch64", "arm64"):
        raise RuntimeError("Diagnostic inference requires the actual ARM NPU")
    import fcntl
    from npu.artifacts import to_wire, from_wire
    from npu.verify import errors
    out = new_directory(out)
    metadata = json.loads((bundle/"manifest.json").read_text())
    if metadata["source_manifest_sha256"] != sha(graphs/"manifest.json"):
        raise ValueError("Diagnostic provenance mismatch")
    lock = open("/run/lock/fpga43dgs-npu.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lib = ct.CDLL(str(library.resolve()))
    lib.mgs_error.restype = ct.c_char_p
    lib.mgs_create.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_size_t, ct.c_size_t]
    lib.mgs_create.restype = ct.c_void_p
    lib.mgs_destroy.argtypes = [ct.c_void_p]
    lib.mgs_bindings.argtypes = [ct.c_void_p]
    lib.mgs_forward.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_void_p,
                               ct.c_size_t, ct.POINTER(ct.c_double)]
    lib.mgs_forward.restype = ct.c_int
    try:
        detailed = lib.mgs_forward_detailed
    except AttributeError:
        detailed = None
    if detailed is not None:
        detailed.argtypes = lib.mgs_forward.argtypes
        detailed.restype = ct.c_int
    result = dict(scope="known-input single-session NPU correctness; not model speed", cases={}, passed=False)
    try:
        for name, item in metadata["partitions"].items():
            if not item["complete"]:
                raise ValueError("Compile failed: "+name)
            oracle = np.load(graphs/name/"oracle.npz", allow_pickle=False)
            x = to_wire(oracle["input"], item["audit"]["input"])
            y = np.empty(item["audit"]["output"]["shape"], np.float32)
            ctx = lib.mgs_create(str(bundle/item["graph"]).encode(),
                                 str(bundle/item["raw"]).encode(), x.nbytes, y.nbytes)
            if not ctx:
                raise RuntimeError(lib.mgs_error().decode())
            samples = result["cases"][name] = []
            try:
                if lib.mgs_bindings(ctx) < 1:
                    raise ValueError("No real NPU computation")
                for i in range(repeats):
                    t = time.perf_counter()
                    width = 5 if detailed is not None else 3
                    times = (ct.c_double*width)()
                    forward = detailed or lib.mgs_forward
                    if forward(ctx, x.ctypes.data, x.nbytes, y.ctypes.data, y.nbytes, times):
                        raise RuntimeError(lib.mgs_error().decode())
                    seconds = time.perf_counter()-t
                    actual = from_wire(y, item["audit"]["output"])
                    comparison = errors(actual, oracle["expected"], 2e-3, 2e-4)
                    exact = np.array_equal(actual, oracle["expected"])
                    if item["export"].get("exact_required", False):
                        comparison["passed"] = bool(exact)
                    samples.append(dict(**comparison, exact=bool(exact), seconds=seconds,
                                        sdk_stage_names=(
                                            ["input_write", "forward_submit", "wait", "output_convert", "sdk_total"]
                                            if detailed is not None else
                                            ["input_write", "execute_wait", "output_convert"]),
                                        sdk_seconds=list(times), npu_executed=True))
                    np.save(out/(name+f"_{i}.npy"), actual)
                    print(name, i, json.dumps(samples[-1]), flush=True)
            finally:
                lib.mgs_destroy(ctx)
        result["passed"] = all(v["passed"] for values in result["cases"].values() for v in values)
    finally:
        save(out/"result.json", result)
        lock.close()
    if not result["passed"]:
        raise RuntimeError("Known convolution gate failed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("export", "run"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--graphs", type=Path)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--library", type=Path)
    p.add_argument("--repeats", type=int, default=2)
    a = p.parse_args()
    if a.action == "export":
        export(a.out)
    else:
        if not all((a.graphs, a.bundle, a.library)) or a.repeats < 1:
            p.error("run needs graphs, bundle, library and positive repeats")
        run(a.graphs, a.bundle, a.library, a.out, a.repeats)


if __name__ == "__main__":
    main()
