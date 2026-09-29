"""Compile exported MVSplat partitions with ICraft, rejecting host fallback."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from common import new_directory, save, sha


def host_abi(dtype, expected):
    if dtype["element_dtype"] != "@fp(32)":
        raise ValueError("Host I/O must be FP32")
    layout = dtype["layout"]
    if layout not in ("@layout(NHWC)", "@layout(NCHW)"):
        raise ValueError("Unreviewed host layout: " + layout)
    logical = dtype["shape"]
    if layout == "@layout(NHWC)":
        logical = [logical[0], logical[3], logical[1], logical[2]]
    if logical != expected:
        raise ValueError(f"Compiled host shape {logical} != expected {expected}")
    return dict(shape=dtype["shape"], layout=layout[8:-1], logical_shape=logical)


def audit(graph, raw, meta):
    net = json.loads(graph.read_text())
    hard, other, compute = [], [], []
    def visit(op):
        if op.get("name", "").endswith(("Conv2dNode", "Conv3dNode", "MatmulNode", "ConvNode")):
            compute.append(op["name"])
        for child in op.get("sub_hard_ops", []):
            visit(child)
    for op in net["ops"]:
        if op["_type_key"] == "icraft::xir::HardOp":
            if op["compile_target"] != "@zhuget(330)":
                raise ValueError("Non-ZG330 HardOp")
            hard.append(op)
            visit(op)
        elif op["_type_key"] not in ("icraft::xir::Input", "icraft::xir::Output"):
            other.append(op["_type_key"])
    if other or not hard or not compute:
        raise ValueError(f"Missing NPU convolution/matmul or host fallback: {other}")
    inputs = [v["dtype"] for op in net["ops"] if op["_type_key"] == "icraft::xir::Input" for v in op["outputs"]]
    outputs = [v["dtype"] for op in net["ops"] if op["_type_key"] == "icraft::xir::Output" for v in op["inputs"]]
    if len(inputs) != 1 or len(outputs) != 1 or raw.stat().st_size != net["params_bytes"]:
        raise ValueError("Unary I/O or parameter length mismatch")
    return dict(passed=True, input=host_abi(inputs[0], meta["input_shape"]),
                output=host_abi(outputs[0], meta["output_shape"]), hard_ops=len(hard),
                compute=compute, host_compute_ops=0, compiler_version=net["icraft_version"],
                graph_sha256=sha(graph), raw_sha256=sha(raw), npu_executed=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graphs", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--compiler", type=Path, default=Path(__file__).resolve().parents[4] / "npu_3dgs/.vendor/icraft-3.36.1/bin")
    p.add_argument("--partitions", nargs="+")
    p.add_argument("--timeout", type=int, default=240)
    a = p.parse_args(argv)
    meta = json.loads((a.graphs / "manifest.json").read_text())
    out = new_directory(a.out.resolve())
    manifest = dict(schema="mvsplat_compiled_v1", source_manifest_sha256=sha(a.graphs / "manifest.json"),
                    weight_sha256=meta["weight_sha256"], input_shape=meta["input_shape"],
                    npu_executed=False, partitions={})
    env = dict(os.environ)
    env["PATH"] = str(a.compiler.resolve())+os.pathsep+env.get("PATH", "")
    for name in (a.partitions or meta["partitions"]):
        m = meta["partitions"][name]
        folder = new_directory(out / name)
        graph = (a.graphs / name / "model.onnx").resolve()
        record = dict(complete=False, stages=[], export=m)
        try:
            if not m["complete"] or sha(graph) != m["graph_sha256"]:
                raise ValueError("Export incomplete or graph hash mismatch")
            n, c, h, w = m["input_shape"]
            commands = [("parse", ["--net_name", name, "--network", graph, "--jr_path", folder,
                "--framework", "onnx", "--target", "zhuge", "--inputs", f"{n},{h},{w},{c}",
                "--inputs_layout", "NHWC", "--inputs_dtype", "FP32", "--pre_method", "nop",
                "--pre_scale", ",".join(["1"]*c), "--pre_mean", ",".join(["0"]*c),
                "--channel_swap", ",".join(map(str, range(c)))])]
            for stage, source in [("optimize", "parsed"), ("quantize", "optimized"),
                                  ("adapt", "quantized"), ("generate", "adapted")]:
                args = ["--json", folder / (name+"_"+source+".json"),
                        "--raw", folder / (name+"_"+source+".raw"), "--jr_path", folder]
                if stage != "generate":
                    args += ["--target", "zhuge"]
                if stage == "quantize":
                    args += ["--qdtype", "tf32", "--no_transinput", "true", "--no_imagemake", "true"]
                commands.append((stage, args))
            for stage, args in commands:
                binary = a.compiler.resolve() / ("icraft-"+stage+".exe")
                cmd = [str(binary)] + list(map(str, args))
                started = time.monotonic()
                with (folder / (stage+".log")).open("wb") as log:
                    run = subprocess.run(cmd, env=env, cwd=folder, stdout=log,
                                         stderr=subprocess.STDOUT, timeout=a.timeout)
                record["stages"].append(dict(stage=stage, exit=run.returncode,
                    seconds=time.monotonic()-started, command=cmd, compiler_sha256=sha(binary)))
                if run.returncode:
                    raise RuntimeError("ICraft " + stage + " failed")
            compiled, params = folder / (name+"_ZG.json"), folder / (name+"_ZG.raw")
            record["audit"] = audit(compiled, params, m)
            record.update(complete=True, graph=str(compiled.relative_to(out)),
                          raw=str(params.relative_to(out)), status="compiled_not_executed")
        except Exception as exc:
            record.update(status="compile_failed", error=type(exc).__name__+": "+str(exc))
        manifest["partitions"][name] = record
        save(out / "manifest.json", manifest)
        print(json.dumps(dict(partition=name, status=record["status"], error=record.get("error"))), flush=True)
    if not any(r["complete"] for r in manifest["partitions"].values()):
        raise RuntimeError("No partition passed compilation and placement audit")


if __name__ == "__main__":
    main()
