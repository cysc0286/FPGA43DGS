"""Check generated ICraft placement and freeze the actual host tensor ABI."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(model, params, tile=128):
    net = json.loads(model.read_text())
    ops = net["ops"]
    hard = [o for o in ops if o["_type_key"] == "icraft::xir::HardOp"]
    compute = [o for o in ops if o["_type_key"] not in ("icraft::xir::Input", "icraft::xir::Output")]
    matmul = [o for o in hard if o["name"] == "icraft::xir::MatmulNode"]
    inputs = [v["dtype"] for o in ops if o["_type_key"] == "icraft::xir::Input" for v in o["outputs"]]
    outputs = [v["dtype"] for o in ops if o["_type_key"] == "icraft::xir::Output" for v in o["inputs"]]
    if len(matmul) != 1 or len(hard) != len(compute) or any(o["compile_target"] != "@zhuget(330)" for o in hard):
        raise ValueError("Expected one dynamic Matmul mapped to ZG330, without host compute fallback")
    if [v["shape"] for v in inputs] != [[1, tile, 128, 1], [1, 128, tile, 1]]:
        raise ValueError("Unexpected compiled input shapes")
    if [v["shape"] for v in outputs] != [[1, 1, tile, tile]]:
        raise ValueError("Unexpected output shape")
    if any(v["element_dtype"] != "@fp(32)" for v in inputs+outputs):
        raise ValueError("Expected FP32 host I/O")
    if any(v["layout"] != "@layout(NHWC)" for v in inputs) or outputs[0]["layout"] != "@layout(***C)":
        raise ValueError("Unexpected compiled host layout; review ABI before use")
    if params.stat().st_size != net["params_bytes"]:
        raise ValueError("Parameter file length mismatch")
    return {"passed": True, "compiler_version": net["icraft_version"], "tile": tile,
            "graph_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "raw_sha256": hashlib.sha256(params.read_bytes()).hexdigest(),
            "params_bytes": net["params_bytes"], "hard_ops": len(hard),
            "matmul_op_id": matmul[0]["op_id"], "host_compute_ops": 0,
            "inputs": inputs, "outputs": outputs, "npu_executed": False,
            "scope": "compiler placement only; runtime binding, output layout and numerical precision need board verification",
            "operations": [{k: o[k] for k in ("op_id", "name", "compile_target")} for o in hard]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tile", type=int, default=128)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Audit output exists")
    result = audit(a.model, a.raw, a.tile)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("inputs", "outputs", "operations")}, indent=2))


if __name__ == "__main__":
    main()
