"""Isolated FP32/NCHWc16 diagnostic. Does not widen the production host ABI."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import platform
import shlex
import sys

import numpy as np


def audit(graph, raw, oracle):
    data = json.loads(graph.read_text())
    inputs = [v["dtype"] for op in data["ops"] if op["_type_key"] == "icraft::xir::Input" for v in op["outputs"]]
    outputs = [v["dtype"] for op in data["ops"] if op["_type_key"] == "icraft::xir::Output" for v in op["inputs"]]
    if len(inputs) != 1 or len(outputs) != 1 or raw.stat().st_size != data["params_bytes"]:
        raise ValueError("Unary graph and parameter size required")
    if inputs[0]["element_dtype"] != "@fp(32)" or inputs[0]["layout"] != "@layout(NHWC)":
        raise ValueError("Expected FP32 NHWC input")
    dtype = outputs[0]["element_dtype"]
    if (not isinstance(dtype, dict) or dtype["storage_dtype"] != "@fp(32)" or
            dtype["expressed_dtype"] != "@fp(32)" or dtype["zero_points"] or
            dtype["scale"] != "@scales([axis=-1])" or outputs[0]["layout"] != "@layout(NCHWc16)"):
        raise ValueError("Only unscaled FP32 NCHWc16 is audited here")
    with np.load(oracle, allow_pickle=False) as payload:
        x, y = payload["input"], payload["expected"]
    n, c, h, w = y.shape
    if c % 16 or outputs[0]["shape"] != [n, c//16, h, w, 16]:
        raise ValueError("This diagnostic does not support channel padding")
    if inputs[0]["shape"] != [x.shape[0], x.shape[2], x.shape[3], x.shape[1]]:
        raise ValueError("Input shape mismatch")
    hard = [op for op in data["ops"] if op["_type_key"] == "icraft::xir::HardOp"]
    if not hard or any(op["compile_target"] != "@zhuget(330)" for op in hard):
        raise ValueError("NPU graph required")
    if any(op["_type_key"] not in ("icraft::xir::Input", "icraft::xir::Output", "icraft::xir::HardOp") for op in data["ops"]):
        raise ValueError("Unexpected host operation")
    return {"input": inputs[0], "output": outputs[0], "sha256": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (graph, raw, oracle)}}


def diagnostic_bridge(readout):
    source = Path(__file__).with_name("bridge.cpp").read_text()
    edits = {
        "#include <memory>": "#include <memory>\n#include <iostream>",
        "!type->element_dtype.isFP32()": "(i==0 && !type->element_dtype.isFP32())",
        "session.apply();": '''session.apply();
        try {
            bool ok=session->backends[0].cast<zg330::ZG330Backend>().precheck();
            std::cerr << "DIAGNOSTIC_PRECHECK=" << ok << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "DIAGNOSTIC_PRECHECK_ERROR=" << e.what() << std::endl;
        }''',
    }
    if readout == "raw":
        edits['''std::ostringstream stream(std::ios::out|std::ios::binary);
        values[0].dump(stream,"SFB");const auto bytes=stream.str();
        if(bytes.size()!=c.out_bytes) throw std::runtime_error("SFB output length mismatch");
        std::memcpy(out,bytes.data(),bytes.size());'''] = 'values[0].read(reinterpret_cast<char*>(out),0,c.out_bytes);'
    for before, after in edits.items():
        if source.count(before) != 1:
            raise ValueError("Bridge changed; re-review diagnostic patch")
        source = source.replace(before, after)
    return source


def board_run(root, readout):
    if platform.machine().lower() not in ("aarch64", "arm64"):
        raise RuntimeError("Board execution required")
    import fcntl
    from npu.verify import errors
    info = audit(root/"graph.json", root/"params.raw", root/"oracle.npz")
    lock = open("/run/lock/fpga43dgs-npu.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with np.load(root/"oracle.npz", allow_pickle=False) as payload:
        x = np.ascontiguousarray(payload["input"].transpose(0, 2, 3, 1))
        expected = payload["expected"]
    y = np.empty(info["output"]["shape"] if readout == "raw" else expected.shape, dtype=np.float32)
    lib = ct.CDLL(str(root/"bridge.so"))
    lib.mgs_error.restype = ct.c_char_p
    lib.mgs_create.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_size_t, ct.c_size_t]
    lib.mgs_create.restype = ct.c_void_p
    lib.mgs_destroy.argtypes = [ct.c_void_p]
    lib.mgs_forward_detailed.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_double)]
    lib.mgs_forward_detailed.restype = ct.c_int
    context = lib.mgs_create(str(root/"graph.json").encode(), str(root/"params.raw").encode(), x.nbytes, y.nbytes)
    if not context:
        raise RuntimeError(lib.mgs_error().decode())
    result = dict(scope="isolated blocked FP32 output; not production", readout=readout, audit=info, samples=[])
    try:
        for index in range(2):
            timing = (ct.c_double*5)()
            if lib.mgs_forward_detailed(context, x.ctypes.data, x.nbytes, y.ctypes.data, y.nbytes, timing):
                raise RuntimeError(lib.mgs_error().decode())
            actual = y.transpose(0, 1, 4, 2, 3).reshape(expected.shape).copy() if readout == "raw" else y.copy()
            np.save(root/f"output_{index}.npy", actual)
            sample = dict(**errors(actual, expected, 2e-3, 2e-4), sdk_seconds=list(timing))
            result["samples"].append(sample)
            print(json.dumps(sample), flush=True)
        result["passed"] = all(item["passed"] for item in result["samples"])
    finally:
        lib.mgs_destroy(context)
        lock.close()
        (root/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    if not result["passed"]:
        raise RuntimeError("Isolated FP32 numerical gate failed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--board-run", type=Path)
    p.add_argument("--graph", type=Path)
    p.add_argument("--raw", type=Path)
    p.add_argument("--oracle", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--board-root")
    p.add_argument("--readout", choices=("raw", "sfb"), default="sfb")
    a = p.parse_args()
    if a.board_run:
        board_run(a.board_run, a.readout)
        return
    if not all((a.graph, a.raw, a.oracle, a.out, a.board_root)):
        p.error("Host mode requires graph, raw, oracle, out and board-root")
    audit(a.graph, a.raw, a.oracle)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/"bridge.cpp").write_text(diagnostic_bridge(a.readout), encoding="utf-8")
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]/"3dgs_compositor/board"))
    import remote
    client = remote.connect()
    root = shlex.quote(a.board_root)
    try:
        remote.run(client, f"test ! -e {root} && mkdir -p {root}")
        sftp = client.open_sftp()
        for source, name in ((a.graph,"graph.json"), (a.raw,"params.raw"), (a.oracle,"oracle.npz"),
                             (a.out/"bridge.cpp","bridge.cpp"), (Path(__file__),"probe.py")):
            sftp.put(str(source), a.board_root+"/"+name)
        sdk = "/root/heterogs_npu/sdk_3.36.1/usr"
        command = (f"cd {root} && timeout 180s g++ -O3 -std=gnu++17 -shared -fPIC bridge.cpp -o bridge.so "
                   f"-I{sdk}/include -L{sdk}/lib/aarch64-linux-gnu -Wl,-rpath,{sdk}/lib/aarch64-linux-gnu "
                   "-licraft_zg330backend -licraft_hostbackend -licraft_xrt -licraft_xir -licraft_utils -ldw -ldl -pthread")
        remote.run(client, command, timeout=200, log=a.out/"build.log")
        command = ("timeout 90s env PYTHONPATH=/root/fpga43dgs_reconstruction/npu_board_v2_20260929/mvsplat "
                   f"LD_LIBRARY_PATH=/root/fpga43dgs_reconstruction/arm_env/lib:{sdk}/lib/aarch64-linux-gnu "
                   "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /root/fpga43dgs_reconstruction/arm_env/bin/python3 "
                   f"{root}/probe.py --board-run {root} --readout {a.readout}")
        status, log = remote.run(client, command, timeout=100, check=False, log=a.out/"board.log")
        print(log)
        (a.out/"exit.txt").write_text(str(status)+"\n")
        for name in ("result.json", "output_0.npy", "output_1.npy"):
            try:
                sftp.get(a.board_root+"/"+name, str(a.out/name))
            except OSError:
                pass
        if status:
            raise SystemExit(status)
    finally:
        client.close()


if __name__ == "__main__":
    main()
