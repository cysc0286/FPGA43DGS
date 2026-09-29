"""Verify a relocated candidate and resource budget without opening the NPU device."""
import argparse
import ctypes
import json
from pathlib import Path
import platform
import numpy as np
from common import new_directory, save, sha
from npu.artifacts import relative_file, validate_bundle
from npu.buffers import buffer_plan
from npu.protocol import BRIDGE_ABI_VERSION, PROTOCOL_VERSION, check_bridge


def check_candidate(root, policy="shared", limit_mib=128):
    root = Path(root)
    files = json.loads((root/"files.json").read_text())
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(files)|{"files.json"}:
        raise ValueError("Candidate has missing or unlisted files")
    for name, info in files.items():
        path = relative_file(root, name)
        if path.stat().st_size != info["bytes"] or sha(path) != info["sha256"]:
            raise ValueError("Candidate hash/size mismatch: "+name)
    deployment = json.loads((root/"deployment.json").read_text())
    if (deployment["schema"] != "mvsplat_candidate_v2" or
            deployment["protocol_version"] != PROTOCOL_VERSION or
            deployment["bridge_abi_version"] != BRIDGE_ABI_VERSION):
        raise ValueError("Candidate/runtime version mismatch")
    names = deployment["partitions"]
    compiled, parts = validate_bundle(root/"compiled", names, "npu")
    graphs, _ = validate_bundle(root/"graphs", names, "onnx_reference")
    if compiled["source_manifest_sha256"] != sha(root/"graphs/manifest.json"):
        raise ValueError("Compiled/source manifest link mismatch")
    if graphs["context_sha256"] != sha(root/"input/context.npz"):
        raise ValueError("Oracle context mismatch")
    for name in names:
        item = graphs["partitions"][name]
        if compiled["partitions"][name]["export"] != item:
            raise ValueError("Compiled graph and numerical oracle metadata differ")
        oracle = relative_file(root/"graphs", name+"/oracle.npz")
        if sha(oracle) != item["oracle_sha256"]:
            raise ValueError("Oracle hash mismatch")
        with np.load(oracle, allow_pickle=False) as values:
            for key, shape in (("input", item["input_shape"]), ("expected", item["output_shape"])):
                value = values[key]
                if list(value.shape) != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError("Oracle shape/dtype/finite mismatch")
    plan = buffer_plan(parts, policy)
    if limit_mib <= 0 or plan["allocated_bytes"] > limit_mib*1024**2:
        raise ValueError("IPC buffers exceed requested budget")
    return dict(complete=True, npu_executed=False, files_verified=len(files),
                candidate_index_sha256=sha(root/"files.json"), buffer_plan=plan,
                partitions=names, estimated_ipc_bytes=plan["allocated_bytes"],
                sdk_session_memory_bytes=None, total_runtime_memory_bytes=None,
                note="Integrity and IPC capacity only; SDK sessions, device access and numerical output untested")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--buffer-policy", choices=("shared", "per_partition"), default="shared")
    p.add_argument("--buffer-limit-mib", type=int, default=128)
    p.add_argument("--require-board", action="store_true", help="Also load ARM bridge dependencies, without opening device")
    p.add_argument("--library", type=Path)
    a = p.parse_args(argv)
    out = new_directory(a.out)
    result = dict(complete=False, npu_executed=False, board_dependencies_checked=False)
    try:
        result.update(check_candidate(a.candidate, a.buffer_policy, a.buffer_limit_mib))
        result["complete"] = False
        if a.require_board:
            if platform.machine().lower() not in ("aarch64", "arm64") or a.library is None:
                raise ValueError("Board preflight requires ARM and --library")
            # Loading the bridge resolves SDK dependencies; only mgs_create opens
            # a device. Do not call it from preflight.
            check_bridge(ctypes.CDLL(str(a.library.resolve())))
            result.update(board_dependencies_checked=True, library_sha256=sha(a.library))
        result["complete"] = True
    except BaseException as exc:
        result["error"] = type(exc).__name__+": "+str(exc)
        raise
    finally:
        save(out/"preflight.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
