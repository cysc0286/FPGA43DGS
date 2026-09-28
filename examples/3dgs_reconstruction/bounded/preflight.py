"""Read-only local environment inspection; no SSH, installs, reboot or device writes.

Default inventory uses the standard library and works on the board's old Python.
--probe explicitly runs individual imports under the resource watchdog.
"""
import argparse
import importlib.util
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from common import ROOT, cpu_env, save, sha256

PROBES = {
    "numpy": 'import numpy as m; print(m.__version__)',
    "cv2": 'import cv2 as m,json; print(json.dumps({"version":m.__version__,"decoder_thread_api":hasattr(m,"CAP_PROP_N_THREADS")}))',
    "plyfile": 'import plyfile; print("PLY reader available")',
    "torch": 'import torch,json; print(json.dumps({"version":torch.__version__,"cuda_build":torch.version.cuda,"hip_build":getattr(torch.version,"hip",None),"cmake_prefix":torch.utils.cmake_prefix_path}))',
    "pycolmap": 'import pycolmap as m,json; print(json.dumps({"version":m.__version__,"has_cuda":m.has_cuda,"cpu_api":all(hasattr(m,x) for x in ["FeatureExtractionOptions","FeatureMatchingOptions","SequentialPairingOptions","IncrementalPipelineOptions","undistort_images"])}))',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path, help="Fresh evidence directory")
    p.add_argument("--probe", action="store_true", help="Import runtime dependencies under a watchdog")
    p.add_argument("--opensplat", type=Path)
    p.add_argument("--rss-mib", type=int, default=560)
    p.add_argument("--reserve-mib", type=int, default=128)
    a = p.parse_args()
    if a.rss_mib <= 0 or a.reserve_mib <= 0:
        p.error("Use positive memory limits")
    a.output.mkdir(parents=True, exist_ok=False)
    modules = {n: importlib.util.find_spec(n) is not None for n in list(PROBES)+["psutil"]}
    search_path = str(Path(sys.executable).parent)+os.pathsep+os.environ.get("PATH", "")
    tools = {n: shutil.which(n, path=search_path) for n in ("cmake", "ninja", "g++", "pkg-config", "ffmpeg")}
    report = {"schema": 1, "platform": platform.platform(), "machine": platform.machine(),
              "python": sys.version, "python_executable": sys.executable, "libc": platform.libc_ver(),
              "python_supported_for_candidate_dependencies": sys.version_info >= (3, 10),
              "modules_discoverable": modules, "tools": tools, "imports": {},
              "no_remote_access": True, "board_fit_verified": False,
              "full_chain_validated": False, "ready_for_attempt": False,
              "scope": "environment only; passing does not prove reconstruction quality, memory fit or speed"}
    if Path("/proc/meminfo").exists():
        report["meminfo"] = Path("/proc/meminfo").read_text()
    if modules["psutil"]:
        import psutil
        report["memory"] = {"total_mib": psutil.virtual_memory().total/2**20,
                            "available_mib": psutil.virtual_memory().available/2**20}
    if a.probe and modules["psutil"]:
        from monitor import run
        env = cpu_env(2)
        for name, code in PROBES.items():
            if not modules[name]:
                report["imports"][name] = {"passed": False, "reason": "module missing"}
                continue
            folder = a.output/name
            try:
                m = run([sys.executable, "-c", code], folder, env, a.rss_mib, 60, a.reserve_mib)
                report["imports"][name] = {"passed": True, "output": (folder/"output.log").read_text().strip(), "measurement": m}
            except Exception as exc:
                report["imports"][name] = {"passed": False, "reason": str(exc)}
        if a.opensplat:
            folder = a.output/"opensplat"
            try:
                run([str(a.opensplat.resolve()), "--resource-info"], folder, env, a.rss_mib, 60, a.reserve_mib)
                info = json.loads((folder/"output.log").read_text())
                report["opensplat"] = {"passed": info["resource_abi"] == 1, "settings": info, "sha256": sha256(a.opensplat)}
            except Exception as exc:
                report["opensplat"] = {"passed": False, "reason": str(exc)}
        checks = report["imports"]
        versions_ok = False
        if all(checks.get(n, {}).get("passed") for n in PROBES):
            torch = json.loads(checks["torch"]["output"])
            colmap = json.loads(checks["pycolmap"]["output"])
            opencv = json.loads(checks["cv2"]["output"])
            versions_ok = (torch["version"].split("+")[0] == "2.7.1" and torch["cuda_build"] is None
                           and torch["hip_build"] is None and colmap["version"] == "4.2.0" and colmap["cpu_api"]
                           and opencv["decoder_thread_api"])
        report["pinned_core_dependencies_match"] = versions_ok
        report["ready_for_attempt"] = bool(versions_ok and report["python_supported_for_candidate_dependencies"]
                                             and report.get("opensplat", {}).get("passed"))
    elif a.probe:
        report["probe_blocker"] = "psutil missing; imports not launched without the watchdog"
    save(a.output/"preflight.json", report)
    print(json.dumps(report, indent=2))
    if a.probe and not report["ready_for_attempt"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
