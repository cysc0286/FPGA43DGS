"""Build the isolated CPU variant with bounded, separately measured compiler jobs."""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from common import ROOT, save, sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=ROOT.parent/"vendor/OpenSplat_bounded_v1")
    ap.add_argument("--jobs", type=int, default=1, help="Compiler jobs; this is independent of training threads")
    ap.add_argument("--pch", choices=["auto", "on", "off"], default="auto", help="Disable on the PC too when validating lean explicit headers")
    ap.add_argument("--opencv", type=Path)
    ap.add_argument("--evidence", type=Path)
    ap.add_argument("--build-dir", type=Path, help="Fresh directory when changing compilers")
    ap.add_argument("--rss-mib", type=int, default=560 if os.name != "nt" else 0)
    ap.add_argument("--reserve-mib", type=int, default=128)
    a = ap.parse_args()
    if a.jobs < 1:
        ap.error("--jobs must be positive")
    source = a.source.resolve()
    manifest = json.loads((source/"heterogs_source.json").read_text())
    for name, expected in manifest["modified_files"].items():
        if sha256(source/name) != expected:
            raise ValueError("Prepared source modified: " + name)
    # Keep LibTorch out of this long-lived coordinator: on a 1 GiB board,
    # retaining its import during C++ compilation needlessly consumes RAM.
    probe = subprocess.check_output([sys.executable, "-c",
        "import torch,json; print(json.dumps(dict(version=torch.__version__,cuda=torch.version.cuda,"
        "hip=getattr(torch.version,'hip',None),prefix=torch.utils.cmake_prefix_path)))"], text=True)
    torch_info = json.loads(probe)
    if torch_info["cuda"] is not None or torch_info["hip"]:
        raise ValueError("Use a CPU-only LibTorch/PyTorch distribution")
    evidence = a.evidence or ROOT.parent/"evidence"/("bounded_build_"+datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))
    evidence.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    extra = []
    opencv = a.opencv
    if os.name == "nt":
        envbat = ROOT.parent/"vendor/export_env.bat"
        if envbat.exists():
            text = subprocess.check_output(["cmd.exe", "/d", "/c", str(envbat)], text=True)
            env.update(line.split("=", 1) for line in text.splitlines() if "=" in line and not line.startswith("="))
        env = {k.upper(): v for k, v in env.items()}
        extra = ["-DCMAKE_C_COMPILER=cl", "-DCMAKE_CXX_COMPILER=cl"]
        opencv = opencv or ROOT.parent/"vendor/opencv/build"
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    cmake = shutil.which("cmake", path=env["PATH"])
    if not cmake:
        raise RuntimeError("cmake is missing")
    build = a.build_dir.resolve() if a.build_dir else source/"build_bounded"
    cmd = [cmake, "-S", str(source), "-B", str(build), "-G", "Ninja", "-DGPU_RUNTIME=CPU",
           "-DCMAKE_BUILD_TYPE=Release", "-DOPENSPLAT_BUILD_VISUALIZER=OFF",
           "-DOPENSPLAT_USE_PCH="+(("ON" if os.name == "nt" else "OFF") if a.pch == "auto" else a.pch.upper()),
           "-DCMAKE_PREFIX_PATH="+torch_info["prefix"]] + extra
    if opencv:
        cmd.append("-DOpenCV_DIR="+str(opencv.resolve()))
    commands = [cmd, [cmake, "--build", str(build), "--target", "opensplat", "--parallel", str(a.jobs)]]
    save(evidence/"request.json", {"commands": commands, "python": sys.version,
         "torch": torch_info["version"], "source": manifest, "scope": "native CPU compilation; not training",
         "limits": {"rss_mib": a.rss_mib, "reserve_mib": a.reserve_mib}})
    from monitor import run as measured
    for i, cmd in enumerate(commands):
        folder = evidence/("step_%d" % i)
        try:
            measured(cmd, folder, env, a.rss_mib, 3600, a.reserve_mib)
        finally:
            if (folder/"output.log").exists():
                shutil.copy2(folder/"output.log", evidence/("build_%d.log" % i))
    exe = build/("opensplat.exe" if os.name == "nt" else "opensplat")
    save(evidence/"result.json", {"executable": str(exe), "sha256": sha256(exe), "success": True})
    print(exe)


if __name__ == "__main__":
    main()
