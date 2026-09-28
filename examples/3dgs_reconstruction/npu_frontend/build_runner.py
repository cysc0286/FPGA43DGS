"""Local build only: CPU protocol test or ARM NPU runner; never deploys files."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sdk_paths(root, need_libraries=True):
    root = root.resolve()
    for prefix in (root, root/"usr"):
        include = prefix/"include"
        if not (include/"icraft-xrt/core/session.h").is_file():
            continue
        for lib in (prefix/"lib/aarch64-linux-gnu", prefix/"lib"):
            if not need_libraries or (lib/"libicraft_xrt.so").is_file():
                return include, lib
    raise ValueError("ICraft headers/libraries not found under SDK prefix or extracted Debian usr tree")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["cpu", "npu"], required=True)
    p.add_argument("--sdk", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--syntax-only", action="store_true")
    p.add_argument("--shared", action="store_true", help="Build the in-process NPU bridge")
    a = p.parse_args()
    if a.kind == "npu" and not a.sdk:
        p.error("NPU build requires --sdk")
    if a.shared and a.kind != "npu":
        p.error("Shared bridge requires NPU kind")
    if a.kind == "npu" and not a.syntax_only and platform.machine().lower() not in ("aarch64", "arm64"):
        p.error("Executable NPU build is intended for ARM64; use --syntax-only to check host headers")
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    source = ROOT/("board/npu_session.cpp" if a.shared else "board/block_runner.cpp")
    target = out/("libhgs_matcher.so" if a.shared else "block_runner_"+a.kind+(".exe" if os.name == "nt" else ""))
    include, lib = sdk_paths(a.sdk, not a.syntax_only) if a.kind == "npu" else (None, None)
    if os.name == "nt":
        setup = ROOT.parent/"vendor/export_env.bat"
        if setup.exists():
            text = subprocess.check_output(["cmd.exe", "/d", "/c", str(setup)], text=True)
            env.update(line.split("=", 1) for line in text.splitlines() if "=" in line and not line.startswith("="))
        env = {k.upper(): v for k, v in env.items()}
        compiler = shutil.which("cl.exe", path=env["PATH"])
        if not compiler:
            raise RuntimeError("MSVC cl.exe missing; run in its developer shell")
        cmd = [compiler, "/nologo", "/std:c++17", "/Zc:__cplusplus", "/EHsc", "/O2", "/fp:strict", "/MD", "/utf-8",
               str(source), "/Fo"+str(out/"block_runner.obj"), "/Fe"+str(target)]
        cmd += ["/DHGS_CPU_REFERENCE"] if a.kind == "cpu" else ["/I"+str(include)]
        if a.syntax_only:
            cmd.append("/Zs")
    else:
        # ICraft uses GNU empty variadic-macro comma elision in its public headers.
        cmd = ["g++", "-O3", "-std=gnu++17", "-ffp-contract=off", str(source), "-o", str(target)]
        if a.kind == "cpu":
            cmd += ["-DHGS_CPU_REFERENCE"]
        else:
            cmd += ["-I"+str(include), "-L"+str(lib), "-Wl,-rpath,"+str(lib),
                    "-licraft_zg330backend", "-licraft_hostbackend", "-licraft_xrt", "-licraft_xir",
                    "-licraft_utils", "-ldw", "-ldl", "-pthread"]
        if a.syntax_only:
            cmd.append("-fsyntax-only")
        if a.shared:
            cmd += ["-shared", "-fPIC"]
    request = {"command": cmd, "kind": a.kind, "syntax_only": a.syntax_only, "machine": platform.machine(),
               "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "npu_executed": False}
    (out/"request.json").write_text(json.dumps(request, indent=2))
    with (out/"build.log").open("wb") as log:
        run = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=out, timeout=180)
    request["exit_code"] = run.returncode
    if not run.returncode and not a.syntax_only:
        request["executable_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    (out/"result.json").write_text(json.dumps(request, indent=2))
    if run.returncode:
        raise RuntimeError("Build failed: "+str(out/"build.log"))
    print(target if not a.syntax_only else out/"result.json")


if __name__ == "__main__":
    main()
