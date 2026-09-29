"""Build the unary ICraft bridge on ARM or check SDK headers on Windows."""
import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from common import new_directory, save, sha


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--syntax-only", action="store_true")
    a = p.parse_args(argv)
    if not a.syntax_only and platform.machine().lower() not in ("aarch64", "arm64"):
        p.error("ARM link required; Windows supports --syntax-only")
    root = Path(__file__).resolve().parents[2]
    prefix = next((base for base in (a.sdk.resolve(), a.sdk.resolve()/"usr")
                   if (base/"include/icraft-xrt/core/session.h").is_file()), None)
    if prefix is None:
        raise ValueError("ICraft SDK headers missing")
    inc = prefix/"include"
    lib = prefix/("lib/aarch64-linux-gnu" if (prefix/"lib/aarch64-linux-gnu").is_dir() else "lib")
    if not a.syntax_only and not (lib/"libicraft_xrt.so").is_file():
        raise ValueError("ARM ICraft SDK libraries missing")
    out = new_directory(a.out.resolve())
    source = Path(__file__).with_name("bridge.cpp")
    env = dict(os.environ)
    if os.name == "nt":
        setup = root / "vendor/export_env.bat"
        if setup.is_file():
            lines = subprocess.check_output(["cmd.exe", "/d", "/c", str(setup)], text=True)
            env.update(line.split("=", 1) for line in lines.splitlines() if "=" in line and not line.startswith("="))
        env = {k.upper():v for k,v in env.items()}
        compiler = shutil.which("cl.exe", path=env["PATH"])
        if not compiler:
            raise ValueError("MSVC missing; run from an x64 Visual Studio Developer shell")
        cmd = [compiler, "/nologo", "/std:c++17", "/Zc:__cplusplus", "/EHsc", "/MD", "/utf-8", "/Zs", "/I"+str(inc), str(source)]
    else:
        cmd = ["g++", "-O3", "-std=gnu++17", "-ffp-contract=off", "-shared", "-fPIC",
               str(source), "-o", str(out / "libmgs_npu.so"), "-I"+str(inc), "-L"+str(lib),
               "-Wl,-rpath,"+str(lib), "-licraft_zg330backend", "-licraft_hostbackend",
               "-licraft_xrt", "-licraft_xir", "-licraft_utils", "-ldw", "-ldl", "-pthread"]
        if a.syntax_only:
            cmd.append("-fsyntax-only")
    with (out / "build.log").open("wb") as log:
        result = subprocess.run(cmd, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
    save(out / "result.json", dict(exit_code=result.returncode, command=cmd,
        source_sha256=sha(source), syntax_only=a.syntax_only, npu_executed=False))
    if result.returncode:
        raise RuntimeError("Bridge compilation failed: " + str(out / "build.log"))


if __name__ == "__main__":
    main()
