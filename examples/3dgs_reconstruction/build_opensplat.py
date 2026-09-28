"""Build the pinned OpenSplat checkout with CPU-only LibTorch and local MSVC."""
import json
import os
import subprocess
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent


def main():
    vendor = ROOT / "vendor"
    env = dict(os.environ)
    setup = vendor / "msvc/setup_x64.bat"
    # Static path passed to cmd, not user content. No global PATH or registry edits.
    envbat = vendor / "export_env.bat"
    envbat.write_text('@echo off\ncall "%~dp0msvc\\setup_x64.bat"\nset\n')
    text = subprocess.check_output(["cmd.exe", "/d", "/c", str(envbat)], text=True)
    env.update(line.split("=", 1) for line in text.splitlines() if "=" in line and not line.startswith("="))
    env = {k.upper(): v for k, v in env.items()}
    scripts = ROOT / ".venv/Scripts"
    env["PATH"] = str(scripts) + os.pathsep + env["PATH"]
    build = vendor / "OpenSplat/build_cpu"
    evidence = ROOT / "evidence"
    evidence.mkdir(exist_ok=True)
    commands = [
        [str(scripts/"cmake.exe"), "-S", str(vendor/"OpenSplat"), "-B", str(build), "-G", "Ninja",
         "-DGPU_RUNTIME=CPU", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_COMPILER=cl", "-DCMAKE_CXX_COMPILER=cl",
         "-DCMAKE_PREFIX_PATH="+torch.utils.cmake_prefix_path,
         "-DOpenCV_DIR="+str(vendor/"opencv/build"), "-DOPENSPLAT_BUILD_VISUALIZER=OFF"],
        [str(scripts/"cmake.exe"), "--build", str(build), "--parallel", "4"],
    ]
    for i, cmd in enumerate(commands):
        with (evidence/f"build_{i}.log").open("w") as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    runtime = [str(Path(torch.__file__).parent/"lib"), str(vendor/"opencv/build/x64/vc16/bin")]
    (ROOT/"runtime_paths.json").write_text(json.dumps(runtime, indent=2))
    print(build/"opensplat.exe")


if __name__ == "__main__":
    main()
