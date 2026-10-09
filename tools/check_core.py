"""Offline source and interface checks; never connects to a board or installs firmware."""
import ast
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", ".venv_mvsplat", "vendor", "build", "evidence", "runs", "data", "__pycache__",
        "archive", "archives", "results", "package", "render_branch", "bounded", "npu_frontend"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mvsplat", action="store_true",
                        help="Also run video, lifecycle and NPU ABI tests (requirements-mvsplat-checks.txt)")
    args = parser.parse_args()
    paths = []
    for folder in ("examples/3dgs_reconstruction", "tools"):
        for parent, dirs, files in os.walk(ROOT/folder):
            dirs[:] = [name for name in dirs if name not in SKIP and not name.startswith("build_")]
            for name in files:
                if name.endswith(".py"):
                    p = Path(parent)/name
                    ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p))
                    paths.append(p)
    print(f"Python syntax: {len(paths)} files passed", flush=True)
    for rel in ("examples/3dgs_reconstruction/tests/test_interfaces.py",
                "examples/3dgs_reconstruction/tests/test_reconstruct_entry.py"):
        subprocess.run([sys.executable, str(ROOT/rel), "-v"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT/"examples/3dgs_reconstruction/pipeline.py"), "--help"], check=True)
    if args.mvsplat:
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-v", "-p", "test_*.py"],
                       cwd=ROOT/"examples/3dgs_reconstruction/mvsplat", check=True)
        subprocess.run([sys.executable, str(ROOT/"examples/3dgs_reconstruction/pipeline.py"),
                        "initialize", "--help"], check=True)
    print("PASS: offline source and selected contracts. No training, NPU execution or board access.")


if __name__ == "__main__":
    main()
