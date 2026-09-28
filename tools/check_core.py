"""Offline source and interface checks; never connects to a board or installs firmware."""
import ast
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", "vendor", "build", "evidence", "runs", "data", "__pycache__"}


def main():
    paths = []
    for folder in ("examples/3dgs_reconstruction", "examples/3dgs_flicker_hw", "tools"):
        for p in (ROOT/folder).rglob("*.py"):
            if not any(part in SKIP or part.startswith("build_") for part in p.relative_to(ROOT/folder).parts):
                ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p))
                paths.append(p)
    print(f"Python syntax: {len(paths)} files passed", flush=True)
    for rel in ("examples/3dgs_reconstruction/tests/test_interfaces.py",
                "examples/3dgs_reconstruction/bounded/test_controls.py"):
        subprocess.run([sys.executable, str(ROOT/rel), "-v"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT/"examples/3dgs_reconstruction/pipeline.py"), "--help"], check=True)
    print("PASS: source, synthetic artifact contracts and resource controls. No training or board execution.")


if __name__ == "__main__":
    main()
