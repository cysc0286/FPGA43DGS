"""Adapter to the existing frozen CPU/FPGA renderer; no new rasterizer."""
import platform
import subprocess
import sys
from pathlib import Path

from ..contracts import RenderInput, RenderResult


def command(inputs, renderer, out, backend="fpga", work_root="/dev/shm"):
    if backend not in ("fpga", "cpu_dense", "cpu_base"):
        raise ValueError("Unknown frozen renderer backend")
    renderer = Path(renderer).resolve()
    if not (renderer / "render.py").is_file() or not (renderer / "manifest.json").is_file():
        raise ValueError("--renderer must name an extracted frozen renderer package")
    return [sys.executable, str(renderer / "render.py"), "--model", str(inputs.model),
            "--camera", str(inputs.camera), "--out", str(Path(out).resolve()),
            "--backend", backend, "--work-root", str(work_root)]


def render(directory, renderer, out, camera="novel_midpoint.bin", backend="fpga", work_root="/dev/shm", plan=False):
    inputs = RenderInput.load(directory, camera)
    args = command(inputs, renderer, out, backend, work_root)
    if plan:
        return {"status": "plan_only", "command": args, "width": inputs.width, "height": inputs.height}
    if platform.machine().lower() not in ("aarch64", "arm64"):
        raise RuntimeError("Frozen binaries require the ARM64 board; use --plan for host-side interface checks")
    if Path(out).exists():
        raise ValueError("Render output exists; use a fresh directory")
    subprocess.run(args, check=True)
    result = RenderResult.load(out)
    if (result.width, result.height) != (inputs.width, inputs.height):
        raise ValueError("Renderer returned dimensions different from the submitted camera")
    return result
