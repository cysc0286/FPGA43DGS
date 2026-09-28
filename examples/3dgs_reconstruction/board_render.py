"""Render a newly reconstructed scene through a frozen CPU or FPGA backend."""
import argparse
import hashlib
import json
import shlex
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/"3dgs_compositor/board"))
import remote


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--backend", choices=["cpu_dense", "fpga"], default="cpu_dense")
    p.add_argument("--package", default="/root/fpga43dgs_releases/20260928T004334/3dgs_renderer_v1_20260928")
    a = p.parse_args()
    local = a.run/("board_cpu" if a.backend == "cpu_dense" else "board_fpga")
    local.mkdir(exist_ok=False)
    name = "reconstruction_" + time.strftime("%Y%m%dT%H%M%S") + "_" + a.backend
    dst = "/root/fpga43dgs_reconstruction/" + name
    manifest = json.loads((a.run/"renderer_input/manifest.json").read_text())
    hold = next(c for c in manifest["cameras"] if c["heldout"])
    cameras = [Path(hold["name"]).stem, "novel_midpoint"]
    c = remote.connect()
    try:
        remote.run(c, "mkdir -p /root/fpga43dgs_reconstruction && mkdir " + shlex.quote(dst))
        s = c.open_sftp()
        for f in ["model.ply"] + [cam+".bin" for cam in cameras]:
            src = a.run/"renderer_input"/f
            s.put(str(src), dst+"/"+f)
            _, result = remote.run(c, "sha256sum "+shlex.quote(dst+"/"+f))
            if result.split()[0] != hashlib.sha256(src.read_bytes()).hexdigest():
                raise RuntimeError("Upload checksum mismatch")
        for cam in cameras:
            command = ["python3", a.package+"/render.py", "--backend", a.backend, "--model", dst+"/model.ply",
                       "--camera", dst+"/"+cam+".bin", "--out", dst+"/"+cam]
            remote.run(c, " ".join(shlex.quote(v) for v in command), timeout=300, log=local/(cam+"_ssh.log"))
            folder = local/cam
            folder.mkdir()
            for f in ("frame.bin", "frame.ppm", "frame_timing.csv", "run.log", "result.json"):
                s.get(dst+"/"+cam+"/"+f, str(folder/f))
        s.close()
        (local/"deployment.json").write_text(json.dumps({"remote": dst, "backend": a.backend, "package": a.package}, indent=2))
    finally:
        c.close()


if __name__ == "__main__":
    main()
