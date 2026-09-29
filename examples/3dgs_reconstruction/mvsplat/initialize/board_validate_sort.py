"""Run multi-view resident-sort validation against the frozen board renderer."""
import argparse
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent / "3dgs_compositor/board"))
import remote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-root", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--scene", help="Reuse an existing validated scene from another candidate")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--renderer", default="/root/fpga43dgs_releases/20260928T004334/3dgs_renderer_v1_20260928")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = args.board_root.rstrip("/")
    scene = args.scene or (root + "/" + args.run + "/renderer_input")
    result = root + "/" + args.run + "/sort_validation"
    command = shlex.join([
        "env", "LD_LIBRARY_PATH=/root/fpga43dgs_reconstruction/arm_env/lib",
        "PYTHONPATH=" + root + "/mvsplat", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1",
        "/root/fpga43dgs_reconstruction/arm_env/bin/python3",
        root + "/mvsplat/initialize/validate_resident.py",
        "--scene", scene, "--renderer", args.renderer,
        "--binary", root + "/mvsplat/initialize/render_resident", "--out", result])
    client = remote.connect()
    try:
        status, output = remote.run(client, command, timeout=300, check=False)
        (args.out / "board.log").write_text(output, encoding="utf-8")
        (args.out / "exit.txt").write_text(str(status) + "\n", encoding="ascii")
        try:
            client.open_sftp().get(result + "/validation.json", str(args.out / "validation.json"))
        except OSError:
            pass
        print("Board exit:", status)
        if status:
            raise SystemExit(status)
    finally:
        client.close()


if __name__ == "__main__":
    main()
