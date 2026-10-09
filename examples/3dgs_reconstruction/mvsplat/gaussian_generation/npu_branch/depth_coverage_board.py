"""Run the isolated depth ONNX coverage probe on the ARM board."""
import argparse
import json
from pathlib import Path
import shlex
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wheels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--board-root", default="/root/fpga43dgs_reconstruction/depth_coverage_20260930")
    p.add_argument("--existing-site", help="Read-only isolated ONNX site from a prior probe")
    a = p.parse_args()
    if a.out.exists():
        p.error("Output evidence directory already exists")
    a.out.mkdir(parents=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "3dgs_compositor/board"))
    import remote
    from initialize.board_sync import NPU_SOURCE_FILES, upload_sources
    client = remote.connect()
    try:
        root = a.board_root
        board_base = "/root/fpga43dgs_reconstruction/npu_board_v2_20260929"
        env = "/root/fpga43dgs_reconstruction/arm_env"
        status, output = remote.run(client, "test ! -e " + shlex.quote(root) +
                                    " && mkdir -p " + shlex.quote(root + "/wheels"), timeout=15)
        if status:
            raise RuntimeError(output)
        upload_sources(client, root + "/mvsplat", NPU_SOURCE_FILES)
        sftp = client.open_sftp()
        if not a.existing_site:
            for wheel in a.wheels.glob("*.whl"):
                sftp.put(str(wheel), root + "/wheels/" + wheel.name)
        sftp.put(str(Path(__file__).with_name("depth_coverage.py")), root + "/depth_coverage.py")
        site = a.existing_site or root + "/site"
        if not a.existing_site:
            install = ("" + env + "/bin/python3 -m pip install --no-index --no-deps --target " + site +
                       " --find-links " + root + "/wheels onnx==1.19.0 protobuf==6.32.1 ml_dtypes==0.5.1")
            status, output = remote.run(client, install, timeout=120, check=False)
            (a.out / "install.log").write_text(output, encoding="utf-8")
            if status:
                raise RuntimeError("Isolated ONNX install failed: " + output)
        command = ("cd " + board_base + "/mvsplat && timeout 360s env "
                   "LD_LIBRARY_PATH=" + env + "/lib "
                   "PYTHONPATH=" + site + ":" + root + "/mvsplat:" + board_base + "/deps "
                   "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 " + env + "/bin/python3 " +
                   root + "/depth_coverage.py --context " + board_base + "/candidate/input/context.npz " +
                   "--weights " + board_base + "/weights/re10k.ckpt --vendor " +
                   board_base + "/vendor/MVSplat_reference --out " + root + "/result")
        status, output = remote.run(client, command, timeout=375, check=False)
        (a.out / "board.log").write_text(output, encoding="utf-8")
        (a.out / "exit.txt").write_text(str(status) + "\n", encoding="ascii")
        try:
            sftp.get(root + "/result/result.json", str(a.out / "result.json"))
            result = json.loads((a.out / "result.json").read_text(encoding="utf-8"))
            for name, item in result.get("modules", {}).items():
                if item.get("exported"):
                    sftp.get(root + "/result/" + name + ".onnx", str(a.out / (name + ".onnx")))
                    sftp.get(root + "/result/" + name + "_oracle.npz", str(a.out / (name + "_oracle.npz")))
                for prefix_name, prefix in item.get("prefixes", {}).items():
                    if prefix.get("exported"):
                        sftp.get(root + "/result/" + prefix_name + ".onnx", str(a.out / (prefix_name + ".onnx")))
                        sftp.get(root + "/result/" + prefix_name + "_oracle.npz", str(a.out / (prefix_name + "_oracle.npz")))
        except OSError as exc:
            (a.out / "fetch_error.txt").write_text(str(exc), encoding="utf-8")
        print(output)
        print("Board exit:", status)
        if status:
            raise SystemExit(status)
    finally:
        client.close()


if __name__ == "__main__":
    main()
