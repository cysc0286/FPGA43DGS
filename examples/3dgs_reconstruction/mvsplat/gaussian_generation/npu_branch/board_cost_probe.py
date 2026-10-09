"""Probe each real NPU partition in a separate board worker process."""
import argparse
import json
from pathlib import Path
import shlex
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent.parent / "3dgs_compositor/board"))
import remote
from initialize.board_sync import NPU_SOURCE_FILES, upload_sources


PARTITIONS = ("backbone_cnn", "regressor_residual", "depth_head", "upsampler",
              "proj_feature", "to_gaussians", "to_disparity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--board-root", default="/root/fpga43dgs_reconstruction/npu_cost_probe_20260930")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output evidence directory already exists")
    args.out.mkdir(parents=True)
    source = "/root/fpga43dgs_reconstruction/npu_board_v2_20260929"
    library = "/root/fpga43dgs_reconstruction/npu_known_conv_detailed_20260930/libmgs_npu_detailed.so"
    client = remote.connect()
    summary = dict(scope="single-session partition costs and numerical gate; no model speed claim",
                   partitions={})
    try:
        qroot = shlex.quote(args.board_root)
        remote.run(client, "test ! -e " + qroot + " && mkdir -p " + qroot)
        remote.run(client, "cp -a " + shlex.quote(source + "/mvsplat") + " " + qroot + "/mvsplat")
        upload_sources(client, args.board_root + "/mvsplat", NPU_SOURCE_FILES)
        sftp = client.open_sftp()
        try:
            for name in PARTITIONS:
                result = args.board_root + "/" + name
                command = (
                    "cd " + shlex.quote(args.board_root + "/mvsplat") + " && timeout 120s env "
                    "LD_LIBRARY_PATH=/root/fpga43dgs_reconstruction/arm_env/lib:"
                    "/root/heterogs_npu/sdk_3.36.1/usr/lib/aarch64-linux-gnu "
                    "PYTHONPATH=" + shlex.quote(args.board_root + "/mvsplat") + " "
                    "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "
                    "/root/fpga43dgs_reconstruction/arm_env/bin/python3 "
                    "-m gaussian_generation.npu_branch.oracle_probe --bundle " + shlex.quote(source + "/candidate/compiled")
                    + " --graphs " + shlex.quote(source + "/candidate/graphs")
                    + " --library " + shlex.quote(library)
                    + " --partitions " + shlex.quote(name)
                    + " --out " + shlex.quote(result)
                    + " --diagnostic-continue"
                )
                code, log = remote.run(client, command, timeout=135, check=False)
                (args.out / (name + ".log")).write_text(log, encoding="utf-8")
                entry = dict(exit_code=code, probe_available=False)
                try:
                    sftp.get(result + "/probe.json", str(args.out / (name + ".json")))
                    probe = json.loads((args.out / (name + ".json")).read_text())
                    calls = probe.get("worker", {}).get("calls", [])
                    entry.update(probe_available=True, npu_executed=probe.get("npu_executed"),
                                 comparisons=probe.get("partitions", {}).get(name, []),
                                 calls=calls, error=probe.get("error"))
                except OSError:
                    pass
                summary["partitions"][name] = entry
                (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
                print(name, json.dumps(dict(exit_code=code, probe_available=entry["probe_available"],
                                            calls=len(entry.get("calls", [])))), flush=True)
        finally:
            sftp.close()
    finally:
        client.close()
    if not all(v["probe_available"] for v in summary["partitions"].values()):
        raise RuntimeError("Some partition probes produced no auditable result")


if __name__ == "__main__":
    main()
