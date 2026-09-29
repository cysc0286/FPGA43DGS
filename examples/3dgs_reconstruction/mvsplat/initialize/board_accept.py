"""Run warm candidate on the board and retrieve the first-frame evidence."""
import argparse
import json
from pathlib import Path
import shlex
import sys
from pathlib import PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent / "3dgs_compositor/board"))
import remote


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--board-root", default="/root/fpga43dgs_reconstruction/initialize_candidate_20260929")
    p.add_argument("--renderer", default="/root/fpga43dgs_releases/20260928T004334/3dgs_renderer_v1_20260928")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--prepare-threads", type=int, default=1)
    schedule = p.add_mutually_exclusive_group()
    schedule.add_argument("--serial-prepare", dest="serial_prepare", action="store_true")
    schedule.add_argument("--overlap-prepare", dest="serial_prepare", action="store_false")
    p.set_defaults(serial_prepare=True)
    p.add_argument("--backend", choices=("cpu", "npu"), default="cpu")
    p.add_argument("--partition-bundle")
    p.add_argument("--partitions", nargs="+")
    p.add_argument("--npu-library")
    p.add_argument("--oracle-graphs")
    p.add_argument("--buffer-policy", choices=("shared", "per_partition"), default="shared")
    p.add_argument("--buffer-limit-mib", type=int, default=128)
    a = p.parse_args()
    if (PurePosixPath(a.run).is_absolute() or ".." in PurePosixPath(a.run).parts
            or not a.run or a.run == "."):
        p.error("Run must be a fresh relative board directory")
    if a.backend == "npu" and not (a.partition_bundle and a.partitions and a.npu_library):
        p.error("NPU requires compiled bundle, selected partitions and bridge")
    a.out.mkdir(parents=True, exist_ok=False)
    root = a.board_root.rstrip("/")
    args = ["env", "LD_LIBRARY_PATH=/root/fpga43dgs_reconstruction/arm_env/lib",
            "LD_PRELOAD=/root/fpga43dgs_reconstruction/arm_env/lib/libgomp.so.1 "
            "/root/fpga43dgs_reconstruction/arm_env/lib/python3.11/site-packages/torch.libs/libgomp-58a43326.so.1.0.0",
            "PYTHONPATH=" + root + "/deps",
            "OPENBLAS_NUM_THREADS=" + str(a.threads), "OMP_NUM_THREADS=" + str(a.threads),
            "MKL_NUM_THREADS=" + str(a.threads),
            "/root/fpga43dgs_reconstruction/arm_env/bin/python3", "pipeline.py", "initialize",
            "--video", "input/nyu_snippet_curl.mp4", "--out", a.run,
            "--weights", "weights/re10k.ckpt", "--vendor", "vendor/MVSplat_reference",
            "--renderer", a.renderer, "--threads", str(a.threads),
            "--prepare-threads", str(a.prepare_threads), "--backend", a.backend]
    if a.serial_prepare:
        args.append("--serial-prepare")
    else:
        args.append("--overlap-prepare")
    if a.backend == "npu":
        if a.oracle_graphs:
            args += ["--oracle-graphs", a.oracle_graphs]
        args += ["--partition-bundle", a.partition_bundle, "--npu-library", a.npu_library,
                 "--buffer-policy", a.buffer_policy, "--buffer-limit-mib", str(a.buffer_limit_mib),
                 "--partitions"] + a.partitions
    monitor = ["/usr/bin/python3", "mvsplat/measure.py", "--out", a.run+".measurement.json",
               "--rss-mib", "650", "--reserve-mib", "128", "--timeout", "600", "--"]
    command = "cd " + shlex.quote(root) + " && " + shlex.join(monitor + args)
    client = remote.connect()
    outcome = dict(remote_run_completed=False, evidence_collected=False)
    try:
        try:
            remote.run(client, command, timeout=660, log=a.out / "board.log")
            outcome["remote_run_completed"] = True
        except Exception as exc:
            outcome["run_error"] = type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            try:
                transport = client.get_transport()
                if transport is None or not transport.is_active():
                    raise RuntimeError("SSH inactive; remote completion unknown; no rerun attempted")
                sftp = client.open_sftp()
                try:
                    for name in ("events.json", "warm_result.json", "input/input.json",
                             "input/input.initial.json", "renderer_input/manifest.json",
                             "renderer_input/adapter_validation.json",
                             "inference/inference.json",
                             "board/frame_000007_fpga/result.json", "board/frame_000007_fpga/frame.bin",
                                 "board/frame_000007_fpga/frame.ppm", "renderer.log"):
                        target = a.out / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            sftp.get(root + "/" + a.run + "/" + name, str(target))
                        except FileNotFoundError:
                            pass
                    sftp.get(root+"/"+a.run+".measurement.json", str(a.out/"measurement.json"))
                    outcome["evidence_collected"] = True
                finally:
                    sftp.close()
            except Exception as exc:
                outcome["collection_error"] = type(exc).__name__ + ": " + str(exc)
            (a.out/"acceptance_status.json").write_text(json.dumps(outcome, indent=2))
    finally:
        client.close()
    result = json.loads((a.out / "warm_result.json").read_text(encoding="utf-8"))
    if not result["complete"] or not outcome["evidence_collected"]:
        raise RuntimeError("Run or evidence collection incomplete")
    print(json.dumps(result["timing"], indent=2))


if __name__ == "__main__":
    main()
