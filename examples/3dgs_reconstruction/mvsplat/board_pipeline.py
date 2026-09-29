"""Run stored video -> poses -> MVSplat -> frozen renderer entirely on ARM."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def build_parser(*, add_help=True):
    p = argparse.ArgumentParser(description=__doc__, add_help=add_help)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--pose-mode", choices=("full_sfm", "fast_pair"), default="full_sfm")
    p.add_argument("--focal-ratio", type=float, default=0.9)
    p.add_argument("--fused", action="store_true", help="Run fast_pair preparation/inference/export in one process")
    p.add_argument("--renderer", type=Path, default=Path("/root/fpga43dgs_releases/20260928T004334/3dgs_renderer_v1_20260928"))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--input-ended-at-epoch", type=float,
                   help="Wall-clock time recorded when the board received the final input frame")
    return p


def main(argv=None):
    p = build_parser()
    a = p.parse_args(argv)
    if platform.machine() not in ("aarch64", "arm64"):
        p.error("This acceptance entry must execute on the ARM board")
    if not a.video.is_file() or a.repeats < 1 or a.threads < 1 or a.focal_ratio <= 0:
        p.error("Invalid input, threads or repeat count")
    if a.fused and a.pose_mode != "fast_pair":
        p.error("--fused requires --pose-mode fast_pair")
    if a.input_ended_at_epoch is not None and (not math.isfinite(a.input_ended_at_epoch)
            or a.input_ended_at_epoch <= 0 or a.input_ended_at_epoch > time.time()):
        p.error("--input-ended-at-epoch must be a finite, positive past receipt time")
    a.out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    runtime = source.parent
    start = time.perf_counter()
    record = dict(complete=False, host=platform.node(), machine=platform.machine(),
                  boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                  input_video_sha256=sha(a.video), video=str(a.video), size=a.size,
                  pose_mode=a.pose_mode, fused=a.fused,
                  focal_ratio=a.focal_ratio if a.pose_mode == "fast_pair" else None,
                  input_ended_at_epoch=a.input_ended_at_epoch,
                  execution_environment=dict(python=sys.executable, python_version=platform.python_version(),
                      threads=a.threads, environment={k: os.environ.get(k) for k in
                      ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "OMP_NUM_THREADS",
                       "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}),
                  runtime_files_sha256={str(f.relative_to(runtime)):sha(f) for f in source.glob("*.py")},
                  stages=[], renderer_runs=[])
    def stage(name, command, *, env=None, timeout=600):
        measure = a.out / (name+".measurement.json")
        log = a.out / (name+".log")
        args = ["/usr/bin/python3", str(source/"measure.py"), "--out", str(measure),
                "--rss-mib", "650", "--reserve-mib", "128", "--timeout", str(timeout), "--"]+list(map(str,command))
        print("BEGIN",name,flush=True)
        try:
            with log.open("w") as f:
                subprocess.run(args, check=True, stdout=f, stderr=subprocess.STDOUT, env=env)
        finally:
            value = json.loads(measure.read_text()) if measure.exists() else {"status":"missing measurement"}
            record["stages"].append(dict(name=name, **value))
            save(a.out/"pipeline_result.json", record)
        print("END",name,value["status"],"seconds",value["wall_seconds"],"RSS_MiB",value["peak_process_tree_rss_mib_sampled"],flush=True)
    try:
        artifacts = a.out/"reconstruction" if a.fused else a.out
        if a.fused:
            stage("reconstruct_fast", [sys.executable, source/"reconstruct_fast.py", "--video", a.video,
                  "--out", artifacts, "--size", str(a.size), "--threads", str(a.threads),
                  "--focal-ratio", str(a.focal_ratio), "--weights", runtime/"weights/re10k.ckpt",
                  "--vendor", runtime/"vendor/MVSplat_reference"])
        elif a.pose_mode == "fast_pair":
            stage("prepare_fast", [sys.executable, source/"prepare_fast.py", "--video", a.video,
                  "--out", a.out/"input", "--size", str(a.size), "--width", "480",
                  "--threads", str(a.threads), "--focal-ratio", str(a.focal_ratio)])
        else:
            pose = a.out/"pose"
            stage("video", [sys.executable, runtime/"pipeline.py", "video", "--video", a.video, "--run", pose,
                            "--frames", "30", "--width", "480", "--threads", str(a.threads)])
            stage("pose", [sys.executable, runtime/"pipeline.py", "pose", "--run", pose, "--width", "480",
                           "--render-width", "160", "--threads", str(a.threads), "--max-features", "2048"])
            stage("prepare", [sys.executable, source/"prepare.py", "--pose-run", pose, "--out", a.out/"input", "--size", str(a.size)])
        if not a.fused:
            stage("infer", [sys.executable, source/"infer.py", "--input", a.out/"input", "--weights", runtime/"weights/re10k.ckpt",
                            "--vendor", runtime/"vendor/MVSplat_reference", "--out", a.out/"inference", "--threads", str(a.threads),
                            "--depth-chunk", "16"])
            stage("export", [sys.executable, source/"export.py", "--input", a.out/"input", "--inference", a.out/"inference", "--out", a.out/"renderer_input"])
        record["first_render_start_seconds"] = time.perf_counter()-start
        record["artifacts_directory"] = str(artifacts)
        metadata = json.loads((artifacts/"renderer_input/manifest.json").read_text())
        # Private CPU libraries are needed for MVSplat, not the frozen SDK binary.
        render_env = dict(os.environ)
        render_env.pop("LD_PRELOAD", None)
        render_env.pop("PYTHONPATH", None)
        render_env["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        render_env["LD_LIBRARY_PATH"] = "/root/heterogs_npu/sdk_3.36.1/usr/lib/aarch64-linux-gnu"
        first = True
        for camera in metadata["cameras"]:
            if camera["role"] != "target":
                continue
            for backend in ("fpga", "cpu_dense"):
                for repeat in range(-1, a.repeats):
                    name = Path(camera["file"]).stem+"_"+backend+"_"+("warmup" if repeat == -1 else str(repeat))
                    folder = a.out/"board"/name
                    stage(name, ["/usr/bin/python3", a.renderer/"render.py", "--model", artifacts/"renderer_input/model.ply",
                                 "--camera", artifacts/"renderer_input"/camera["file"], "--out", folder, "--backend", backend], env=render_env)
                    value = json.loads((folder/"result.json").read_text())
                    if not value["complete"] or value["frame_sha256"] != sha(folder/"frame.bin"):
                        raise ValueError("Incomplete render or framebuffer hash changed")
                    if first and backend == "fpga":
                        verified_at = time.time()
                        record["first_verified_fpga_target"] = dict(camera=camera["file"], directory=name,
                                                                     frame_sha256=value["frame_sha256"],
                                                                     completed_at_epoch=verified_at)
                        record["pipeline_start_to_first_verified_fpga_frame_seconds"] = time.perf_counter()-start
                        if a.input_ended_at_epoch is not None:
                            record["input_end_to_first_verified_fpga_frame_seconds"] = verified_at-a.input_ended_at_epoch
                        first = False
                    record["renderer_runs"].append(dict(directory=name, camera=camera["file"], warmup=repeat == -1, **value))
        record["complete"] = True
        record["gaussians"] = metadata["gaussians"]
        record["training_steps"] = 0
        record["runtime_compute"] = {"video_pose_inference_export":"ARM CPU", "render":"ARM CPU preprocessing plus FPGA, and CPU Dense comparison", "NPU":"not used", "computer":"orchestration and result review only"}
    finally:
        record["total_validation_seconds"] = time.perf_counter()-start
        save(a.out/"pipeline_result.json", record)
    print(json.dumps({k:v for k,v in record.items() if k not in ("stages","renderer_runs","runtime_files_sha256")}, indent=2),flush=True)


if __name__ == "__main__":
    main()
