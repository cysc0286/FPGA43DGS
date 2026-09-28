"""Versioned, measured video -> COLMAP CPU -> OpenSplat CPU -> renderer inputs."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent
from modules.gaussian_generation.training import command as training_command


def measured(command, root, name, env):
    start = time.perf_counter()
    peak = 0
    with (root / (name + ".log")).open("w", encoding="utf-8") as log:
        proc = subprocess.Popen([str(c) for c in command], stdout=log, stderr=subprocess.STDOUT, env=env)
        parent = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                procs = [parent] + parent.children(recursive=True)
                rss = sum(p.memory_info().rss for p in procs if p.is_running())
                peak = max(peak, rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(.1)
    result = {"command": [str(c) for c in command], "wall_s": time.perf_counter()-start,
              "peak_process_tree_rss_mib_sampled": peak/1024**2, "rss_sample_interval_s": .1,
              "returncode": proc.returncode}
    (root/(name+"_measurement.json")).write_text(json.dumps(result, indent=2))
    print(name, json.dumps(result), flush=True)
    if proc.returncode:
        raise RuntimeError(f"{name} failed: inspect {root / (name+'.log')}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--opensplat", type=Path)
    p.add_argument("--through", choices=["sfm", "train", "export"], default="export")
    p.add_argument("--resume", action="store_true", help="Continue only stages whose previous measurement succeeded")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--max-gaussians", type=int, default=20000)
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    if a.iterations < 200 or a.iterations % 10 or a.frames < 5 or a.max_gaussians < 1:
        raise ValueError("Use >=200 iterations divisible by 10, >=5 frames and a positive Gaussian budget")
    a.run = a.run.resolve()
    if a.run.exists() and not a.resume:
        raise ValueError("Run exists. Use a fresh name to preserve evidence, or explicitly --resume")
    a.run.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))
    runtime_paths = ROOT / "runtime_paths.json"
    if runtime_paths.exists():
        env["PATH"] = os.pathsep.join(json.loads(runtime_paths.read_text())) + os.pathsep + env.get("PATH", "")
    config = {"video": str(a.video.resolve()), "iterations": a.iterations, "max_gaussians": a.max_gaussians,
              "frames": a.frames, "threads": a.threads, "platform": platform.platform(),
              "cpu": platform.processor(), "python": sys.version, "device": "cpu"}
    cfg = a.run/"config.json"
    if cfg.exists() and json.loads(cfg.read_text()) != config:
        raise ValueError("Resume configuration mismatch")
    prior_frames = a.run/"frames.json"
    if a.resume and prior_frames.exists():
        expected = json.loads(prior_frames.read_text())["video_sha256"]
        if hashlib.sha256(a.video.read_bytes()).hexdigest() != expected:
            raise ValueError("Resume input video content changed")
    cfg.write_text(json.dumps(config, indent=2))
    def stage(name, cmd):
        prior = a.run/(name+"_measurement.json")
        if a.resume and prior.exists():
            old = json.loads(prior.read_text())
            if old["returncode"] == 0:
                print("Preserved completed stage", name, flush=True)
                return
            raise RuntimeError("Failed stage preserved. Inspect and choose fresh run or explicit repair")
        measured(cmd, a.run, name, env)
    base = [sys.executable, ROOT/"stages.py"]
    stage("frames", base+["frames", "--run", a.run, "--video", a.video.resolve(), "--frames", a.frames])
    stage("sfm", base+["sfm", "--run", a.run, "--threads", a.threads])
    if a.through == "sfm":
        return
    if a.opensplat is None:
        raise ValueError("--opensplat must identify the CPU-built executable")
    project = json.loads((a.run/"project.json").read_text())
    stage("train", training_command(a.run, a.opensplat.resolve(), a.iterations,
                                    a.max_gaussians, project["heldout_image"]))
    if a.through == "export":
        stage("export", base+["export", "--run", a.run])
    records = {s: json.loads((a.run/(s+"_measurement.json")).read_text())
               for s in ("frames", "sfm", "train", "export") if (a.run/(s+"_measurement.json")).exists()}
    summary = {"device": "cpu", "stage_wall_sum_s": sum(r["wall_s"] for r in records.values()),
               "peak_stage_rss_mib": max(r["peak_process_tree_rss_mib_sampled"] for r in records.values()),
               "resumed": a.resume, "stages": records,
               "boundary": "sum of measured subprocess stages, excludes setup/download/compile and board rendering"}
    (a.run/"pipeline_measurement.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
