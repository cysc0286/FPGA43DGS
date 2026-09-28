"""Budgeted CPU chain; stage-level resume with content verification, no SSH."""
import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path
try:
    from .common import ROOT, cpu_env, save, sha256
except ImportError:
    from common import ROOT, cpu_env, save, sha256

sys.path.insert(0, str(ROOT.parent))
from modules.gaussian_generation.training import command as training_command

STAGE_PRODUCTS = {
    "frames": ["frames", "frames.json"],
    "sfm": ["project", "project.json", "sfm.json", "database.db", "sparse", "undistorted"],
    "train": ["splat.ply", "cameras.json", "validation"],
    "export": ["renderer_input", "adapter_validation.json"],
    "evaluate": ["quality.json", "comparison.png"],
}


def inventory(root, name):
    items = {}
    for relative in STAGE_PRODUCTS[name]:
        p = root/relative
        if not p.exists():
            raise ValueError("Missing stage output: " + str(p))
        files = sorted(p.rglob("*")) if p.is_dir() else [p]
        files = [f for f in files if f.is_file()]
        if not files:
            raise ValueError("Empty stage output: " + str(p))
        for f in files:
            items[f.relative_to(root).as_posix()] = sha256(f)
    return items


def load_profile(name):
    config = json.loads((ROOT/"profiles.json").read_text())[name]
    for key in ("frames", "sfm_width", "render_width", "max_features", "overlap", "iterations",
                "max_gaussians", "threads", "rss_mib", "reserve_mib", "timeout_s"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError("Invalid profile: " + key)
    if config["frames"] < 5 or config["iterations"] < 200 or config["iterations"] % 10:
        raise ValueError("Need >=5 frames and >=200 iterations divisible by 10")
    if config["render_width"] > config["sfm_width"]:
        raise ValueError("Training cannot enlarge the source image")
    return config


def commands(run, video, exe, c, heldout="<resolved after SfM>"):
    base = [sys.executable, str(ROOT.parent/"stages.py")]
    return {
        "frames": base+["frames", "--run", str(run), "--video", str(video), "--frames", str(c["frames"]),
                        "--width", str(c["sfm_width"]), "--threads", str(c["threads"])],
        "sfm": base+["sfm", "--run", str(run), "--threads", str(c["threads"]), "--width", str(c["sfm_width"]),
                     "--render-width", str(c["render_width"]), "--max-features", str(c["max_features"]),
                     "--overlap", str(c["overlap"])],
        "train": training_command(run, exe, c["iterations"], c["max_gaussians"], heldout),
        "export": base+["export", "--run", str(run)],
        "evaluate": [sys.executable, str(ROOT.parent/"evaluate.py"), "--run", str(run)],
    }


def verify_receipt(root, name, command):
    receipt = json.loads((root/"stages"/name/"receipt.json").read_text())
    measure = json.loads((root/"stages"/name/"measurement.json").read_text())
    if measure["status"] != "passed" or measure["command"] != command:
        raise ValueError("Resume stage command/status mismatch: " + name)
    if receipt["outputs"] != inventory(root, name):
        raise ValueError("Resume output content changed: " + name)
    return measure


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=["cpu_control", "arm_candidate"], default="arm_candidate")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--opensplat", type=Path, required=True)
    p.add_argument("--through", choices=["sfm", "train", "export"], default="export")
    p.add_argument("--plan", action="store_true", help="Print commands/settings only; do not create a run or execute stages")
    p.add_argument("--resume", action="store_true", help="Only completed stages; not optimizer-state resume")
    a = p.parse_args()
    c = load_profile(a.profile)
    run, video, exe = a.run.resolve(), a.video.resolve(), a.opensplat.resolve()
    plan = {"profile": a.profile, "settings": c, "commands": commands(run, video, exe, c),
            "stages": ["frames", "sfm"] + ([] if a.through == "sfm" else ["train"])
                      + (["export", "evaluate"] if a.through == "export" else []),
            "device": "CPU", "board_access": False,
            "memory_policy": "50ms sampled RSS watchdog, not an OS-enforced limit or a guarantee of board fit"}
    if a.plan:
        print(json.dumps(plan, indent=2))
        return
    # Delay optional dependency imports until execution; plan works with stdlib Python.
    from monitor import run as measured
    import psutil
    if run.exists() and not a.resume:
        raise ValueError("Run exists. Use a fresh name or explicit --resume")
    if a.resume and not (run/"config.json").is_file():
        raise ValueError("Cannot resume an uninitialized run")
    sources = list(ROOT.glob("*.py")) + [ROOT/"profiles.json", ROOT.parent/"stages.py", ROOT.parent/"evaluate.py"]
    sources += sorted((ROOT.parent/"modules").rglob("*.py")) + [ROOT.parent/"pipeline.py"]
    config = {"schema": 2, "profile": a.profile, "settings": c, "iterations": c["iterations"],
              "video": str(video), "video_sha256": sha256(video), "opensplat": str(exe), "opensplat_sha256": sha256(exe),
              "source_sha256": {f.relative_to(ROOT.parent).as_posix(): sha256(f) for f in sorted(sources)},
              "python": sys.version, "platform": platform.platform(),
              "packages": {n: importlib.metadata.version(n) for n in ["pycolmap", "numpy", "plyfile", "psutil"]},
              "device": "cpu"}
    if a.resume and json.loads((run/"config.json").read_text()) != config:
        raise ValueError("Resume configuration/input/binary/source/environment changed; use a fresh run")
    run.mkdir(parents=True, exist_ok=True)
    if not a.resume:
        save(run/"config.json", config)
        save(run/"plan.json", plan)
        save(run/"environment.json", {"cpu_count": psutil.cpu_count(), "platform": platform.platform(),
             "memory_total_mib": psutil.virtual_memory().total/2**20,
             "memory_available_mib": psutil.virtual_memory().available/2**20})
    env = cpu_env(c["threads"], c["resources"])
    cap = run/"capability"
    if not cap.exists():
        measured([str(exe), "--resource-info"], cap, env, c["rss_mib"], 60, c["reserve_mib"])
    elif json.loads((cap/"measurement.json").read_text())["status"] != "passed":
        raise ValueError("Previous capability check failed; evidence preserved")
    abi = json.loads((cap/"output.log").read_text())
    expected = dict(c["resources"], resource_abi=1)
    if abi != expected:
        raise ValueError("Executable did not acknowledge the resource settings")
    records = {}
    try:
        for name in plan["stages"]:
            heldout = json.loads((run/"project.json").read_text())["heldout_image"] if name == "train" else "unused"
            cmd = commands(run, video, exe, c, heldout)[name]
            folder = run/"stages"/name
            if folder.exists():
                if not a.resume:
                    raise ValueError("Unexpected stage folder")
                records[name] = verify_receipt(run, name, cmd)
                print("Verified completed stage", name, flush=True)
                continue
            if name == "train" and json.loads((run/"sfm.json").read_text())["points3D"] > c["max_gaussians"]:
                raise ValueError("Initial sparse points exceed the Gaussian cap; use a new profile/run, no silent thinning")
            print("Starting", name, flush=True)
            records[name] = measured(cmd, folder, env, c["rss_mib"], c["timeout_s"], c["reserve_mib"])
            save(folder/"receipt.json", {"outputs": inventory(run, name)})
            # Retain compatibility with the existing offline quality/evidence readers.
            save(run/(name+"_measurement.json"), records[name])
            shutil.copy2(folder/"output.log", run/(name+".log"))
        status = "passed_through_"+a.through
    except BaseException:
        status = "incomplete"
        raise
    finally:
        save(run/"pipeline_measurement.json", {"status": status, "device": "cpu", "stages": records,
             "stage_wall_sum_s": sum(v["wall_s"] for v in records.values()),
             "peak_stage_rss_mib": max([v["peak_process_tree_rss_mib_sampled"] for v in records.values()] or [0]),
             "boundary": "includes evaluation when requested; excludes capability/setup/compile; failed stage logs remain in stages/"})
    print(status, run, flush=True)


if __name__ == "__main__":
    main()
