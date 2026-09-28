"""The CPU trainer command shared by legacy and bounded orchestration."""
from pathlib import Path


def command(run, executable, iterations, max_gaussians, heldout):
    run = Path(run)
    if iterations < 200 or iterations % 10 or max_gaussians <= 0:
        raise ValueError("Need >=200 iterations divisible by 10 and a positive Gaussian cap")
    return [str(executable), str(run / "project"), "--cpu", "-n", str(iterations),
            "--max-gaussians", str(max_gaussians), "--sh-degree", "0", "--densify-from", "100",
            "--densify-until", str(max(101, iterations*3//4)), "--refine-every", "100",
            "--val-image", heldout, "--val-render", str(run / "validation"),
            "--output-cameras", str(run / "cameras.json"), "-o", str(run / "splat.ply")]


def train(run, executable, profile="arm_candidate", plan=False):
    """Train an existing PoseSet only, with the existing sampled resource limits."""
    import json
    import shutil
    from bounded.common import cpu_env, save, sha256
    from bounded.run import load_profile, inventory
    from ..contracts import PoseSet, GaussianScene
    run, executable = Path(run).resolve(), Path(executable).resolve()
    poses = PoseSet.load(run)
    settings = load_profile(profile)
    args = command(run, executable, settings["iterations"], settings["max_gaussians"], poses.heldout_image)
    if plan:
        return {"status": "plan_only", "command": args, "settings": settings}
    if any((run / n).exists() for n in ("splat.ply", "cameras.json", "validation", "stages/train", "stages/train_capability", "training_request.json")):
        raise ValueError("Training artifacts exist; use a new run to preserve earlier results")
    sfm = json.loads((run / "sfm.json").read_text())
    if sfm["points3D"] > settings["max_gaussians"]:
        raise ValueError("Sparse initialization exceeds Gaussian cap")
    from bounded.monitor import run as measured
    env = cpu_env(settings["threads"], settings["resources"])
    save(run / "training_request.json", {"profile": profile, "settings": settings, "command": args,
         "executable_sha256": sha256(executable), "pose_files": {
             f.relative_to(run).as_posix(): sha256(f) for f in
             sorted([p for p in poses.project.rglob("*") if p.is_file()] + [run/"project.json", run/"sfm.json"])}})
    capability = run / "stages/train_capability"
    measured([str(executable), "--resource-info"], capability, env, settings["rss_mib"], 60, settings["reserve_mib"])
    if json.loads((capability / "output.log").read_text()) != dict(settings["resources"], resource_abi=1):
        raise ValueError("Trainer does not acknowledge resource settings")
    folder = run / "stages/train"
    result = measured(args, folder, env, settings["rss_mib"], settings["timeout_s"], settings["reserve_mib"])
    scene = GaussianScene.load(run)
    save(folder / "receipt.json", {"outputs": inventory(run, "train")})
    save(run / "train_measurement.json", result)
    shutil.copy2(folder / "output.log", run / "train.log")
    # Existing evaluation reads the iteration count from config.json.
    if not (run / "config.json").exists():
        save(run / "config.json", {"iterations": settings["iterations"], "profile": profile,
             "scope": "independent Gaussian module, not a full pipeline/resume receipt"})
    return scene
