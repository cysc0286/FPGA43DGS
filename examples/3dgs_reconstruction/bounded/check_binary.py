"""Local functional checks of the compiled controls; not an ARM benchmark."""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path
import numpy as np
from plyfile import PlyData
from common import ROOT, cpu_env, save, sha256
from monitor import run
from run import load_profile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--opensplat", required=True, type=Path)
    p.add_argument("--reference-run", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    exe = a.opensplat.resolve()
    settings = load_profile("arm_candidate")["resources"]
    env = cpu_env(2, settings)
    result = {"scope": "host CPU controls and short backward/densification check, not performance/quality acceptance",
              "executable_sha256": sha256(exe), "settings": settings, "checks": {}}
    try:
        run([exe, "--resource-info"], a.output/"capability", env, 3072, 60, 128)
        info = json.loads((a.output/"capability/output.log").read_text())
        assert info == dict(settings, resource_abi=1)
        result["checks"]["settings_acknowledged"] = True
        for key, bad in [("HGS_CPU_THREADS", "-1"), ("HGS_IMAGE_SLOTS", "2"), ("HGS_GRADIENT_MIB", "abc")]:
            folder = a.output/key
            try:
                run([exe, "--resource-info"], folder, dict(env, **{key: bad}), 3072, 60, 128)
            except RuntimeError:
                assert json.loads((folder/"measurement.json").read_text())["status"] == "process_failed"
            else:
                raise AssertionError("Accepted invalid setting: " + key)
            result["checks"][key+"_rejects_invalid"] = True
        work = a.output/"training"
        work.mkdir()
        shutil.copytree(a.reference_run/"project", work/"project")
        meta = json.loads((a.reference_run/"project.json").read_text())
        initial = json.loads((a.reference_run/"sfm.json").read_text())["points3D"]
        # Deliberately only 12 spare entries: exercises the capacity boundary.
        cap = initial + 12
        cmd = [exe, work/"project", "--cpu", "-n", "200", "--max-gaussians", str(cap),
               "--sh-degree", "0", "--densify-from", "1", "--densify-until", "150", "--refine-every", "100",
               "--save-every", "100", "--val-image", meta["heldout_image"], "--val-render", work/"validation",
               "--output-cameras", work/"cameras.json", "-o", work/"splat.ply"]
        measurement = run(cmd, a.output/"train_process", env, 3072, 300, 128)
        points = {}
        for f in sorted(work.glob("splat*.ply")):
            data = PlyData.read(f)["vertex"].data
            assert 0 < len(data) <= cap
            assert all(np.isfinite(data[n]).all() for n in data.dtype.names)
            points[f.name] = len(data)
        assert len(points) == 3, "Expected two saved checkpoints and final model"
        log = (a.output/"train_process/output.log").read_text()
        match = re.search(r"Densify 100: \+clone (\d+) \+split (\d+) -prune (\d+), total (\d+)", log)
        assert match, "Densification was not exercised"
        clone, split, pruned, total = map(int, match.groups())
        assert initial+clone+split <= cap and clone+split > 0
        result["checks"]["backward_and_densification"] = {"initial": initial, "cap": cap,
            "clone": clone, "split_children": split, "pruned": pruned, "total": total, "checkpoints": points,
            "wall_s_single_functional_run": measurement["wall_s"],
            "peak_rss_mib_sampled": measurement["peak_process_tree_rss_mib_sampled"]}
        # Initial points are rejected before cache preparation; no silent downsampling.
        bad = list(cmd)
        bad[bad.index("--max-gaussians")+1] = str(initial-1)
        try:
            run(bad, a.output/"initial_cap_rejection", env, 3072, 60, 128)
        except RuntimeError:
            assert "Initial sparse points exceed strict Gaussian cap" in (a.output/"initial_cap_rejection/output.log").read_text()
        else:
            raise AssertionError("Accepted oversized initial point set")
        result["checks"]["initial_cap_rejection"] = True
        try:
            run(cmd+["--resume", a.reference_run/"splat.ply"], a.output/"resume_cap_rejection", env, 3072, 60, 128)
        except RuntimeError:
            assert "Resumed model exceeds strict Gaussian cap" in (a.output/"resume_cap_rejection/output.log").read_text()
        else:
            raise AssertionError("Oversized loaded model bypassed Gaussian cap")
        result["checks"]["resume_cap_rejection"] = True
        result["passed"] = True
    finally:
        save(a.output/"checks.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
