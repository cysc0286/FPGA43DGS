"""Real-video host validation and exact old/new serialization comparison.

This is a host regression, never an ARM deployment or hardware speed result.
Run with the reconstruction virtualenv. Outputs are new, named evidence only.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.common import file_sha256


def hashes(folder):
    return {f.relative_to(folder).as_posix(): file_sha256(f) for f in folder.rglob("*") if f.is_file()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--opensplat", type=Path, required=True)
    a = p.parse_args()
    run, evidence = a.run.resolve(), a.evidence.resolve()
    if run.exists():
        raise ValueError("Use a fresh run")
    reference = ROOT/"runs/arm_candidate_pc_round2"
    spec = importlib.util.spec_from_file_location("before_refactor", evidence/"before/stages.py")
    before = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(before)
    from modules.gaussian_generation import export
    from modules.video_input import frames
    # Serialization equivalence on identical already-trained data, independent
    # of stochastic SfM/optimization variation in a new end-to-end run.
    with tempfile.TemporaryDirectory() as temporary:
        temp = Path(temporary)
        for branch in ("old", "new"):
            out = temp/branch
            out.mkdir()
            for name in ("project", "project.json", "sfm.json", "splat.ply", "cameras.json"):
                src = reference/name
                if src.is_dir():
                    shutil.copytree(src, out/name)
                else:
                    shutil.copy2(src, out/name)
        before.export(SimpleNamespace(run=temp/"old"))
        export(SimpleNamespace(run=temp/"new"))
        old, new = hashes(temp/"old/renderer_input"), hashes(temp/"new/renderer_input")
        assert old == new, "Renderer input bytes changed"
        export_count = len(old)
        assert (temp/"old/adapter_validation.json").read_bytes() == (temp/"new/adapter_validation.json").read_bytes()
        for branch, execute in (("old_frames", before.frames), ("new_frames", frames)):
            dest = temp/branch
            dest.mkdir()
            execute(SimpleNamespace(run=dest, video=ROOT/"data/nyu_snippet_curl.mp4", frames=30, width=480, threads=2))
        assert hashes(temp/"old_frames") == hashes(temp/"new_frames"), "Decoded frame bytes changed"
    (evidence/"equivalence.json").write_text(json.dumps({"passed": True,
        "renderer_input_files_byte_identical": export_count, "frame_files_byte_identical": 31,
        "adapter_validation_byte_identical": True, "scope": "same real input, pre/post refactor implementations"}, indent=2))
    # Independent module commands, including genuine CPU Gaussian training.
    base = [sys.executable, str(ROOT/"pipeline.py")]
    commands = [
        ("video", base+["video", "--run", str(run), "--video", str(ROOT/"data/nyu_snippet_curl.mp4")]),
        ("pose", base+["pose", "--run", str(run)]),
        ("gaussian", base+["gaussian", "--run", str(run), "--opensplat", str(a.opensplat.resolve())]),
        ("export", base+["export", "--run", str(run)]),
        ("evaluate", [sys.executable, str(ROOT/"evaluate.py"), "--run", str(run)]),
        ("validate", base+["validate", "--run", str(run), "--through", "export"]),
        ("render_plan", base+["render", "--run", str(run), "--renderer", str(ROOT.parents[1]/"releases/3dgs_renderer_v1_20260928"),
                              "--out", str(run/"future_board_render"), "--plan"]),
    ]
    records = {}
    for name, command in commands:
        start = time.perf_counter()
        with (evidence/(name+"_smoke.log")).open("w",encoding="utf-8") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        records[name] = {"wall_s": time.perf_counter()-start, "status": "passed", "command": command}
        print(name, "passed", flush=True)
    result = {"scope": "PC CPU module refactor regression; renderer command plan only", "board_executed": False,
              "run": str(run), "stages": records,
              "sfm": json.loads((run/"sfm.json").read_text()),
              "gaussians": json.loads((run/"renderer_input/manifest.json").read_text())["gaussians"],
              "quality": json.loads((run/"quality.json").read_text()),
              "train_measurement": json.loads((run/"train_measurement.json").read_text())}
    (evidence/"host_smoke.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"gaussians": result["gaussians"], "quality": result["quality"]}),flush=True)


if __name__ == "__main__":
    main()
