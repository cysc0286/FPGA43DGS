"""Reconstruct on the ARM board with MVSplat, or run individual legacy stages."""
import argparse
import dataclasses
import json
from pathlib import Path
import sys

from modules.contracts import FrameSet, PoseSet, GaussianScene, RenderInput, RenderResult


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "reconstruct":
        from mvsplat.board_pipeline import main as reconstruct
        return reconstruct(argv[1:])
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="stage", required=True)
    commands.add_parser("reconstruct",
                        help="Board video-to-image path using pretrained MVSplat (default: full SfM)")
    for name in ("video", "pose", "gaussian", "export", "render", "validate"):
        s = commands.add_parser(name, help="Legacy OpenSplat stage" if name == "gaussian" else None)
        s.add_argument("--run", required=True, type=Path)
        if name == "video":
            s.add_argument("--video", required=True, type=Path)
            s.add_argument("--frames", type=int, default=30)
            s.add_argument("--width", type=int, default=480)
            s.add_argument("--threads", type=int, default=2)
        if name == "pose":
            s.add_argument("--width", type=int, default=480)
            s.add_argument("--render-width", type=int, default=160)
            s.add_argument("--threads", type=int, default=2)
            s.add_argument("--max-features", type=int, default=2048)
            s.add_argument("--overlap", type=int, default=10)
        if name == "gaussian":
            s.add_argument("--opensplat", required=True, type=Path)
            s.add_argument("--profile", choices=("arm_candidate", "cpu_control"), default="arm_candidate")
            s.add_argument("--plan", action="store_true")
        if name == "render":
            s.add_argument("--renderer", required=True, type=Path)
            s.add_argument("--out", required=True, type=Path)
            s.add_argument("--backend", choices=("fpga", "cpu_dense", "cpu_base"), default="fpga")
            s.add_argument("--camera", default="novel_midpoint.bin")
            s.add_argument("--work-root", default="/dev/shm")
            s.add_argument("--plan", action="store_true")
        if name == "validate":
            scope = s.add_mutually_exclusive_group()
            scope.add_argument("--through", choices=("video", "pose", "gaussian", "export", "render"),
                               help="Validate all artifacts through this stage (default: export)")
            scope.add_argument("--only", choices=("video", "pose", "gaussian", "export", "render"),
                               help="Validate one handoff without requiring upstream artifacts")
            s.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    a.run = a.run.resolve()
    for k in ("frames", "width", "render_width", "threads", "max_features", "overlap"):
        if hasattr(a, k) and getattr(a, k) <= 0:
            p.error(k + " must be positive")
    if a.stage == "video":
        if a.frames < 5:
            p.error("Need at least five frames for SfM")
        if (a.run / "frames").exists() or (a.run / "frames.json").exists():
            p.error("Frame outputs exist; use a fresh run")
        import cv2
        from modules.video_input import frames
        cv2.setNumThreads(a.threads)
        a.run.mkdir(parents=True, exist_ok=True)
        result = frames(a)
    elif a.stage == "pose":
        if any((a.run / n).exists() for n in ("database.db", "sparse", "project", "undistorted", "sfm.json", "project.json")):
            p.error("Pose outputs exist; use a fresh run")
        if a.render_width > a.width:
            p.error("Training images must not enlarge the SfM images")
        import cv2
        from modules.pose_estimation import sfm
        cv2.setNumThreads(a.threads)
        result = sfm(a)
    elif a.stage == "gaussian":
        from modules.gaussian_generation.training import train
        result = train(a.run, a.opensplat, a.profile, a.plan)
    elif a.stage == "export":
        from modules.gaussian_generation import export
        result = export(a)
    elif a.stage == "render":
        from modules.rendering import render
        result = render(a.run / "renderer_input", a.renderer, a.out, a.camera, a.backend, a.work_root, a.plan)
    else:
        order = ("video", "pose", "gaussian", "export", "render")
        checks = []
        selected = (a.only,) if a.only else order[:order.index(a.through or "export")+1]
        for name in selected:
            if name == "render" and a.out is None:
                p.error("--out is required to validate a render result")
            value = {"video": lambda: FrameSet.load(a.run), "pose": lambda: PoseSet.load(a.run),
                     "gaussian": lambda: GaussianScene.load(a.run),
                     "export": lambda: RenderInput.load(a.run / "renderer_input"),
                     "render": lambda: RenderResult.load(a.out)}[name]()
            checks.append({"stage": name, "status": "passed", "artifact": dataclasses.asdict(value)})
        result = {"checks": checks, "scope": "artifact interface validation, not a new board execution"}
    print(json.dumps(dataclasses.asdict(result) if dataclasses.is_dataclass(result) else result, indent=2, default=str))


if __name__ == "__main__":
    main()
