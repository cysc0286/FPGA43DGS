"""Compatibility entry point; implementations live in modules/."""
import argparse
from pathlib import Path
from modules.common import file_sha256, save


def frames(a):
    from modules.video_input import frames as execute
    return execute(a)


def sfm(a):
    from modules.pose_estimation import sfm as execute
    return execute(a)


def export(a):
    from modules.gaussian_generation import export as execute
    return execute(a)


def validate_adapter(a):
    from modules.gaussian_generation import validate_adapter as execute
    return execute(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["frames", "sfm", "export", "validate_adapter"])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--render-width", type=int, default=320)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-features", type=int, default=4096)
    ap.add_argument("--overlap", type=int, default=10)
    a = ap.parse_args()
    if min(a.frames, a.width, a.render_width, a.threads, a.max_features, a.overlap) <= 0:
        ap.error("Counts, widths and thread settings must be positive")
    import cv2
    cv2.setNumThreads(a.threads)
    globals()[a.stage](a)


if __name__ == "__main__":
    main()
