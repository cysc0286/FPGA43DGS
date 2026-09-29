"""Run existing fast geometry, encoder and adapter in one bounded CPU process."""
import argparse
from pathlib import Path
import time

from common import new_directory, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--vendor", type=Path, required=True)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--focal-ratio", type=float, default=0.9)
    a = p.parse_args()
    out = new_directory(a.out)
    start = time.perf_counter()
    result = dict(complete=False, stages=[], execution="One process; same geometry, weights and adapter; no training")
    try:
        # On ARM, PyTorch's OpenMP TLS must be reserved before OpenCV is loaded.
        import torch  # noqa: F401
        t = time.perf_counter()
        from prepare_fast import main as prepare
        prepare(["--video", str(a.video), "--out", str(out/"input"), "--size", str(a.size),
                 "--threads", str(a.threads), "--focal-ratio", str(a.focal_ratio)])
        result["stages"].append(dict(name="prepare_fast", wall_seconds=time.perf_counter()-t))
        t = time.perf_counter()
        from infer import main as infer
        infer(["--input", str(out/"input"), "--weights", str(a.weights), "--vendor", str(a.vendor),
               "--out", str(out/"inference"), "--threads", str(a.threads), "--depth-chunk", "16"])
        result["stages"].append(dict(name="infer", wall_seconds=time.perf_counter()-t))
        t = time.perf_counter()
        from export import main as export
        export(["--input", str(out/"input"), "--inference", str(out/"inference"),
                "--out", str(out/"renderer_input")])
        result["stages"].append(dict(name="export", wall_seconds=time.perf_counter()-t))
        result["complete"] = True
    finally:
        result["wall_seconds"] = time.perf_counter()-start
        save(out/"reconstruction.json", result)


if __name__ == "__main__":
    main()
