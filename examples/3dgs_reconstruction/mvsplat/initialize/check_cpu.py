"""Real-video host check of split preparation and warm inference; no renderer/device."""
import argparse
import json
from pathlib import Path
import time
from common import new_directory, save, sha
from initialize.model_runtime import ModelRuntime
from warm_pipeline import overlap_prepare_infer


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--vendor", type=Path, required=True)
    p.add_argument("--reference-input", type=Path, required=True)
    p.add_argument("--reference-inference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args(argv)
    out = new_directory(a.out)
    ref_infer = json.loads((a.reference_inference/"inference.json").read_text())
    if ref_infer["threads"] != a.threads:
        raise ValueError("Exact-output comparison requires matching Torch thread counts")
    runtime = ModelRuntime(a.weights, a.vendor, a.threads, 16, out/"runtime")
    from video_input.prepare import main as prepare
    from export import main as export
    inference, timings = overlap_prepare_infer(prepare,
        ["--video", str(a.video), "--out", str(out/"input"), "--threads", "1", "--size", "128"],
        runtime.infer, out/"inference")
    reference = json.loads((a.reference_input/"input.json").read_text(encoding="utf-8"))
    current = json.loads((out/"input/input.json").read_text(encoding="utf-8"))
    comparisons = dict(context_identical=sha(out/"input/context.npz")==sha(a.reference_input/"context.npz"),
                       views_identical=current["views"]==reference["views"],
                       geometry_identical=current["geometry"]==reference["geometry"],
                       gaussian_identical=inference["gaussian_sha256"]==sha(a.reference_inference/"gaussians.npz"))
    export(["--input", str(out/"input"), "--inference", str(out/"inference"), "--out", str(out/"renderer_input")])
    result = dict(complete=all(comparisons.values()), board_executed=False, comparisons=comparisons,
                  scheduling=timings, preheat_seconds_record_only=runtime.load_seconds,
                  scope="host video/pose/inference/export regression; no first-frame timing or quality measurement")
    save(out/"result.json", result)
    print(json.dumps(result), flush=True)
    if not result["complete"]:
        raise ValueError("Host warm CPU output differs from cold reference")


if __name__ == "__main__":
    main()
