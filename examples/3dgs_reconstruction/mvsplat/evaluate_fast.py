"""Evaluate a fast board run against video frames and the board CPU renderer."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from common import new_directory, save, sha
from evaluate import framebuffer, local_error, quality


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    a = p.parse_args()
    run = a.run
    pipeline = json.loads((run / "pipeline_result.json").read_text())
    if not pipeline["complete"]:
        raise ValueError("Board pipeline did not complete")
    artifacts = run / ("reconstruction" if pipeline.get("fused", False) else "")
    meta = json.loads((artifacts / "input/input.json").read_text())
    if meta["input_video_sha256"] != pipeline["input_video_sha256"]:
        raise ValueError("Input video hashes differ")
    gates = json.loads(Path(__file__).with_name("acceptance.json").read_text())
    result = dict(video_sha256=pipeline["input_video_sha256"], pose_mode=pipeline["pose_mode"],
                  focal_ratio=pipeline["focal_ratio"], calibration=meta["calibration"],
                  first_fpga_image_seconds=pipeline["video_to_first_fpga_image_seconds"],
                  gaussians=pipeline["gaussians"], stage_measurements=pipeline["stages"],
                  views={}, all_views_pass=True, scope=meta["target_scope"],
                  quality_gate=gates["reconstruction_vs_rgb"],
                  renderer_comparison="FPGA versus board CPU Dense; no independent SH reference in this fast-path report")
    panels = []
    for view in meta["views"]:
        if view["role"] != "target":
            continue
        stem = Path(view["name"]).stem
        truth = np.asarray(Image.open(artifacts / "input/images" / view["name"]).convert("RGB"),
                           dtype=np.float64) / 255
        if sha(artifacts / "input/images" / view["name"]) != view["prepared_sha256"]:
            raise ValueError("Target image hash changed")
        images = {}
        records = {}
        for backend in ("cpu_dense", "fpga"):
            matches = [r for r in pipeline["renderer_runs"] if r["camera"] == stem + ".bin"
                       and r["backend"] == backend and not r["warmup"]]
            if not matches:
                raise ValueError(f"Missing {stem} {backend} measured run")
            record = matches[0]
            path = run / "board" / record["directory"] / "frame.bin"
            if sha(path) != record["frame_sha256"]:
                raise ValueError("Board framebuffer hash changed")
            images[backend] = framebuffer(path)
            if images[backend].shape != truth.shape:
                raise ValueError("Frame shape differs from target")
            records[backend] = record
        score = quality(images["fpga"], truth)
        gate_pass = ((score["psnr_db"] is None or score["psnr_db"] >= gates["reconstruction_vs_rgb"]["minimum_psnr_db"])
                     and score["ssim_rgb_gaussian_11_sigma1p5_valid"] >= gates["reconstruction_vs_rgb"]["minimum_ssim"])
        result["views"][stem] = dict(fpga_vs_video=score,
                                      fpga_vs_cpu_dense=quality(images["fpga"], images["cpu_dense"]),
                                      local_error=local_error(images["fpga"], truth),
                                      fpga_timed_ms=records["fpga"]["total_ms"],
                                      cpu_dense_timed_ms=records["cpu_dense"]["total_ms"],
                                      fpga_timed_speedup=records["cpu_dense"]["total_ms"] / records["fpga"]["total_ms"],
                                      passed=gate_pass)
        result["all_views_pass"] &= gate_pass
        baseline = None
        if a.baseline:
            old = a.baseline / "board" / f"{stem}_fpga_0" / "frame.bin"
            baseline = framebuffer(old)
        panels.append((stem, truth, baseline, images["cpu_dense"], images["fpga"]))
    out = new_directory(a.out)
    width = 4 * 256
    canvas = Image.new("RGB", (width, len(panels) * 290 + 32), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(("Video target", "Full SfM FPGA", "Fast CPU Dense", "Fast FPGA")):
        draw.text((column * 256 + 8, 8), label, fill="black")
    for row, (stem, truth, baseline, cpu, fpga) in enumerate(panels):
        y = 32 + row * 290
        for column, image in enumerate((truth, baseline, cpu, fpga)):
            if image is not None:
                tile = Image.fromarray(np.rint(image * 255).astype(np.uint8)).resize((256, 256))
                canvas.paste(tile, (column * 256, y))
        score = result["views"][stem]["fpga_vs_video"]
        draw.text((8, y + 264), f"{stem}: FPGA {score['psnr_db']:.2f} dB, SSIM {score['ssim_rgb_gaussian_11_sigma1p5_valid']:.4f}", fill="black")
    canvas.save(out / "comparison.png")
    save(out / "quality_and_timing.json", result)
    print(json.dumps(dict(all_views_pass=result["all_views_pass"],
                          first_fpga_image_seconds=result["first_fpga_image_seconds"],
                          views={name: dict(psnr_db=value["fpga_vs_video"]["psnr_db"],
                                            ssim=value["fpga_vs_video"]["ssim_rgb_gaussian_11_sigma1p5_valid"],
                                            passed=value["passed"])
                                 for name, value in result["views"].items()}), indent=2))
    if not result["all_views_pass"]:
        raise SystemExit("Fast reconstruction quality gate failed; report and images retained")


if __name__ == "__main__":
    main()
