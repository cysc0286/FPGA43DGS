"""Evaluate held-out RGB quality, adapter loss and board renderer timing separately."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import new_directory, save

spec = importlib.util.spec_from_file_location("reconstruction_quality", Path(__file__).resolve().parents[1]/"evaluate.py")
quality_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality_module)
quality = quality_module.quality


def framebuffer(path):
    raw = path.read_bytes()
    if raw[:8] != b"GSSOUT01":
        raise ValueError("Invalid framebuffer magic")
    w, h = struct.unpack_from("<II", raw, 8)
    if len(raw) != 16+w*h*20:
        raise ValueError("Invalid framebuffer size")
    pixels = np.frombuffer(raw, dtype=[("rgba", "<f4", (4,)), ("last", "<u4")], offset=16)
    image = pixels["rgba"][:, :3].reshape(h, w, 3)
    if not np.isfinite(image).all():
        raise ValueError("Nonfinite framebuffer")
    return np.clip(image, 0, 1)


def passes(value, gate):
    return (value["psnr_db"] is None or value["psnr_db"] >= gate["minimum_psnr_db"]) and value["ssim_rgb_gaussian_11_sigma1p5_valid"] >= gate["minimum_ssim"]


def local_error(actual, target, tile=16):
    pixel = np.mean(np.abs(actual-target), axis=2)
    tiles = [(float(pixel[y:y+tile, x:x+tile].mean()), x, y)
             for y in range(0, pixel.shape[0], tile)
             for x in range(0, pixel.shape[1], tile)]
    worst, x, y = max(tiles)
    return dict(rgb_mae=float(np.mean(np.abs(actual-target))),
                pixel_rgb_mae_p95=float(np.percentile(pixel, 95)),
                worst_16x16_tile_rgb_mae=worst,
                worst_16x16_tile_xy=[x, y])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--board", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    out = new_directory(a.out)
    meta = json.loads((a.input/"input.json").read_text(encoding="utf-8"))
    board = json.loads((a.board/"board_summary.json").read_text())
    if not board["complete"]:
        raise ValueError("Board run not complete")
    gates = json.loads(Path(__file__).with_name("acceptance.json").read_text())
    result = dict(scope=meta["target_scope"], comparison_domain="Display RGB [0,1], Gaussian SSIM 11x11 sigma1.5 valid window",
                  gates=gates, gaussians=board["gaussians"], views={}, passed=True,
                  lpips="not measured", power="not measured", npu="not used",
                  timing_scope="Renderer total_ms covers timed attribute, group/sort and render subprocesses inside render.py; process_wall_ms covers the whole separately launched board command, including Python/package startup and measurement overhead. Neither includes upstream reconstruction or SSH transfer.")
    rows, panels = [], []
    for view in meta["views"]:
        if view["role"] != "target":
            continue
        stem = Path(view["name"]).stem
        gt = np.array(Image.open(a.input/"images"/view["name"]).convert("RGB"))/255.
        with np.load(a.reference/(stem+".npz")) as f:
            ref3, ref4 = np.clip(f["sh3"], 0, 1), np.clip(f["sh4"], 0, 1)
        q = dict(reference_sh4_vs_rgb=quality(ref4, gt), sh3_vs_sh4=quality(ref3, ref4))
        images = {}
        for backend in ("cpu_dense", "fpga"):
            runs = [r for r in board["runs"] if r["camera"] == stem+".bin" and r["backend"] == backend and not r["warmup"]]
            image = framebuffer(a.board/runs[0]["directory"]/"frame.bin")
            images[backend] = image
            q[backend+"_vs_rgb"] = quality(image, gt)
            q[backend+"_vs_reference_sh3"] = quality(image, ref3)
            times = np.array([r["total_ms"] for r in runs])
            measurements = [json.loads((a.board/r["directory"]/"measurement.json").read_text()) for r in runs]
            q[backend+"_timing"] = dict(repeats=len(runs), mean_ms=float(times.mean()), median_ms=float(np.median(times)),
                 min_ms=float(times.min()), max_ms=float(times.max()),
                 output_repeatability=len({r["frame_sha256"] for r in runs}) == 1,
                 peak_tree_rss_mib_sampled=max(m["peak_process_tree_rss_mib_sampled"] for m in measurements),
                 cpu_seconds_mean=float(np.mean([m["child_cpu_seconds"] for m in measurements])),
                 process_wall_ms_mean=float(np.mean([m["wall_seconds"] for m in measurements]))*1000)
            for r in runs:
                rows.append({k:r[k] for k in ("camera", "backend", "directory", "attributes_ms", "group_sort_ms", "render_ms", "total_ms", "frame_sha256")})
        q["fpga_vs_cpu_dense"] = quality(images["fpga"], images["cpu_dense"])
        q["fpga_vs_rgb_local_error"] = local_error(images["fpga"], gt)
        q["fpga_vs_reference_sh3_local_error"] = local_error(images["fpga"], ref3)
        q["board_total_speedup_cpu_dense_over_fpga"] = q["cpu_dense_timing"]["mean_ms"]/q["fpga_timing"]["mean_ms"]
        q["board_process_wall_speedup_cpu_dense_over_fpga"] = (q["cpu_dense_timing"]["process_wall_ms_mean"]/
                                                               q["fpga_timing"]["process_wall_ms_mean"])
        q["passed"] = (passes(q["fpga_vs_rgb"], gates["reconstruction_vs_rgb"]) and
             passes(q["fpga_vs_reference_sh3"], gates["fpga_vs_world_covariance_sh3_reference"]) and
             passes(q["sh3_vs_sh4"], gates["sh3_vs_sh4_reference"]) and
             q["cpu_dense_timing"]["output_repeatability"] and q["fpga_timing"]["output_repeatability"])
        result["passed"] &= q["passed"]
        result["views"][stem] = q
        panels.append((stem, [gt, ref4, images["cpu_dense"], images["fpga"]]))
    size = max(256, meta["views"][0]["width"])
    canvas = Image.new("RGB", (4*size, len(panels)*(size+44)+36), "#f5f5f5")
    draw = ImageDraw.Draw(canvas)
    for col, label in enumerate(("Held-out video frame", "MVSplat CPU reference / SH4", "Board CPU Dense / SH3", "Board FPGA / SH3")):
        draw.text((col*size+8, 10), label, fill="#111111")
    for row, (stem, imgs) in enumerate(panels):
        y = 36+row*(size+44)
        for col, image in enumerate(imgs):
            tile = Image.fromarray(np.rint(image*255).astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST)
            canvas.paste(tile, (col*size, y))
        q = result["views"][stem]
        label = f"{stem} | FPGA vs video: {q['fpga_vs_rgb']['psnr_db']:.2f} dB / SSIM {q['fpga_vs_rgb']['ssim_rgb_gaussian_11_sigma1p5_valid']:.4f} | timed renderer {q['board_total_speedup_cpu_dense_over_fpga']:.3f}x, process {q['board_process_wall_speedup_cpu_dense_over_fpga']:.3f}x"
        draw.text((8, y+size+12), label, fill="#111111")
    canvas.save(out/"comparison.png")
    save(out/"quality_and_timing.json", result)
    with (out/"timing.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"passed": result["passed"], "views": {k:dict(psnr=v["fpga_vs_rgb"]["psnr_db"], ssim=v["fpga_vs_rgb"]["ssim_rgb_gaussian_11_sigma1p5_valid"], timed_renderer_speedup=v["board_total_speedup_cpu_dense_over_fpga"], process_wall_speedup=v["board_process_wall_speedup_cpu_dense_over_fpga"]) for k,v in result["views"].items()}}, indent=2))
    if not result["passed"]:
        raise SystemExit("Bridge acceptance failed; retain evidence and investigate")


if __name__ == "__main__":
    main()
