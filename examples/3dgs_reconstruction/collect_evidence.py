"""Archive compact evidence without copying dependencies, video, or model binaries."""
import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    args = ap.parse_args()
    registry = ROOT/"evidence/versions.csv"
    old_rows = []
    if registry.exists():
        with registry.open(newline="", encoding="utf-8") as f:
            old_rows = list(csv.DictReader(f))
    names = [r.name for r in args.runs]
    if len(set(names)) != len(names) or set(names) & {r["run"] for r in old_rows}:
        raise ValueError("Duplicate run name; version records are append-only")
    for r in args.runs:
        if (ROOT/"evidence"/r.name).exists():
            raise ValueError("Evidence archive already exists: " + r.name)
    rows = []
    for run in args.runs:
        archive = ROOT/"evidence"/run.name
        archive.mkdir(exist_ok=False)
        for f in run.glob("*.json"):
            shutil.copy2(f, archive/f.name)
        for name in ("train.log", "sfm.log", "comparison.png"):
            shutil.copy2(run/name, archive/name)
        shutil.copy2(run/"renderer_input/manifest.json", archive/"renderer_manifest.json")
        if (run/"board_cpu").exists():
            shutil.copytree(run/"board_cpu", archive/"board_cpu")
            novel = cv2.imread(str(run/"board_cpu/novel_midpoint/frame.ppm"))
            cv2.imwrite(str(archive/"novel_view.png"), novel)
        sfm = json.loads((run/"sfm.json").read_text())
        quality = json.loads((run/"quality.json").read_text())
        model = json.loads((run/"renderer_input/manifest.json").read_text())
        measures = {s: json.loads((run/(s+"_measurement.json")).read_text()) for s in ("frames", "sfm", "train", "export")}
        (archive/"pipeline_measurement.json").write_text(json.dumps({"stage_wall_sum_s": sum(v['wall_s'] for v in measures.values()),
            "peak_stage_rss_mib": max(v['peak_process_tree_rss_mib_sampled'] for v in measures.values()),
            "stages": measures, "boundary": "subprocess stage sum, excludes setup/download/build/evaluation/board rendering"}, indent=2))
        log = (run/"train.log").read_text()
        assert "Using CPU" in log and "Using CUDA" not in log and not sfm["has_cuda"]
        row = {"run": run.name, "registered": sfm["registered"], "initial_points": sfm["points3D"],
               "reprojection_px": sfm["mean_reprojection_error_px"], "gaussians": model["gaussians"],
               "sfm_s": measures["sfm"]["wall_s"], "train_s": measures["train"]["wall_s"],
               "stage_sum_s": sum(v['wall_s'] for v in measures.values()),
               "peak_rss_mib": max(v['peak_process_tree_rss_mib_sampled'] for v in measures.values()),
               "native_float_psnr_db": float(re.search(r'validation PSNR: ([\d.]+)', log).group(1)),
               "native_png_psnr_db": quality["native_cpu_vs_heldout"]["psnr_db"],
               "native_png_ssim": quality["native_cpu_vs_heldout"]["ssim_rgb_gaussian_11_sigma1p5_valid"],
               "model_sha256": model["model_sha256"]}
        rows.append(row)
        files = {str(f.relative_to(archive)).replace('\\','/'): hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in archive.rglob('*') if f.is_file()}
        (archive/"sha256.json").write_text(json.dumps(files, indent=2))
    temporary = registry.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(old_rows)
        w.writerows(rows)
    temporary.replace(registry)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
