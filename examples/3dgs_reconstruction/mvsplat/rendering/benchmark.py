"""Board-only paired live renderer timing and post-timing image comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from initialize.session import render_environment
from rendering.runtime import LiveRenderer


def summarize(records):
    times = np.array([r["wall_ms"] for r in records])
    return dict(n=len(times), mean_ms=float(times.mean()), median_ms=float(np.median(times)),
                p95_ms=float(np.percentile(times, 95)), max_ms=float(times.max()),
                stages_ms={key:statistics.mean(r[key] for r in records) for key in
                           ("project_ms", "sort_ms", "engine_ms", "pack_ms", "upload_ms", "wait_ms", "read_ms")})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True, help="Prior frozen FPGA validation.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--profiles", default="sort_compare,exact4,batch4,batch2,batch1,keep24k,keep16k,cpu4")
    p.add_argument("--repeats", type=int, default=6)
    a = p.parse_args()
    if platform.machine() != "aarch64" or not 1 <= a.repeats <= 100:
        p.error("Real ARM board and bounded repeat count required")
    a.out.mkdir(parents=True, exist_ok=False)
    meta = json.loads((a.scene/"manifest.json").read_text())
    cameras = [c["file"] for c in meta["cameras"] if c["role"] == "target"]
    references = {r["camera"]:r for r in json.loads(a.reference.read_text())["resident"]}
    configs = dict(uncached1=dict(threads=1, cached=False), exact1=dict(threads=1),
                   exact4=dict(threads=4), batch16=dict(threads=4, batch=16),
                   batch32=dict(threads=4, batch=32), keep24k=dict(threads=4, max_gaussians=24576),
                   keep16k=dict(threads=4, max_gaussians=16384), cpu4=dict(threads=4, cpu=True),
                   sort_compare=dict(threads=4,comparison_sort=True),batch4=dict(threads=4,batch=4),
                   batch2=dict(threads=4,batch=2),batch1=dict(threads=4,batch=1),
                   preview24k=dict(threads=4,batch=2,max_gaussians=24576,uniform_preview=True),
                   preview16k=dict(threads=4,batch=2,max_gaussians=16384,uniform_preview=True),
                   preview8k=dict(threads=4,batch=2,max_gaussians=8192,uniform_preview=True))
    for config in configs.values():
        config.setdefault("batch",8)  # Preserve the named historical experiment.
    result = dict(board_executed=True, scene_sha256=hashlib.sha256((a.scene/"model.ply").read_bytes()).hexdigest(),
                  scope="loaded scene camera-to-RGB; no video preparation or display latency", profiles={})
    exact_images={}
    for name in a.profiles.split(","):
        directory=a.out/name
        directory.mkdir()
        entry=dict(configuration=configs[name], measurements=[], checks=[])
        timed_frames=[]
        with LiveRenderer(a.binary, render_environment(), directory/"native.log", **configs[name]) as runtime:
            entry["scene"]=runtime.load_scene(a.scene/"model.ply")
            # First render is reported independently, never silently discarded.
            f=runtime.render_camera((a.scene/cameras[0]).read_bytes())
            entry["first_frame"]=f.metadata
            for i,cam in enumerate(cameras*a.repeats):
                f=runtime.render_camera((a.scene/cam).read_bytes())
                entry["measurements"].append(dict(f.metadata,camera=cam))
                timed_frames.append((cam,f.rgb))
            # Expensive diagnostics run AFTER the timed loop.
            for cam in cameras:
                f=runtime.render_camera((a.scene/cam).read_bytes(),include_raw=True)
                check=f.archive(directory/Path(cam).stem)
                check["camera"]=cam
                check["matches_frozen_frame"]=check["frame_sha256"]==references[cam]["frame_sha256"]
                expected_ppm=references[cam].get("ppm_sha256")
                check["matches_frozen_rgb"]=expected_ppm==hashlib.sha256(
                    (directory/Path(cam).stem/"frame.ppm").read_bytes()).hexdigest() if expected_ppm else None
                if name=="uncached1" or (cam not in exact_images and not configs[name].get("max_gaussians") and not configs[name].get("cpu")):
                    exact_images[cam]=f.rgb.copy()
                if cam in exact_images:
                    diff=f.rgb.astype(np.float32)-exact_images[cam]
                    mse=float(np.square(diff).mean())
                    check["psnr_vs_exact_db"]=None if mse==0 else float(10*np.log10(255**2/mse))
                    check["max_rgb_error"]=int(np.abs(diff).max())
                    check["mean_rgb_error"]=float(np.abs(diff).mean())
                entry["checks"].append(check)
            frame_hashes={c["camera"]:c["rgb_sha256"] for c in entry["checks"]}
            entry["all_timed_frames_match_audited_view"]=all(
                hashlib.sha256(rgb.tobytes()).hexdigest()==frame_hashes[cam] for cam,rgb in timed_frames)
        entry["summary"]=summarize(entry["measurements"])
        result["profiles"][name]=entry
        (a.out/"results.json").write_text(json.dumps(result,indent=2))
        if not entry["all_timed_frames_match_audited_view"]:
            raise ValueError("A timed frame differs from its audited view")
        if not configs[name].get("max_gaussians") and not configs[name].get("cpu"):
            if not all(c["matches_frozen_frame"] and c["matches_frozen_rgb"] for c in entry["checks"]):
                raise ValueError("Full-point candidate differs from frozen FPGA")
        print(json.dumps(dict(profile=name,summary=entry["summary"],
              exact=[c["matches_frozen_frame"] for c in entry["checks"]],
              psnr=[c.get("psnr_vs_exact_db") for c in entry["checks"]])),flush=True)
    result["complete"]=True
    (a.out/"results.json").write_text(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
