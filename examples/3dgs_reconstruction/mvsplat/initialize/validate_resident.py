"""Exercise real board view changes and compare against frozen CPU/FPGA outputs."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import time

from common import new_directory, save, sha
from initialize.renderer_runtime import RendererRuntime
from initialize.session import render_environment


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--renderer", type=Path, required=True)
    p.add_argument("--binary", type=Path, default=Path(__file__).with_name("render_resident"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if platform.machine().lower() not in ("aarch64", "arm64"):
        p.error("Real ARM board required")
    out = new_directory(a.out.resolve())
    scene, renderer = a.scene.resolve(), a.renderer.resolve()
    meta = json.loads((scene/"manifest.json").read_text())
    model = scene/"model.ply"
    if sha(model) != meta["model_sha256"]:
        raise ValueError("Scene checksum changed")
    cameras = [c["file"] for c in meta["cameras"] if c["role"] == "target"]
    if len(cameras) < 2:
        raise ValueError("At least two different target views required")
    result = dict(complete=False, model_sha256=sha(model), board_executed=True,
                  resident=[], frozen=[], scope="camera-to-frame checks; not video-to-frame timing")
    env = render_environment()
    try:
        with RendererRuntime(renderer, a.binary.resolve(), env, out/"resident.log") as runtime:
            result["scene"] = runtime.load_scene(model)
            for index, camera in enumerate(cameras + cameras[:1]):
                directory = "resident_"+str(index)
                start = time.monotonic()
                record = runtime.render_camera((scene/camera).read_bytes(), out/directory)
                result["resident"].append(dict(record, camera=camera, directory=directory,
                    command_to_frame_seconds=time.monotonic()-start,
                    ppm_sha256=sha(out/directory/"frame.ppm")))
        result["returned_view_same_hash"] = (result["resident"][0]["frame_sha256"] ==
                                              result["resident"][-1]["frame_sha256"])
        # Both frozen backends have identical process/package startup boundaries.
        # Do not compare their wall time against resident time as pure hardware gain.
        for camera in cameras:
            for backend in ("cpu_dense", "fpga"):
                directory = Path(camera).stem+"_"+backend
                start = time.monotonic()
                with (out/(directory+".log")).open("wb") as log:
                    subprocess.run(["/usr/bin/python3", str(renderer/"render.py"), "--model", str(model),
                        "--camera", str(scene/camera), "--out", str(out/directory), "--backend", backend],
                        check=True, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=240)
                record = json.loads((out/directory/"result.json").read_text())
                result["frozen"].append(dict(record, camera=camera, directory=directory,
                    process_wall_seconds=time.monotonic()-start,
                    ppm_sha256=sha(out/directory/"frame.ppm")))
        result["matches_frozen_fpga"] = all(
            all(r[key] == next(f[key] for f in result["frozen"]
                if f["camera"] == r["camera"] and f["backend"] == "fpga")
                for key in ("frame_sha256", "ppm_sha256")) for r in result["resident"])
        result["complete"] = result["returned_view_same_hash"] and result["matches_frozen_fpga"]
        if not result["complete"]:
            raise ValueError("Resident output differs after view change or from frozen FPGA")
    except BaseException as exc:
        result["error"] = type(exc).__name__+": "+str(exc)
        raise
    finally:
        save(out/"validation.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
