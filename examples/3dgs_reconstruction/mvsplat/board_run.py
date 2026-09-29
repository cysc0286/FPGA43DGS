"""Upload a new bridge result and measure the unchanged ARM/FPGA renderer."""
import argparse
import json
from pathlib import Path, PurePosixPath
import shlex
import sys
import time

from common import new_directory, save, sha

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "3dgs_compositor/board"))
from modules.contracts import RenderInput, RenderResult
import remote


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--remote-root", required=True)
    p.add_argument("--renderer", default="/root/fpga43dgs_releases/20260928T004334/3dgs_renderer_v1_20260928")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--regression", action="store_true")
    a = p.parse_args()
    dest = PurePosixPath(a.remote_root)
    if not str(dest).startswith("/root/fpga43dgs_reconstruction/") or ".." in dest.parts or a.repeats < 1:
        p.error("Use a fresh directory under the board reconstruction workspace")
    manifest = json.loads((a.input / "manifest.json").read_text(encoding="utf-8"))
    cameras = [c for c in manifest["cameras"] if c["role"] == "target"]
    for camera in cameras:
        RenderInput.load(a.input, camera["file"])
    out = new_directory(a.out)
    record = dict(complete=False, renderer=a.renderer, host=remote.HOST, remote_root=str(dest),
                  model_sha256=sha(a.input / "model.ply"), gaussians=manifest["gaussians"], runs=[])
    client = remote.connect()
    sftp = client.open_sftp()
    try:
        remote.run(client, "mkdir " + shlex.quote(str(dest)), timeout=15)
        measurement_script = Path(__file__).with_name("measure.py")
        sftp.put(str(measurement_script), str(dest / "measure.py"))
        _, check = remote.run(client, "sha256sum " + shlex.quote(str(dest / "measure.py")), timeout=15)
        if check.split()[0] != sha(measurement_script):
            raise ValueError("Measurement script hash mismatch")
        _, identity = remote.run(client, "cat /proc/sys/kernel/random/boot_id\nuname -a\ncat /proc/meminfo", timeout=15)
        (out / "inventory.txt").write_text(identity, encoding="utf-8")
        if a.regression:
            target = dest / "frozen_regression"
            cmd = shlex.join(["python3", a.renderer+"/render.py", "--out", str(target), "--backend", "fpga"])
            remote.run(client, cmd, timeout=360, log=out / "regression.log")
            local = new_directory(out / "regression")
            for name in ("result.json", "frame.bin", "frame.ppm", "frame_timing.csv"):
                sftp.get(str(target/name), str(local/name))
            RenderResult.load(local)
            expected = ROOT.parents[1] / "releases/3dgs_renderer_v1_20260928/reference/v0_fpga.bin"
            record["frozen_fpga_regression_bitwise_equal"] = sha(local/"frame.bin") == sha(expected)
            if not record["frozen_fpga_regression_bitwise_equal"]:
                raise ValueError("Frozen FPGA output changed; stop new model testing")
            print("Frozen FPGA regression: byte exact", flush=True)
        remote.run(client, "mkdir " + shlex.quote(str(dest / "input")), timeout=15)
        files = [a.input / "model.ply", a.input / "manifest.json"] + [a.input / c["file"] for c in cameras]
        t = time.perf_counter()
        for path in files:
            target = dest / "input" / path.name
            sftp.put(str(path), str(target))
            _, check = remote.run(client, "sha256sum " + shlex.quote(str(target)), timeout=20)
            if check.split()[0] != sha(path):
                raise ValueError("Uploaded file hash mismatch")
        record["upload_seconds_including_hash_checks"] = time.perf_counter()-t
        record["uploaded_file_bytes"] = sum(f.stat().st_size for f in files)
        record["downloaded_file_bytes"] = 0
        record["download_seconds"] = 0
        for camera in cameras:
            stem = Path(camera["file"]).stem
            for backend in ("cpu_dense", "fpga"):
                # Warm-up is separate and retained. All timing repetitions remain visible.
                for repeat in range(-1, a.repeats):
                    name = f"{stem}_{backend}_" + ("warmup" if repeat == -1 else str(repeat))
                    target = dest / name
                    command = ["python3", str(dest / "measure.py"), "--out", str(dest / (name+".measurement.json")), "--",
                               "python3", a.renderer+"/render.py", "--model", str(dest/"input/model.ply"),
                               "--camera", str(dest/"input"/camera["file"]), "--out", str(target), "--backend", backend]
                    remote.run(client, shlex.join(command), timeout=360, log=out/(name+".log"))
                    local = new_directory(out / name)
                    t = time.perf_counter()
                    for file in ("result.json", "frame.bin", "frame.ppm", "frame_timing.csv", "run.log"):
                        sftp.get(str(target/file), str(local/file))
                        record["downloaded_file_bytes"] += (local/file).stat().st_size
                    sftp.get(str(dest/(name+".measurement.json")), str(local/"measurement.json"))
                    record["download_seconds"] += time.perf_counter()-t
                    RenderResult.load(local)
                    value = json.loads((local/"result.json").read_text())
                    if value["model_sha256"] != record["model_sha256"] or value["camera_sha256"] != sha(a.input/camera["file"]):
                        raise ValueError("Board input provenance mismatch")
                    record["runs"].append(dict(directory=name, camera=camera["file"], backend=backend,
                                               warmup=repeat == -1, **{k:v for k,v in value.items() if k != "backend"}))
                    save(out/"board_summary.json", record)
                    print(name, "total_ms", value["total_ms"], flush=True)
        record["complete"] = True
    finally:
        save(out / "board_summary.json", record)
        sftp.close()
        client.close()


if __name__ == "__main__":
    main()
