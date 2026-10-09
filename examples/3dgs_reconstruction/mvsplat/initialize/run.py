"""Warm resources before video input, then prepare and render one scene."""
import argparse
import json
import os
from pathlib import Path
import platform
from common import new_directory, save, sha
from initialize.events import EventLog


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, help="Completed local video; omit to supply path after READY")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--vendor", type=Path, required=True)
    p.add_argument("--renderer", type=Path, required=True)
    p.add_argument("--resident-binary", type=Path, default=Path(__file__).with_name("render_resident"))
    p.add_argument("--live-renderer", type=Path,
                   help="Native camera-to-frame backend; replaces the three-process renderer")
    p.add_argument("--render-profile", choices=("mainline", "custom"), default="mainline",
                   help="Native renderer defaults to the accepted full-scene mainline; custom allows tuning")
    p.add_argument("--render-threads", type=int, choices=(1, 2, 3, 4))
    p.add_argument("--render-max-gaussians", type=int,
                   help="Explicit lossy scene budget (custom profile); 0 retains every Gaussian")
    p.add_argument("--render-batch", type=int, choices=(1, 2, 4, 8, 16, 32))
    p.add_argument("--render-uniform-preview", action="store_true", default=None,
                   help="Uniform thinning with enlarged footprint; requires a Gaussian budget")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--threads", type=int, default=4, help="Torch intra-op threads; measured 30TAI CPU default")
    p.add_argument("--prepare-threads", type=int, default=1, help="OpenCV/decoder budget")
    schedule = p.add_mutually_exclusive_group()
    schedule.add_argument("--serial-prepare", dest="serial_prepare", action="store_true",
                          help="Complete poses before inference (measured default)")
    schedule.add_argument("--overlap-prepare", dest="serial_prepare", action="store_false",
                          help="Overlap PnP with inference; use --threads 3 on the four-core board")
    p.set_defaults(serial_prepare=True)
    p.add_argument("--focal-ratio", type=float, default=0.9)
    p.add_argument("--backend", choices=("cpu", "onnx_reference", "npu"), default="cpu")
    p.add_argument("--partition-bundle", type=Path)
    p.add_argument("--partitions", nargs="+")
    p.add_argument("--npu-library", type=Path)
    p.add_argument("--oracle-graphs", type=Path,
                   help="Real-input export oracles; NPU default is graphs beside compiled bundle")
    p.add_argument("--worker-python", type=Path)
    p.add_argument("--buffer-policy", choices=("shared", "per_partition"), default="shared")
    p.add_argument("--buffer-limit-mib", type=int, default=128)
    a = p.parse_args(argv)
    if a.render_max_gaussians is not None and not 0 <= a.render_max_gaussians <= 1000000:
        p.error("--render-max-gaussians must be in [0, 1000000]")
    if a.render_max_gaussians and a.live_renderer is None:
        p.error("--render-max-gaussians requires --live-renderer")
    if a.render_uniform_preview and not a.render_max_gaussians:
        p.error("--render-uniform-preview requires --render-max-gaussians")
    if a.live_renderer is not None:
        from rendering.pipeline_adapter import validate_pipeline_configuration
        configuration = {key: getattr(a, "render_" + key)
                         for key in ("threads", "max_gaussians", "batch", "uniform_preview")}
        try:
            validate_pipeline_configuration(a.render_profile,
                {key: value for key, value in configuration.items() if value is not None})
        except ValueError as exc:
            p.error(str(exc))
    if platform.machine().lower() not in ("aarch64", "arm64"):
        p.error("First FPGA frame acceptance requires the ARM board")
    if min(a.threads, a.prepare_threads) < 1 or (not a.serial_prepare and
            a.threads+a.prepare_threads > (os.cpu_count() or 4)):
        p.error("Overlapping inference/preparation must fit the CPU thread budget")
    if a.backend != "cpu" and (a.partition_bundle is None or not a.partitions):
        p.error("Non-CPU backend needs --partition-bundle and explicit --partitions")
    out = new_directory(a.out)
    events = EventLog(out / "events.json")
    record = dict(complete=False, mode="warm_fast_pair", backend=a.backend,
                  result_scope="one target frame; not full video coverage")
    session = None
    try:
        from initialize.session import WarmSession
        session = WarmSession(a, out)
        record["initialize"] = session.record
        event = events.emit("READY", backend=a.backend, weight_sha256=sha(a.weights),
                            video_processed=False)
        print(json.dumps(event), flush=True)
        if a.video is None:
            a.video = Path(input().strip())
        from video_input.receipt import receive_file
        receipt = receive_file(a.video, events)
        record["video_receipt"] = receipt
        from warm_pipeline import prepare_scene
        record.update(prepare_scene(receipt["path"], out, session.model, session.renderer, events,
            size=a.size, prepare_threads=a.prepare_threads, focal_ratio=a.focal_ratio,
            overlap=not a.serial_prepare))
        record["complete"] = True
    except BaseException as exc:
        record["error"] = type(exc).__name__+": "+str(exc)
        raise
    finally:
        record.update(events=events.events, timing=events.summary())
        save(out / "warm_result.json", record)
        if session is not None:
            try:
                session.close()
            except Exception as exc:
                record["cleanup_error"] = type(exc).__name__+": "+str(exc)
                save(out / "warm_result.json", record)
                if "error" not in record:
                    raise
    print(json.dumps(record["timing"], indent=2), flush=True)


if __name__ == "__main__":
    main()
