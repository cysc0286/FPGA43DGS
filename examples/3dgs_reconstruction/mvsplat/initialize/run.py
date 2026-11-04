"""Warm resources before video input, then prepare and render one scene."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
from common import new_directory, save, sha
from initialize.events import EventLog


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--video", type=Path, help="Completed local video; omit to supply path after READY")
    source.add_argument("--camera", type=Path, help="Linux UVC capture node, e.g. /dev/v4l/by-id/...")
    p.add_argument("--calibration", type=Path)
    p.add_argument("--capture-seconds", type=float, default=5.)
    p.add_argument("--camera-width", type=int, default=1280)
    p.add_argument("--camera-height", type=int, default=720)
    p.add_argument("--camera-fps", type=float, default=30.)
    p.add_argument("--capture-settle", type=float, default=1.)
    p.add_argument("--capture-now", action="store_true", help="Start after READY without waiting for Enter")
    p.add_argument("--auto-select", action="store_true", help="Always enabled for camera input")
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--min-parallax", type=float, default=1.)
    p.add_argument("--max-reprojection", type=float, default=2.)
    p.add_argument('--ring-pairs', type=int, choices=(0, 1, 2, 4, 8), default=0,
                   help='Explicit approximate fixed-centre rotation panorama; 0 keeps ordinary two-view geometry')
    p.add_argument('--ring-candidates', type=int, default=24)
    p.add_argument('--ring-feature-contrast', type=float, default=.04,
                   help='Explicit SIFT detection contrast for weak texture; rotation acceptance gates are unchanged')
    p.add_argument('--ring-auto-coverage', action='store_true',
                   help='Choose multiple pairs by measured angular coverage, bounded by --ring-pairs')
    p.add_argument('--ring-coverage-target', type=float, default=.95)
    p.add_argument('--ring-min-pairs', type=int, default=1,
                   help='Continue auto selection to this many disjoint frame pairs after angular target is met')
    p.add_argument('--ring-max-gaussians', type=int, default=65536)
    p.add_argument("--interactive", action="store_true", help="Keep the scene resident and serve browser controls")
    p.add_argument("--viewer-host", default="127.0.0.1")
    p.add_argument("--viewer-port", type=int, default=8080)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--vendor", type=Path, required=True)
    p.add_argument("--renderer", type=Path, required=True)
    p.add_argument("--resident-binary", type=Path,
                   default=Path(__file__).resolve().parents[1]/"rendering/render_resident")
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
    p.add_argument('--warm-kernels', action='store_true',
                   help='Before video, run a discarded synthetic pair through the retained model')
    p.add_argument("--prepare-threads", type=int, default=1, help="OpenCV/decoder budget")
    schedule = p.add_mutually_exclusive_group()
    schedule.add_argument("--serial-prepare", dest="serial_prepare", action="store_true",
                          help="Complete poses before inference (measured default)")
    schedule.add_argument("--overlap-prepare", dest="serial_prepare", action="store_false",
                          help="Overlap PnP with inference; use --threads 3 on the four-core board")
    p.set_defaults(serial_prepare=True)
    p.add_argument("--focal-ratio", type=float, default=0.9)
    from gaussian_generation.npu_branch.config import add_arguments, BackendConfig
    add_arguments(p)
    a = p.parse_args(argv)
    if a.ring_pairs == 8 and not a.ring_auto_coverage:
        p.error('eight panorama pairs require automatic coverage; add --ring-auto-coverage')
    try:
        BackendConfig.from_args(a)
    except ValueError as exc:
        p.error(str(exc))
    if a.camera is not None:
        if a.calibration is None:
            p.error("--camera requires --calibration for the selected capture mode")
        from video_input.capture import validate_capture
        try:
            validate_capture(a.camera_width, a.camera_height, a.camera_fps, a.capture_seconds, a.capture_settle)
        except ValueError as exc:
            p.error(str(exc))
    if a.calibration:
        from video_input.calibration import Calibration
        calibration = Calibration(a.calibration)
        if a.camera and calibration.size != (a.camera_width, a.camera_height):
            p.error("Calibration image size differs from camera capture mode")
    if a.interactive and a.live_renderer is None:
        p.error("--interactive requires --live-renderer")
    if (not 8 <= a.ring_candidates <= 64 or not 1024 <= a.ring_max_gaussians <= 131072
            or (a.ring_pairs and (a.size != 128 or a.live_renderer is None))):
        p.error('Ring mode requires 128x128, native renderer and bounded sampling/point budget')
    if not math.isfinite(a.ring_feature_contrast) or not 0 < a.ring_feature_contrast <= .04:
        p.error('--ring-feature-contrast must be finite and in (0, .04]')
    if (not math.isfinite(a.ring_coverage_target) or not 0 < a.ring_coverage_target <= 1
            or (a.ring_auto_coverage and (not a.ring_pairs or
                                          not 1 <= a.ring_min_pairs <= a.ring_pairs))
            or (not a.ring_auto_coverage and a.ring_min_pairs != 1)):
        p.error('Automatic coverage needs an enabled ring pair budget and target in (0,1]')
    if (not 0 <= a.viewer_port <= 65535 or not 3 <= a.candidates <= 12
            or not math.isfinite(a.min_parallax) or a.min_parallax <= 0
            or not math.isfinite(a.max_reprojection) or a.max_reprojection <= 0):
        p.error("Invalid viewer port or selection thresholds")
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
        if a.camera is not None:
            from video_input.capture import capture_clip
            if not a.capture_now:
                guidance = ("请保持镜头中心固定并水平缓慢转满一圈："
                            if a.ring_pairs else "请缓慢平移摄像头并保持视野重叠：")
                input("摄像头已就绪，按回车开始采集；" + guidance)
            record["camera_capture"] = capture_clip(a.camera, out/"capture",
                width=a.camera_width, height=a.camera_height, fps=a.camera_fps,
                seconds=a.capture_seconds, settle=a.capture_settle, events=events)
            a.video = Path(record["camera_capture"]["path"])
        elif a.video is None:
            a.video = Path(input().strip())
        from video_input.receipt import receive_file
        receipt = receive_file(a.video, events)
        record["video_receipt"] = receipt
        if a.ring_pairs:
            from initialize.ring import prepare_ring_scene
            record['mode'] = 'sparse_fixed_centre_rotation_candidate'
            record.update(prepare_ring_scene(receipt['path'], out, session.model,
                session.renderer, events, groups='auto' if a.ring_auto_coverage else a.ring_pairs,
                max_pairs=a.ring_pairs, min_pairs=a.ring_min_pairs,
                coverage_target=a.ring_coverage_target,
                candidates=a.ring_candidates, feature_contrast=a.ring_feature_contrast,
                max_gaussians=a.ring_max_gaussians, size=a.size,
                prepare_threads=a.prepare_threads, focal_ratio=a.focal_ratio, calibration=a.calibration))
        else:
            from initialize.pipeline import prepare_scene
            record.update(prepare_scene(receipt["path"], out, session.model, session.renderer, events,
                size=a.size, prepare_threads=a.prepare_threads, focal_ratio=a.focal_ratio,
                overlap=not a.serial_prepare, calibration=a.calibration,
                auto_select=a.auto_select or a.camera is not None, candidates=a.candidates,
                min_parallax=a.min_parallax, max_reprojection=a.max_reprojection))
        if a.interactive:
            from web.viewer import serve_scene
            record["interactive"] = serve_scene(session.renderer.runtime, out/"input", out/"inference",
                                                 host=a.viewer_host, port=a.viewer_port)
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
