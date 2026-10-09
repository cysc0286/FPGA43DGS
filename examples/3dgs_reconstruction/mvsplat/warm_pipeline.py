"""Post-video work only: context/pose overlap, inference, export, first frame."""
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

from common import sha


def overlap_prepare_infer(prepare, prepare_args, infer, inference_out,
                          on_first_target_ready=None):
    """Run Torch on its initialization thread; only OpenCV preparation is a worker.

    Context and first-target publication are immutable and one-shot. A later
    target PnP failure is propagated, even if the first frame was published.
    """
    context = Future()
    first_target = Future()
    started = time.monotonic()
    timing = {}

    def publish(directory, digest):
        if context.done():
            raise ValueError("Context published more than once")
        timing["context_ready_seconds"] = time.monotonic()-started
        context.set_result((directory, digest))

    def publish_first(directory, digest, target):
        if first_target.done():
            raise ValueError("First target published more than once")
        timing["first_target_ready_seconds"] = time.monotonic()-started
        first_target.set_result((directory, digest, target))

    def work():
        try:
            kwargs = {"on_context_ready": publish}
            if on_first_target_ready is not None:
                kwargs["on_first_target_ready"] = publish_first
            prepare(prepare_args, **kwargs)
            if not context.done():
                raise ValueError("Preparation did not publish context")
            if on_first_target_ready is not None and not first_target.done():
                raise ValueError("Preparation did not publish first target")
            timing["poses_complete_seconds"] = time.monotonic()-started
        except BaseException as exc:
            if not context.done():
                context.set_exception(exc)
            if on_first_target_ready is not None and not first_target.done():
                first_target.set_exception(exc)
            raise

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-pose") as pool:
        preparing = pool.submit(work)
        directory, digest = context.result()
        timing["inference_begin_seconds"] = time.monotonic()-started
        result = infer(directory, inference_out, digest)
        timing["inference_end_seconds"] = time.monotonic()-started
        if on_first_target_ready is not None:
            first_directory, first_digest, target = first_target.result()
            if first_digest != digest:
                raise ValueError("First target/context digest mismatch")
            on_first_target_ready(first_directory, first_digest, target, result, timing)
        preparing.result()
    timing["combined_seconds"] = time.monotonic()-started
    timing["overlap_seconds"] = max(0., min(timing["inference_end_seconds"],
        timing["poses_complete_seconds"])-timing["inference_begin_seconds"])
    return result, timing


def prepare_scene(video, out, runtime, renderer, events, size=128,
                  prepare_threads=1, focal_ratio=0.9, overlap=True):
    from scene_preparation import main as prepare
    from export import main as export
    out = Path(out)
    args = ["--video", str(video), "--out", str(out / "input"),
            "--size", str(size), "--threads", str(prepare_threads),
            "--focal-ratio", str(focal_ratio)]
    first_frame = None
    first_inference = None
    critical_gaussians = None
    critical_rows_sha256 = None
    critical_camera_path = None

    def publish_first_frame(input_dir, digest, target_index, inference, scheduling):
        nonlocal first_frame, first_inference, critical_gaussians, critical_rows_sha256, critical_camera_path
        from export import camera_bytes, gaussian_rows
        meta = json.loads((input_dir / "input.initial.json").read_text(encoding="utf-8"))
        if meta["context_sha256"] != digest:
            raise ValueError("Initial target/context provenance failed")
        target = next(v for v in meta["views"] if v["role"] == "target")
        if target["index"] != target_index:
            raise ValueError("Published first target changed")
        critical_gaussians = runtime.take_gaussians(inference["gaussian_sha256"])
        rows_started = time.monotonic()
        rows, _ = gaussian_rows(critical_gaussians)
        scheduling["gaussian_row_conversion_seconds"] = time.monotonic()-rows_started
        critical_rows_sha256 = hashlib.sha256(rows.tobytes()).hexdigest()
        initial_folder = out / "renderer_critical"
        initial_folder.mkdir()
        camera_name = Path(target["name"]).stem + ".bin"
        camera_path = initial_folder / camera_name
        critical_camera_path = camera_path
        camera_path.write_bytes(camera_bytes(target))
        resident = renderer.load_rows(rows)
        scheduling["renderer_scene_load_seconds"] = resident["load_seconds"]
        if resident["rows_sha256"] != critical_rows_sha256:
            raise ValueError("Resident Gaussian rows hash mismatch")
        events.emit("SCENE_READY", gaussian_sha256=inference["gaussian_sha256"],
                    rows_sha256=critical_rows_sha256, camera=camera_name,
                    resident_scene=resident,
                    scope="validated in-memory first-target scene; PLY audit follows")
        frame_dir = out / "board" / (Path(camera_name).stem + "_fpga")
        frame = renderer.render_camera(camera_path.read_bytes(), frame_dir)
        if not frame["complete"] or frame["frame_sha256"] != sha(frame_dir / "frame.bin"):
            raise ValueError("Incomplete first target frame")
        events.emit("FRAME_COMPLETE", frame_sha256=frame["frame_sha256"], camera=camera_name)
        first_frame, first_inference = frame, inference

    if overlap:
        def infer_memory(directory, destination, digest):
            return runtime.infer(directory, destination, digest, defer_archive=True)
        inference, timing = overlap_prepare_infer(prepare, args, infer_memory, out / "inference",
                                                   on_first_target_ready=publish_first_frame)
    else:
        t = time.monotonic()
        prepare(args)
        inference = runtime.infer(out / "input", out / "inference", sha(out / "input/context.npz"))
        timing = dict(combined_seconds=time.monotonic()-t, overlap_seconds=0.)
    meta = json.loads((out / "input/input.json").read_text(encoding="utf-8"))
    if not inference["complete"] or meta["context_sha256"] != inference["input_sha256"]:
        raise ValueError("Inference/context contract failed")
    # Preserve the video-side timing breakdown in the returned record. These
    # are diagnostic fields, not a replacement for the end-to-end boundary.
    timing["video_prepare_stages"] = dict(meta.get("timing", {}))
    timing["video_prepare_total_seconds"] = float(meta.get("seconds", 0.0))
    if first_frame is None:
        export(["--input", str(out / "input"), "--inference", str(out / "inference"),
                "--out", str(out / "renderer_input")],
               gaussians=runtime.take_gaussians(inference["gaussian_sha256"]))
    else:
        runtime.persist_gaussians(inference, critical_gaussians)
        export(["--input", str(out / "input"), "--inference", str(out / "inference"),
                "--out", str(out / "renderer_input")], gaussians=critical_gaussians)
        manifest = json.loads((out / "renderer_input/manifest.json").read_text(encoding="utf-8"))
        if manifest["rows_sha256"] != critical_rows_sha256:
            raise ValueError("Deferred PLY differs from first-frame Gaussian rows")
        if sha(critical_camera_path) != next(
                c["sha256"] for c in manifest["cameras"] if c["role"] == "target"):
            raise ValueError("Deferred camera differs from first-frame camera")
    if first_frame is not None:
        return dict(frame=first_frame, inference=first_inference, scheduling=timing,
                    renderer_input=str(out / "renderer_input"),
                    first_target_fast_path=True)
    folder = out / "renderer_input"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if sha(folder / "model.ply") != manifest["model_sha256"]:
        raise ValueError("Exported model hash mismatch")
    target = next(c for c in manifest["cameras"] if c["role"] == "target")
    resident = renderer.load_scene(folder / "model.ply")
    events.emit("SCENE_READY", gaussian_sha256=inference["gaussian_sha256"],
                model_sha256=manifest["model_sha256"], camera=target["file"],
                resident_scene=resident, scope="validated scene files and CPU-resident Gaussian rows")
    frame_dir = out / "board" / (Path(target["file"]).stem + "_fpga")
    frame = renderer.render(folder / "model.ply", folder / target["file"], frame_dir)
    if not frame["complete"] or frame["frame_sha256"] != sha(frame_dir / "frame.bin"):
        raise ValueError("Incomplete first frame")
    events.emit("FRAME_COMPLETE", frame_sha256=frame["frame_sha256"], camera=target["file"])
    return dict(frame=frame, inference=inference, scheduling=timing)
