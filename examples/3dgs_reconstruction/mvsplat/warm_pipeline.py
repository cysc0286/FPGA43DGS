"""Post-video work only: context/pose overlap, inference, export, first frame."""
from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import time

from common import sha


def overlap_prepare_infer(prepare, prepare_args, infer, inference_out):
    """Run Torch on its initialization thread; only OpenCV preparation is a worker.

    Context publication is immutable and one-shot. A PnP failure is propagated
    even if inference already completed; no partial scene is published.
    """
    context = Future()
    started = time.monotonic()
    timing = {}

    def publish(directory, digest):
        if context.done():
            raise ValueError("Context published more than once")
        timing["context_ready_seconds"] = time.monotonic()-started
        context.set_result((directory, digest))

    def work():
        try:
            prepare(prepare_args, on_context_ready=publish)
            if not context.done():
                raise ValueError("Preparation did not publish context")
            timing["poses_complete_seconds"] = time.monotonic()-started
        except BaseException as exc:
            if not context.done():
                context.set_exception(exc)
            raise

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-pose") as pool:
        preparing = pool.submit(work)
        directory, digest = context.result()
        timing["inference_begin_seconds"] = time.monotonic()-started
        result = infer(directory, inference_out, digest)
        timing["inference_end_seconds"] = time.monotonic()-started
        preparing.result()
    timing["combined_seconds"] = time.monotonic()-started
    timing["overlap_seconds"] = max(0., min(timing["inference_end_seconds"],
        timing["poses_complete_seconds"])-timing["inference_begin_seconds"])
    return result, timing


def prepare_scene(video, out, runtime, renderer, events, size=128,
                  prepare_threads=1, focal_ratio=0.9, overlap=True):
    from video_input.prepare import main as prepare
    from export import main as export
    out = Path(out)
    args = ["--video", str(video), "--out", str(out / "input"),
            "--size", str(size), "--threads", str(prepare_threads),
            "--focal-ratio", str(focal_ratio)]
    if overlap:
        inference, timing = overlap_prepare_infer(prepare, args, runtime.infer, out / "inference")
    else:
        t = time.monotonic()
        prepare(args)
        inference = runtime.infer(out / "input", out / "inference", sha(out / "input/context.npz"))
        timing = dict(combined_seconds=time.monotonic()-t, overlap_seconds=0.)
    meta = json.loads((out / "input/input.json").read_text(encoding="utf-8"))
    if not inference["complete"] or meta["context_sha256"] != inference["input_sha256"]:
        raise ValueError("Inference/context contract failed")
    export(["--input", str(out / "input"), "--inference", str(out / "inference"),
            "--out", str(out / "renderer_input")])
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
