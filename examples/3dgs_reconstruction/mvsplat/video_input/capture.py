"""Bounded Linux UVC short-clip acquisition; no model or FPGA initialization."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import time

from common import save, sha


def validate_capture(width, height, fps, seconds, settle):
    import math
    if (type(width) is not int or type(height) is not int
            or not 64 <= width <= 1920 or not 64 <= height <= 1080
            or not all(math.isfinite(v) for v in (fps, seconds, settle))
            or not 1 <= fps <= 120 or not 1 <= seconds <= 60 or not 0 <= settle <= 10):
        raise ValueError("Capture limits: 64..1920×64..1080, 1..120 fps, 1..60 s, settle 0..10 s")


def probe(device):
    """Read-only enumeration. A missing utility/driver is an explicit error."""
    results = {}
    for key, command in (
        ("usb", ["lsusb", "-t"]),
        ("formats", ["v4l2-ctl", "-d", str(device), "--list-formats-ext"]),
        ("controls", ["v4l2-ctl", "-d", str(device), "--list-ctrls-menus"])):
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=True)
        results[key] = result.stdout
    return results


def inspect_clip(path, ffprobe="ffprobe", *, decoded_frames=False):
    """Inspect strict acquisition packets or decoded file presentation frames.

    UVC MJPEG capture keeps its original packet-PTS contract by default. A
    general compressed file may contain B-frames whose packet PTS legitimately
    reorder; callers must explicitly request decoded presentation order. Never
    sort timestamps, since that would hide genuinely malformed frame order.
    """
    collection = "frames" if decoded_frames else "packets"
    timestamp = "best_effort_timestamp_time" if decoded_frames else "pts_time"
    entries = ("frame=best_effort_timestamp_time" if decoded_frames
               else "packet=pts_time,duration_time")
    result = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_streams", "-show_frames" if decoded_frames else "-show_packets", "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate:"+entries,
        "-of", "json", str(path)], capture_output=True, text=True,
        timeout=120 if decoded_frames else 30, check=True)
    record = json.loads(result.stdout)
    if (not isinstance(record, dict) or not isinstance(record.get("streams"), list)
            or len(record["streams"]) != 1 or not isinstance(record.get(collection), list)
            or len(record[collection]) < 3):
        raise ValueError("Captured clip needs one video stream and at least three frames")
    import math
    stream = record["streams"][0]
    try:
        if (not isinstance(stream["codec_name"], str) or not stream["codec_name"]
                or any(type(stream[key]) is not int or stream[key] <= 0 for key in ("width", "height"))):
            raise ValueError("Invalid codec or frame dimensions")
        fps = float(Fraction(stream["avg_frame_rate"]))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid frame rate")
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError("Invalid or missing video stream metadata") from exc
    try:
        times = [float(frame[timestamp]) for frame in record[collection]]
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid or missing frame timestamps") from exc
    if not all(math.isfinite(v) for v in times) or any(b <= a for a, b in zip(times, times[1:])):
        scope = "decoded" if decoded_frames else "captured"
        raise ValueError("Invalid or non-monotonic "+scope+" frame timestamps")
    stream["fps"] = fps
    return stream, times


def capture_clip(device, out, *, width=1280, height=720, fps=30., seconds=5., settle=1.,
                 events=None, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    validate_capture(width, height, fps, seconds, settle)
    device = Path(device)
    if not device.as_posix().startswith("/dev/") or not device.exists():
        raise ValueError("Camera must be an existing Linux /dev capture node (probe it first)")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    # Bound compressed storage at the muxer; do not require raw-video-sized space.
    available = shutil.disk_usage(out).free - 64*1024**2
    if available < max(8*1024**2, width*height*2):
        raise OSError("Insufficient free capture storage")
    storage_limit = min(256*1024**2, available)
    # Board FFmpeg 4.2's AVI muxer reports doubled FPS/empty frame slots for UVC.
    # Matroska preserves timestamps without changing or re-encoding JPEG packets.
    partial, final = out/"capture.partial.mkv", out/"capture.mkv"
    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-n", "-f", "v4l2",
               "-input_format", "mjpeg", "-video_size", f"{width}x{height}",
               "-framerate", str(fps), "-i", str(device), "-ss", str(settle),
               "-t", str(seconds), "-an", "-c:v", "copy", "-fs", str(storage_limit), str(partial)]
    record = dict(complete=False, device=str(device), resolved_device=str(device.resolve()),
                  requested=dict(width=width, height=height, fps=fps, seconds=seconds, settle=settle),
                  command=command, storage_limit_bytes=storage_limit,
                  timestamp_scope="demuxed clip PTS; not sensor exposure timestamps")
    started = time.monotonic()
    if events is not None:
        events.emit("CAPTURE_STARTED", device=str(device), requested=record["requested"])
    try:
        with (out/"ffmpeg.log").open("wb") as log:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=log,
                           timeout=seconds+settle+20, check=True)
        stream, timestamps = inspect_clip(partial, ffprobe)
        if (stream["codec_name"] != "mjpeg" or (stream["width"], stream["height"]) != (width, height)
                or abs(stream["fps"]-fps) > max(.1, fps*.01)):
            raise ValueError("Camera negotiated another mode; inspect ffmpeg.log and probe formats")
        # Catch early disconnects that return a successful but truncated recording.
        if timestamps[-1]-timestamps[0] < seconds - max(.25, 2/fps):
            raise ValueError("Captured clip shorter than requested; camera may have disconnected")
        measured_fps = (len(timestamps)-1)/(timestamps[-1]-timestamps[0])
        stream.update(measured_fps=measured_fps,
                      max_frame_gap_seconds=max(b-a for a, b in zip(timestamps, timestamps[1:])))
        if measured_fps < fps*.9:
            raise ValueError("Capture lost over 10% of requested frame rate; inspect USB load/exposure")
        partial.rename(final)
        record.update(complete=True, path=str(final.resolve()), sha256=sha(final),
                      actual=stream, frame_pts_seconds=timestamps, frames=len(timestamps),
                      acquisition_seconds=time.monotonic()-started)
        if events is not None:
            events.emit("CAPTURE_COMPLETE", path=record["path"], frames=len(timestamps))
        return record
    except BaseException as exc:
        record["error"] = type(exc).__name__+": "+str(exc)
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic()-started
        save(out/"capture.json", record)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="/dev/video0")
    p.add_argument("--probe", action="store_true")
    p.add_argument("--out", type=Path)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.)
    p.add_argument("--seconds", type=float, default=5.)
    p.add_argument("--settle", type=float, default=1.)
    a = p.parse_args(argv)
    if a.probe:
        print(json.dumps(probe(a.device), ensure_ascii=False, indent=2))
    elif a.out is None:
        p.error("--out is required for capture")
    else:
        print(json.dumps(capture_clip(a.device, a.out, width=a.width, height=a.height,
            fps=a.fps, seconds=a.seconds, settle=a.settle), indent=2))


if __name__ == "__main__":
    main()
