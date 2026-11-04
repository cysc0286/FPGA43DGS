"""Ordered lifecycle events and the video-end to verified-frame clock."""
import json
from pathlib import Path
import time


ORDER = ("READY", "VIDEO_COMPLETE", "SCENE_READY", "FRAME_COMPLETE")


class EventLog:
    def __init__(self, path):
        self.path = Path(path)
        self.events = []

    def emit(self, name, **details):
        primary = [item["event"] for item in self.events if item["event"] in ORDER]
        capture = [item["event"] for item in self.events if item["event"].startswith("CAPTURE_")]
        if name in ("CAPTURE_STARTED", "CAPTURE_COMPLETE"):
            expected = [] if name == "CAPTURE_STARTED" else ["CAPTURE_STARTED"]
            valid = primary == ["READY"] and capture == expected
        else:
            valid = (name in ORDER and len(primary) == ORDER.index(name)
                     and (name != "VIDEO_COMPLETE" or not capture or capture[-1] == "CAPTURE_COMPLETE"))
        if not valid:
            raise ValueError("Lifecycle event out of order: " + name)
        if set(details) & {"event", "epoch", "monotonic"}:
            raise ValueError("Reserved event fields")
        item = dict(event=name, epoch=time.time(), monotonic=time.monotonic(), **details)
        pending = self.events + [item]
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(schema="mvsplat_lifecycle_v1", events=pending),
                                       indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self.path)
        self.events = pending
        return item

    def summary(self):
        by_name = {event["event"]: event for event in self.events}
        result = dict(complete="FRAME_COMPLETE" in by_name)
        for before, after, field in (("CAPTURE_STARTED", "CAPTURE_COMPLETE", "capture_seconds_record_only"),
                                     ("CAPTURE_COMPLETE", "FRAME_COMPLETE", "capture_complete_to_frame_seconds"),
                                     ("CAPTURE_STARTED", "FRAME_COMPLETE", "capture_start_to_frame_seconds"),
                                     ("READY", "VIDEO_COMPLETE", "ready_to_receipt_seconds_record_only"),
                                     ("VIDEO_COMPLETE", "SCENE_READY", "gaussian_scene_seconds"),
                                     ("SCENE_READY", "FRAME_COMPLETE", "first_render_seconds"),
                                     ("VIDEO_COMPLETE", "FRAME_COMPLETE", "scene_preparation_seconds")):
            if before in by_name and after in by_name:
                result[field] = by_name[after]["monotonic"] - by_name[before]["monotonic"]
        return result


def first_frame_timing(pipeline):
    """Normalize pre- and post-input-receipt timing records without inventing a receipt."""
    if "first_verified_fpga_target" in pipeline:
        return dict(
            input_end_to_first_verified_fpga_frame_seconds=pipeline.get(
                "input_end_to_first_verified_fpga_frame_seconds"),
            pipeline_start_to_first_verified_fpga_frame_seconds=pipeline[
                "pipeline_start_to_first_verified_fpga_frame_seconds"],
            first_verified_fpga_target=pipeline["first_verified_fpga_target"],
        )
    return dict(
        input_end_to_first_verified_fpga_frame_seconds=None,
        pipeline_start_to_first_verified_fpga_frame_seconds=pipeline[
            "video_to_first_fpga_image_seconds"],
        first_verified_fpga_target=None,
        legacy_timing_note="No input-end receipt was recorded by this historical run.",
    )
