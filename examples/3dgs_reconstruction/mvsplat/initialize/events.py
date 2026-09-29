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
        if name not in ORDER or len(self.events) != ORDER.index(name):
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
        for before, after, field in (("READY", "VIDEO_COMPLETE", "ready_to_receipt_seconds_record_only"),
                                     ("VIDEO_COMPLETE", "SCENE_READY", "gaussian_scene_seconds"),
                                     ("SCENE_READY", "FRAME_COMPLETE", "first_render_seconds"),
                                     ("VIDEO_COMPLETE", "FRAME_COMPLETE", "scene_preparation_seconds")):
            if before in by_name and after in by_name:
                result[field] = by_name[after]["monotonic"] - by_name[before]["monotonic"]
        return result
