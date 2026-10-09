"""Adapt the live backend to the existing first-frame evidence contract."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from initialize.renderer_runtime import frozen_renderer_module
from rendering.runtime import LiveRenderer, load_mainline_profile


def validate_pipeline_configuration(profile, configuration):
    """Keep the accepted profile fixed; tuning requires an explicit custom run."""
    if profile not in ("mainline", "custom"):
        raise ValueError("render profile must be mainline or custom")
    if profile == "mainline":
        accepted = load_mainline_profile()["runtime"]
        changed = [key for key, value in configuration.items()
                   if key not in accepted or type(value) is not type(accepted[key])
                   or value != accepted[key]]
        if changed:
            raise ValueError("Mainline settings cannot be overridden: " + ", ".join(changed)
                             + "; select --render-profile custom for experiments")


class PipelineRenderer:
    def __init__(self, renderer, binary, environment, log_path, *, profile="mainline",
                 **configuration):
        validate_pipeline_configuration(profile, configuration)
        self.manifest = frozen_renderer_module(Path(renderer)).verify()
        if profile == "mainline":
            self.runtime = LiveRenderer.mainline(binary, environment, log_path)
        else:
            self.runtime = LiveRenderer(binary, environment, log_path, **configuration)
            self.runtime.configuration["profile"] = "custom"
        self.scene = None
        self.last_rgb = None

    def load_rows(self, rows):
        rows = np.ascontiguousarray(rows, dtype="<f4")
        self.scene = self.runtime.load_rows(rows)
        self.scene["rows_sha256"] = hashlib.sha256(rows.tobytes()).hexdigest()
        return dict(self.scene)

    def load_scene(self, model):
        self.scene = self.runtime.load_scene(model)
        self.scene["sha256"] = hashlib.sha256(Path(model).read_bytes()).hexdigest()
        return dict(self.scene)

    def render_camera(self, camera, out, work_root=None):
        started = time.monotonic()
        frame = self.runtime.render_camera(camera, include_raw=True)
        self.last_rgb = frame.rgb
        record = frame.archive(out)
        record.update(backend="fpga", package=self.manifest["name"], resident_scene=self.scene,
                      total_ms=(time.monotonic()-started)*1000,
                      live_frame_ms=frame.metadata["wall_ms"],
                      endpoint="first-frame compatibility: raw frame and evidence archived")
        (Path(out)/"result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record

    def render(self, model, camera, out, work_root=None):
        if model is not None:
            self.load_scene(model)
        return self.render_camera(Path(camera).read_bytes(), out, work_root)

    def close(self):
        self.runtime.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
