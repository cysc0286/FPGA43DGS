"""Small live interface: load a scene once, return complete RGB frames in memory."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import time

import numpy as np


@dataclass
class LiveFrame:
    rgb: np.ndarray
    metadata: dict
    raw: bytes = b""

    def archive(self, folder):
        """Explicitly outside render latency. Callers choose when to archive."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        h, w = self.rgb.shape[:2]
        data = self.rgb.tobytes()
        (folder / "frame.ppm").write_bytes(("P6\n%d %d\n255\n" % (w, h)).encode() + data)
        if self.raw:
            (folder / "frame.bin").write_bytes(self.raw)
        record = dict(self.metadata, rgb_sha256=hashlib.sha256(data).hexdigest(),
                      frame_sha256=hashlib.sha256(self.raw).hexdigest() if self.raw else None)
        (folder / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record


class LiveRenderer:
    def __init__(self, binary, environment, log_path, *, threads=4, max_gaussians=0,
                 batch=2, cached=True, cpu=False, comparison_sort=False, uniform_preview=False,
                 compact_payload=False, depth_layout=False):
        self.process = None
        self.log = Path(log_path).open("wb")
        self.pending = bytearray()
        self.scene = None
        self.revision = 0
        self.configuration = dict(threads=threads, max_gaussians=max_gaussians,
                                  batch=batch, cached=cached, comparison_sort=comparison_sort,
                                  uniform_preview=uniform_preview,
                                  compact_payload=compact_payload, depth_layout=depth_layout,
                                  backend="cpu_dense" if cpu else "fpga")
        command = [str(binary), "--threads", str(threads), "--max-gaussians",
                   str(max_gaussians), "--batch", str(batch)]
        if not cached:
            command.append("--uncached")
        if cpu:
            command.append("--cpu")
        if comparison_sort:
            command.append("--comparison-sort")
        if uniform_preview:
            command.append("--uniform-preview")
        if compact_payload:
            command.append("--compact-payload")
        if depth_layout:
            command.append("--depth-layout")
        try:
            environment = dict(environment)
            environment.setdefault("OMP_WAIT_POLICY", "PASSIVE")
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.log, env=environment, bufsize=0)
            deadline = time.monotonic() + 40
            while self._line(deadline) != b"LIVE_READY 1":
                pass  # SDK startup banners precede the framed protocol.
        except BaseException:
            self.close()
            raise

    def _receive(self, deadline):
        if not select.select([self.process.stdout], [], [], max(0, deadline-time.monotonic()))[0]:
            raise TimeoutError("Live renderer response timed out")
        data = os.read(self.process.stdout.fileno(), 65536)
        if not data:
            raise RuntimeError("Live renderer exited: " + str(self.process.poll()))
        self.pending.extend(data)

    def _line(self, deadline):
        while b"\n" not in self.pending:
            self._receive(deadline)
            if len(self.pending) > 65536:
                raise ValueError("Live protocol header too long")
        end = self.pending.index(b"\n")
        value = bytes(self.pending[:end])
        del self.pending[:end+1]
        return value

    def _exact(self, size, deadline):
        if size < 0 or size > 2048*2048*20+16:
            raise ValueError("Live payload size")
        while len(self.pending) < size:
            self._receive(deadline)
        data = bytes(self.pending[:size])
        del self.pending[:size]
        return data

    def _send(self, value):
        # Unbuffered pipe writes may be partial, especially for large LOADROWS.
        view = memoryview(value)
        while view:
            n = self.process.stdin.write(view)
            if not n:
                raise BrokenPipeError("Live renderer input closed")
            view = view[n:]

    def _loaded(self, started):
        line = self._line(time.monotonic()+120).decode().split()
        if len(line) != 4 or line[0] != "SCENE":
            raise RuntimeError("Scene load failed: " + " ".join(line))
        self.revision += 1
        self.scene = dict(gaussians=int(line[1]), selected=int(line[2]), native_load_ms=float(line[3]),
                          load_seconds=time.monotonic()-started, revision=self.revision)
        return dict(self.scene)

    def load_scene(self, model):
        path = Path(model).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        started = time.monotonic()
        self._send(("LOAD " + json.dumps(str(path), ensure_ascii=False) + "\n").encode())
        return self._loaded(started)

    def load_rows(self, rows):
        rows = np.asarray(rows)
        if rows.ndim != 2 or rows.shape[1] != 62 or not 0 < len(rows) <= 1000000:
            raise ValueError("Gaussian row shape")
        rows = np.ascontiguousarray(rows, dtype="<f4")
        if not np.isfinite(rows).all():
            raise ValueError("Nonfinite Gaussian rows")
        started = time.monotonic()
        self._send(("LOADROWS %d\n" % len(rows)).encode())
        self._send(memoryview(rows).cast("B"))
        return self._loaded(started)

    def render_camera(self, camera, *, include_raw=False):
        """Latency ends after complete RGB payload receipt and shape validation."""
        started = time.monotonic()
        camera = bytes(camera)
        if self.scene is None:
            raise ValueError("Load a scene before rendering")
        if len(camera) != 136 or camera[:8] != b"FLCAM001":
            raise ValueError("Camera ABI")
        if not np.isfinite(np.frombuffer(camera, dtype="<f8", offset=24)).all():
            raise ValueError("Nonfinite camera")
        self._send((b"RENDER_RAW\n" if include_raw else b"RENDER\n") + camera)
        deadline = time.monotonic()+120
        line = self._line(deadline)
        if not line.startswith(b"FRAME "):
            raise RuntimeError("Invalid frame response")
        record = json.loads(line[6:])
        w, h = record["width"], record["height"]
        if not 0 < w <= 2048 or not 0 < h <= 2048 or record["rgb_bytes"] != w*h*3:
            raise ValueError("Frame shape")
        if record["raw_bytes"] != (16+w*h*20 if include_raw else 0):
            raise ValueError("Raw frame shape")
        rgb = np.frombuffer(self._exact(record["rgb_bytes"], deadline), dtype=np.uint8).reshape(h, w, 3)
        raw = self._exact(record["raw_bytes"], deadline)
        if raw and (raw[:8] != b"GSSOUT01" or
                    tuple(np.frombuffer(raw, dtype="<u4", count=2, offset=8)) != (w, h)):
            raise ValueError("Raw framebuffer ABI")
        record.update(complete=True, wall_ms=(time.monotonic()-started)*1000,
                      scene_revision=self.revision, configuration=dict(self.configuration),
                      endpoint="complete RGB framebuffer received; archive excluded")
        return LiveFrame(rgb, record, raw)

    def close(self):
        p = self.process
        try:
            if p is not None:
                if p.poll() is None:
                    try:
                        self._send(b"QUIT\n")
                        p.wait(timeout=35)
                    except (OSError, subprocess.TimeoutExpired):
                        p.terminate()
                        try:
                            p.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            p.kill()
                            p.wait(timeout=5)
                p.stdin.close()
                p.stdout.close()
        finally:
            self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
