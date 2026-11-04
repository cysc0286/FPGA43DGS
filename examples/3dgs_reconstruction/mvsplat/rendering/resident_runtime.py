"""Keep the FPGA board session open while using the frozen scene preprocessors."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import struct
import tempfile
import time
import numpy as np

from common import sha


def frozen_renderer_module(renderer):
    spec = importlib.util.spec_from_file_location("frozen_gs_render", renderer / "render.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def framebuffer_rgb(raw):
    """Preserve frozen Python float64, ties-to-even display conversion exactly."""
    if len(raw) < 16 or raw[:8] != b"GSSOUT01":
        raise ValueError("Framebuffer ABI")
    width, height = struct.unpack_from("<II", raw, 8)
    if not width or not height or len(raw) != 16 + width * height * 20:
        raise ValueError("Framebuffer size")
    values = np.frombuffer(raw, dtype="<f4", offset=16).reshape(-1, 5)[:, :4]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite framebuffer")
    rgb = np.rint(np.clip(values[:, :3].astype(np.float64), 0, 1) * 255).astype(np.uint8)
    return width, height, rgb.reshape(height, width, 3)


class RendererRuntime:
    def __init__(self, renderer, binary, environment, log_path):
        self.renderer, self.binary = Path(renderer), Path(binary)
        self.environment = environment
        self.frozen = frozen_renderer_module(self.renderer)
        self.manifest = self.frozen.verify()
        self.log = Path(log_path).open("w", encoding="utf-8")
        self.pending = {}
        self.attributes = None
        self.sorter = None
        self.scene = None
        self.last_rgb = None
        self.process = None
        try:
            self.process = subprocess.Popen([str(self.binary), "serve"], stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=self.log, bufsize=0,
                                            env=self.environment)
            deadline = time.monotonic() + 30
            self.startup_output = []
            while True:
                line = self._line(max(0, deadline-time.monotonic()))
                if line == "RENDER_READY":
                    break
                self.startup_output.append(line)
            self.attributes = subprocess.Popen([str(self.binary.with_name("attributes_resident"))],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, bufsize=0, env=self.environment)
            if self._line(30, self.attributes) != "ATTRIBUTES_READY":
                raise ValueError("Attribute runtime did not initialize")
            sorter_binary = self.binary.with_name("group_sort_resident")
            if sorter_binary.is_file():
                self.sorter = subprocess.Popen([str(sorter_binary)], stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=self.log, bufsize=0, env=self.environment)
                if self._line(30, self.sorter) != "SORT_READY":
                    raise ValueError("Sort runtime did not initialize")
        except Exception:
            self.close()
            raise

    def _line(self, timeout, process=None):
        process = process or self.process
        pending = self.pending.get(process.pid, b"")
        deadline = time.monotonic() + timeout
        while b"\n" not in pending:
            if not select.select([process.stdout], [], [], max(0, deadline-time.monotonic()))[0]:
                raise TimeoutError("Persistent renderer response timed out")
            block = os.read(process.stdout.fileno(), 4096)
            if not block:
                raise RuntimeError("Persistent renderer exited: " + str(process.poll()))
            pending += block
        line, self.pending[process.pid] = pending.split(b"\n", 1)
        return line.decode("utf-8", "replace").strip()

    def load_scene(self, model):
        model = Path(model).resolve()
        digest = sha(model)
        if self.scene is not None and self.scene.get("sha256") == digest:
            return self.scene
        started = time.monotonic()
        self.attributes.stdin.write(("LOAD "+json.dumps(str(model), ensure_ascii=False)+"\n").encode())
        self.attributes.stdin.flush()
        response = self._line(60, self.attributes)
        if not response.startswith("SCENE_LOADED "):
            raise ValueError("Scene load failed: " + response)
        self.scene = dict(path=str(model), sha256=digest, gaussians=int(response.split()[1]),
                          load_seconds=time.monotonic()-started, residence="CPU Gaussian rows")
        return self.scene

    def load_rows(self, rows):
        rows = np.asarray(rows)
        if rows.dtype != np.dtype("<f4") or rows.ndim != 2 or rows.shape[1] != 62 \
                or not 0 < len(rows) <= 1000000 or not np.isfinite(rows).all():
            raise ValueError("Invalid in-memory Gaussian rows")
        payload = np.ascontiguousarray(rows).tobytes()
        digest = hashlib.sha256(payload).hexdigest()
        if self.scene is not None and self.scene.get("rows_sha256") == digest:
            return self.scene
        started = time.monotonic()
        self.attributes.stdin.write(("LOADROWS " + str(len(rows)) + "\n").encode())
        self.attributes.stdin.write(payload)
        self.attributes.stdin.flush()
        response = self._line(60, self.attributes)
        if response != "SCENE_LOADED " + str(len(rows)):
            raise ValueError("In-memory scene load failed: " + response)
        self.scene = dict(rows_sha256=digest, gaussians=len(rows),
                          load_seconds=time.monotonic()-started,
                          residence="CPU Gaussian rows from validated memory")
        return self.scene

    def render(self, model, camera, out, work_root=Path("/dev/shm")):
        model = None if model is None else Path(model)
        camera = Path(camera)
        if (model is not None and not model.is_file()) or not camera.is_file():
            raise FileNotFoundError("Missing model or camera")
        if model is not None:
            self.load_scene(model)
        return self.render_camera(camera.read_bytes(), out, work_root)

    def render_camera(self, camera, out, work_root=Path("/dev/shm")):
        """Submit a 136-byte camera to the loaded scene; retain RGB in memory."""
        started = time.monotonic()
        self.last_rgb = None
        camera = bytes(camera)
        if len(camera) != 136 or camera[:8] != b"FLCAM001":
            raise ValueError("Camera payload ABI")
        if not np.isfinite(np.frombuffer(camera, dtype="<f8", offset=24)).all():
            raise ValueError("Nonfinite camera payload")
        out = Path(out)
        out.mkdir(parents=True, exist_ok=False)
        record = dict(complete=False, backend="fpga", camera_sha256=hashlib.sha256(camera).hexdigest(),
                      package=self.manifest["name"])
        if self.scene is not None and "sha256" in self.scene:
            record["model_sha256"] = self.scene["sha256"]
        try:
            scene = self.scene
            if scene is None:
                raise ValueError("No resident scene loaded")
            record["resident_scene"] = scene
            with tempfile.TemporaryDirectory(prefix="gs-resident-", dir=str(work_root)) as temporary:
                work = Path(temporary)
                with (out / "run.log").open("w") as log:
                    self.attributes.stdin.write(("PROJECTMEM "+
                        json.dumps(str(work / "attr.bin"), ensure_ascii=False)+"\n").encode())
                    self.attributes.stdin.write(camera)
                    self.attributes.stdin.flush()
                    if self._line(120, self.attributes) != "PROJECT_COMPLETE":
                        raise ValueError("Resident projection failed")
                    projected = time.monotonic()
                    if self.sorter is None:
                        subprocess.run([str(self.renderer / "bin/group_sort"), str(work / "attr.bin"),
                                        str(work / "scene.bin")], check=True, env=self.environment,
                                       stdout=log, stderr=subprocess.STDOUT, timeout=120)
                    else:
                        command = "SORT " + json.dumps(str(work / "attr.bin")) + " " + json.dumps(str(work / "scene.bin"))
                        self.sorter.stdin.write((command + "\n").encode())
                        self.sorter.stdin.flush()
                        response = self._line(120, self.sorter)
                        if response.startswith("points="):
                            log.write(response + "\n")
                            response = self._line(120, self.sorter)
                        if response != "SORT_COMPLETE":
                            raise ValueError("Resident sort failed")
                    sorted_at = time.monotonic()
                prefix = work / "frame"
                self.process.stdin.write((str(work / "scene.bin") + " " + str(prefix) + "\n").encode())
                self.process.stdin.flush()
                deadline = time.monotonic() + 180
                while True:
                    response = self._line(max(0, deadline - time.monotonic()))
                    if response == "RENDER_COMPLETE " + str(prefix):
                        break
                rendered = time.monotonic()
                record.update(attributes_ms=(projected-started)*1000,
                              group_sort_ms=(sorted_at-projected)*1000,
                              render_ms=(rendered-sorted_at)*1000,
                              scene_sha256=sha(work / "scene.bin"),
                              frame_sha256=sha(work / "frame.bin"))
                shutil.copy2(work / "frame.bin", out / "frame.bin")
                shutil.copy2(work / "frame_timing.csv", out / "frame_timing.csv")
                raw = (out / "frame.bin").read_bytes()
                width, height, self.last_rgb = framebuffer_rgb(raw)
                (out / "frame.ppm").write_bytes(
                    ("P6\n%d %d\n255\n" % (width, height)).encode() + self.last_rgb.tobytes())
                record["rgb_sha256"] = hashlib.sha256(self.last_rgb.tobytes()).hexdigest()
                record["frame_pack_ms"] = (time.monotonic() - rendered) * 1000
            record["complete"] = record["frame_sha256"] == sha(out / "frame.bin")
            if not record["complete"]:
                raise ValueError("Framebuffer hash changed")
        finally:
            record["total_ms"] = (time.monotonic() - started)*1000
            (out / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record

    def close(self):
        errors = []
        try:
            for process in (self.process, self.attributes, self.sorter):
                if process is None or process.stdin.closed:
                    continue
                try:
                    if process.poll() is None:
                        try:
                            process.stdin.write(b"QUIT\n")
                            process.stdin.flush()
                            process.wait(timeout=35)
                        except (OSError, subprocess.TimeoutExpired):
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait(timeout=5)
                except Exception as exc:
                    errors.append(exc)
                finally:
                    process.stdout.close()
                    process.stdin.close()
        finally:
            self.log.close()
        if errors:
            raise errors[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
