"""Frozen PLY, camera and framebuffer handoff validation; no board access."""
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from common import sha as file_sha256

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_files(root, names):
    for name in names:
        path = Path(root) / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("Missing or empty interface file: " + str(path))


def local_name(name):
    if not isinstance(name, str) or not name or Path(name).name != name or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError("Expected a single artifact filename")
    return name


@dataclass(frozen=True)
class RenderInput:
    model: Path
    camera: Path
    width: int
    height: int

    @classmethod
    def load(cls, directory, camera="novel_midpoint.bin"):
        directory = Path(directory).resolve()
        camera = local_name(camera)
        require_files(directory, ["model.ply", "manifest.json", camera])
        meta = read_json(directory / "manifest.json")
        if file_sha256(directory / "model.ply") != meta["model_sha256"]:
            raise ValueError("Renderer model hash mismatch")
        expected = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)] + [f"f_rest_{i}" for i in range(45)] + ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
        with (directory / "model.ply").open("rb") as f:
            lines = []
            for _ in range(128):
                line = f.readline(1024)
                lines.append(line.decode("ascii").strip())
                if lines[-1] == "end_header":
                    break
            else:
                raise ValueError("Invalid or oversized PLY header")
            elements = [v for v in lines if v.startswith("element ")]
            properties = [v for v in lines if v.startswith("property ")]
            if lines[:2] != ["ply", "format binary_little_endian 1.0"] or len(elements) != 1 or not elements[0].startswith("element vertex ") or properties != ["property float " + k for k in expected]:
                raise ValueError("Renderer PLY must use the frozen 62-float layout")
            count = int(elements[0].split()[-1])
            if not 0 < count <= 1000000 or count != meta["gaussians"] or (directory / "model.ply").stat().st_size != f.tell() + count * 248:
                raise ValueError("Renderer Gaussian count or payload length mismatch")
        raw = (directory / camera).read_bytes()
        if len(raw) != 136 or raw[:8] != b"FLCAM001":
            raise ValueError("Camera ABI must be FLCAM001, 136 bytes")
        w, h, sw, sh, *values = struct.unpack("<4I14d", raw[8:])
        if min(w, h, sw, sh) <= 0 or max(w, h, sw, sh) > 2048 or not all(math.isfinite(v) for v in values) or min(values[-2:]) <= 0:
            raise ValueError("Invalid camera dimensions, pose or intrinsics")
        rot = values[:9]
        if max(abs(sum(rot[3*k+i]*rot[3*k+j] for k in range(3)) - (i == j)) for i in range(3) for j in range(3)) > 1e-5:
            raise ValueError("Camera rotation must be orthonormal")
        determinant = rot[0]*(rot[4]*rot[8]-rot[5]*rot[7]) - rot[1]*(rot[3]*rot[8]-rot[5]*rot[6]) + rot[2]*(rot[3]*rot[7]-rot[4]*rot[6])
        if abs(determinant - 1) > 1e-5:
            raise ValueError("Camera rotation must preserve handedness")
        return cls(directory / "model.ply", directory / camera, w, h)


@dataclass(frozen=True)
class RenderResult:
    directory: Path
    width: int
    height: int

    @classmethod
    def load(cls, directory):
        directory = Path(directory).resolve()
        require_files(directory, ["frame.bin", "frame.ppm", "result.json"])
        meta = read_json(directory / "result.json")
        if not meta["complete"] or file_sha256(directory / "frame.bin") != meta["frame_sha256"]:
            raise ValueError("Incomplete or changed render result")
        with (directory / "frame.bin").open("rb") as f:
            header = f.read(16)
            if len(header) != 16 or header[:8] != b"GSSOUT01":
                raise ValueError("Framebuffer ABI must be GSSOUT01")
            w, h = struct.unpack("<II", header[8:])
            if not w or not h or (directory / "frame.bin").stat().st_size != 16 + 20*w*h:
                raise ValueError("Framebuffer dimensions/length mismatch")
        return cls(directory, w, h)
