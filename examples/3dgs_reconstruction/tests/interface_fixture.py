"""Small synthetic file-contract fixture; not reconstruction or quality evidence."""
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib


def make_reference(directory):
    root = Path(directory)
    for name in ("frames", "project/images", "project/sparse/0", "renderer_input"):
        (root/name).mkdir(parents=True)
    width, height, count = 160, 120, 30

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind+data))

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">2I5B", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress((b"\0" + b"\x40\x60\x80"*width)*height)) + chunk(b"IEND", b"")
    records = []
    images = bytearray(struct.pack("<Q", count))
    tracks = bytearray()
    native_cameras = []
    for i in range(count):
        name = f"frame_{i:06d}.png"
        (root/"frames"/name).write_bytes(png)
        (root/"project/images"/name).write_bytes(png)
        records.append(dict(file=name, video_frame=i, time_s=i/10, width=width, height=height))
        tx = -i*.01
        images += struct.pack("<i7di", i+1, 1, 0, 0, 0, tx, 0, 0, 1) + name.encode() + b"\0"
        images += struct.pack("<Qddq", 1, 79.5 + 100*tx/3, 59.5, 1)
        tracks += struct.pack("<ii", i+1, 0)
        native_cameras.append(dict(id=i, img_name=name, width=width, height=height, fx=100, fy=100,
                                   rotation=[[1,0,0],[0,1,0],[0,0,1]], position=[-tx,0,0]))
    (root/"frames.json").write_text(json.dumps(dict(scope="synthetic contract fixture", frames=records)))
    camera = struct.pack("<QiiQQ4d", 1, 1, 1, width, height, 100, 100, 79.5, 59.5)
    sparse = root/"project/sparse/0"
    (sparse/"cameras.bin").write_bytes(camera)
    (sparse/"images.bin").write_bytes(images)
    point = struct.pack("<QQ3d3BdQ", 1, 1, 0, 0, 3, 64, 96, 128, 0, count) + tracks
    (sparse/"points3D.bin").write_bytes(point)
    (root/"project.json").write_text(json.dumps(dict(image_count=count, heldout_image="frame_000015.png",
        cameras={"1":dict(width=width, height=height, params=[100,100,79.5,59.5])})))
    (root/"sfm.json").write_text(json.dumps(dict(registered=count, input_images=count, points3D=1)))
    props = ["x","y","z","nx","ny","nz"] + [f"f_dc_{i}" for i in range(3)]
    props += [f"f_rest_{i}" for i in range(45)] + ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    values = {name:0.0 for name in props}
    values.update(z=3.0, rot_0=1.0, scale_0=math.log(.1), scale_1=math.log(.1), scale_2=math.log(.1))
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
    header += "".join("property float "+name+"\n" for name in props) + "end_header\n"
    model = header.encode() + struct.pack("<62f", *(values[name] for name in props))
    (root/"splat.ply").write_bytes(model)
    (root/"cameras.json").write_text(json.dumps(native_cameras))
    output = root/"renderer_input"
    (output/"model.ply").write_bytes(model)
    (output/"manifest.json").write_text(json.dumps(dict(gaussians=1, model_sha256=hashlib.sha256(model).hexdigest())))
    (output/"novel_midpoint.bin").write_bytes(b"FLCAM001" + struct.pack("<4I14d",
        width,height,width,height, 1,0,0,0,1,0,0,0,1, .15,0,0, 100,100))
    return root
