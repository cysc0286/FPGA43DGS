"""CPU Gaussian optimization and export to the frozen renderer ABI."""
import json
from pathlib import Path
from ..common import file_sha256, save
from ..contracts import GaussianScene, PoseSet, RenderInput

def export(a):
    import numpy as np
    import pycolmap as p
    from plyfile import PlyData, PlyElement
    import struct
    root = a.run
    PoseSet.load(root)
    GaussianScene.load(root)
    out = root / "renderer_input"
    out.mkdir()
    src = PlyData.read(root / "splat.ply")["vertex"].data
    props = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)] + [f"f_rest_{i}" for i in range(45)] + ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    dst = np.zeros(len(src), dtype=[(name, "<f4") for name in props])
    required = [k for k in props if not k.startswith("f_rest_") and k not in ("nx", "ny", "nz")]
    if not set(required) <= set(src.dtype.names):
        raise ValueError("Missing Gaussian properties")
    rest = sorted((k for k in src.dtype.names if k.startswith("f_rest_")), key=lambda s: int(s[7:]))
    per_color = len(rest) // 3
    if per_color not in (0, 3, 8, 15) or len(rest) != per_color * 3:
        raise ValueError("Unknown SH layout")
    for key in props:
        if key in src.dtype.names and not key.startswith("f_rest_"):
            dst[key] = src[key]
    for color in range(3):
        for j in range(per_color):
            dst[f"f_rest_{color * 15 + j}"] = src[f"f_rest_{color * per_color + j}"]
    arr = dst.view("<f4").reshape(len(dst), 62)
    if not (0 < len(dst) <= 1000000) or not np.isfinite(arr).all():
        raise ValueError("Invalid model")
    PlyData([PlyElement.describe(dst, "vertex")], text=False, byte_order="<").write(out / "model.ply")
    rec = p.Reconstruction(root / "project" / "sparse" / "0")
    heldout = json.loads((root / "project.json").read_text())["heldout_image"]
    cameras = []
    for im in sorted(rec.images.values(), key=lambda im: im.name):
        cam = rec.cameras[im.camera_id]
        c2w = im.cam_from_world().inverse()
        rot, pos = c2w.rotation.matrix(), c2w.translation
        vals = [*rot.flatten().tolist(), *pos.tolist(), cam.focal_length_x, cam.focal_length_y]
        blob = b"FLCAM001" + struct.pack("<4I14d", cam.width, cam.height, cam.width, cam.height, *vals)
        (out / (Path(im.name).stem + ".bin")).write_bytes(blob)
        cameras.append({"name": im.name, "heldout": im.name == heldout, "width": cam.width, "height": cam.height,
                        "rotation": rot.tolist(), "position": pos.tolist(), "fx": cam.focal_length_x, "fy": cam.focal_length_y})
    # Novel view between the two middle registered poses. No ground truth exists for this exact pose.
    left, right = cameras[len(cameras)//2-1:len(cameras)//2+1]
    u, _, vh = np.linalg.svd(np.array(left["rotation"]) + np.array(right["rotation"]))
    rot = u @ np.diag([1., 1., np.linalg.det(u @ vh)]) @ vh
    pos = (np.array(left["position"]) + np.array(right["position"])) / 2
    vals = [*rot.flatten().tolist(), *pos.tolist(), left["fx"], left["fy"]]
    (out / "novel_midpoint.bin").write_bytes(b"FLCAM001" + struct.pack("<4I14d", left["width"], left["height"], left["width"], left["height"], *vals))
    save(out / "manifest.json", {"gaussians": len(src), "source_sh_rest_per_color": per_color,
         "source_ply_sha256": file_sha256(root/"splat.ply"),
         "model_sha256": file_sha256(out/"model.ply"),
         "cameras": cameras, "novel_view": "rotation projected to SO(3), position midpoint; no exact GT"})
    validate_adapter(a)
    return RenderInput.load(out)

def validate_adapter(a):
    """Cross-check exported ABI against independent OpenSplat camera/PLY exports."""
    import numpy as np
    from plyfile import PlyData
    root = a.run
    meta = json.loads((root/"renderer_input/manifest.json").read_text())
    native = {c["img_name"]: c for c in json.loads((root/"cameras.json").read_text())}
    rot_errors, pos_errors = [], []
    for c in meta["cameras"]:
        n = native[c["name"]]
        rot_errors.append(float(np.max(np.abs(np.array(c["rotation"])-np.array(n["rotation"])))))
        pos_errors.append(float(np.max(np.abs(np.array(c["position"])-np.array(n["position"])))))
        if (c["width"], c["height"]) != (n["width"], n["height"]) or abs(c["fx"]-n["fx"]) > 1e-3 or abs(c["fy"]-n["fy"]) > 1e-3:
            raise ValueError("Intrinsics changed across model/camera export")
    src = PlyData.read(root/"splat.ply")["vertex"].data
    dst = PlyData.read(root/"renderer_input/model.ply")["vertex"].data
    exact = all(np.array_equal(src[k], dst[k]) for k in src.dtype.names if not k.startswith("f_rest_"))
    per_color = meta["source_sh_rest_per_color"]
    exact_sh = all(np.array_equal(src[f"f_rest_{ch*per_color+j}"], dst[f"f_rest_{ch*15+j}"])
                   for ch in range(3) for j in range(per_color))
    zero = all(np.all(dst[f"f_rest_{ch*15+j}"] == 0) for ch in range(3) for j in range(per_color, 15))
    finite = bool(np.isfinite(dst.view("<f4")).all())
    if not (exact and exact_sh and zero and finite and max(rot_errors) < 1e-5 and max(pos_errors) < 1e-5):
        raise ValueError("Gaussian or coordinate contract mismatch")
    save(root/"adapter_validation.json", {"camera_count": len(rot_errors),
         "max_rotation_abs_error_vs_opensplat_export": max(rot_errors),
         "max_position_abs_error_vs_opensplat_export": max(pos_errors),
         "source_gaussian_fields_preserved_bitwise": exact and exact_sh,
         "padded_sh_coefficients_all_zero": zero, "all_finite": finite, "gaussians": len(dst)})
