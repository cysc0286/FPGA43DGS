"""Convert world covariances and activated MVSplat outputs into frozen PLY ABI."""
import argparse
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

from common import load_gaussians, new_directory, save, sha


def factor_covariances(covariances):
    cov = np.asarray(covariances, np.float64)
    sym = (cov + cov.swapaxes(-1, -2)) * .5
    values, vectors = np.linalg.eigh(sym)
    tolerance = np.maximum(values[:, -1] * 2e-6, 1e-20)
    if np.any(values[:, 0] < -tolerance) or np.any(values[:, -1] <= 0):
        raise ValueError("Covariance is not positive semidefinite")
    clamped = values <= 0
    values = np.maximum(values, np.maximum(values[:, -1:] * 1e-12, 1e-30))
    # Eigenvectors may form a reflection. Flip an eigenvector to recover SO(3).
    vectors[:, :, 0] *= np.where(np.linalg.det(vectors) < 0, -1., 1.)[:, None]
    xyzw = Rotation.from_matrix(vectors).as_quat()
    return np.log(np.sqrt(values)).astype(np.float32), xyzw[:, [3, 0, 1, 2]].astype(np.float32), int(clamped.sum())


def recover_covariances(log_scales, quaternion_wxyz):
    r = Rotation.from_quat(np.asarray(quaternion_wxyz, np.float64)[:, [1, 2, 3, 0]]).as_matrix()
    variances = np.exp(2 * np.asarray(log_scales, np.float64))
    return (r * variances[:, None, :]) @ r.swapaxes(-1, -2)


def camera_bytes(view):
    c2w = np.array(view["c2w"], dtype=np.float64)
    k = np.array(view["intrinsics_normalized"], dtype=np.float64)
    if not np.allclose(k[:2, 2], .5, atol=1e-8) or abs(k[0, 1]) > 1e-8:
        raise ValueError("Frozen renderer requires centered intrinsics and zero skew")
    w, h = view["width"], view["height"]
    return b"FLCAM001" + struct.pack("<4I14d", w, h, w, h,
             *c2w[:3, :3].reshape(-1), *c2w[:3, 3], k[0, 0] * w, k[1, 1] * h)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--inference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    t = time.perf_counter()
    result = json.loads((a.inference / "inference.json").read_text())
    if not result["complete"] or sha(a.inference / "gaussians.npz") != result["gaussian_sha256"]:
        raise ValueError("Inference not validated or output changed")
    if sha(a.input / "context.npz") != result["input_sha256"]:
        raise ValueError("Camera/input provenance mismatch")
    g = load_gaussians(a.inference / "gaussians.npz")
    n = len(g["means"])
    scales, rotations, eigen_clamps = factor_covariances(g["covariances"])
    opacity = np.clip(g["opacities"].astype(np.float64), 1e-7, 1-1e-7)
    props = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
    props += [f"f_rest_{i}" for i in range(45)] + ["opacity"]
    props += [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    rows = np.zeros((n, 62), dtype="<f4")
    rows[:, :3] = g["means"]
    rows[:, 6:9] = g["harmonics"][:, :, 0]
    rows[:, 9:54] = g["harmonics"][:, :, 1:16].reshape(n, 45)
    rows[:, 54] = np.log(opacity / (1-opacity))
    rows[:, 55:58] = scales
    rows[:, 58:62] = rotations
    if not np.isfinite(rows).all():
        raise ValueError("Nonfinite exported Gaussian")
    out = new_directory(a.out)
    vertices = rows.view(np.dtype([(key, "<f4") for key in props])).reshape(n)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False, byte_order="<").write(out / "model.ply")
    # Validate serialized bytes, not just the pre-serialization working arrays.
    restored = PlyData.read(out / "model.ply")["vertex"].data.copy().view("<f4").reshape(n, 62)
    cov_back = recover_covariances(restored[:, 55:58], restored[:, 58:62])
    cov_norm = np.maximum(np.linalg.norm(g["covariances"], axis=(1, 2)), 1e-30)
    relative = np.linalg.norm(cov_back-g["covariances"], axis=(1, 2))/cov_norm
    alpha_back = 1/(1+np.exp(-restored[:, 54].astype(np.float64)))
    checks = dict(world_covariance_relative_frobenius_max=float(relative.max()),
                  covariance_eigenvalues_clamped=eigen_clamps,
                  opacity_max_abs_error=float(np.max(abs(alpha_back-g["opacities"]))),
                  means_bitwise_preserved=bool(np.array_equal(restored[:, :3], g["means"])),
                  sh0_to_sh3_bitwise_preserved=bool(np.array_equal(restored[:, 6:9], g["harmonics"][:, :, 0]) and
                      np.array_equal(restored[:, 9:54], g["harmonics"][:, :, 1:16].reshape(n, 45))),
                  source_sh_degree=4, exported_sh_degree=3,
                  sh4_coefficient_max_abs=float(np.max(abs(g["harmonics"][:, :, 16:]))))
    if relative.max() > 1e-5 or checks["opacity_max_abs_error"] > 2e-7 or not checks["means_bitwise_preserved"] or not checks["sh0_to_sh3_bitwise_preserved"]:
        raise ValueError("World-space Gaussian serialization contract failed")
    meta = json.loads((a.input / "input.json").read_text(encoding="utf-8"))
    cameras = []
    for view in meta["views"]:
        name = Path(view["name"]).stem + ".bin"
        (out / name).write_bytes(camera_bytes(view))
        cameras.append({**view, "file": name, "sha256": sha(out/name)})
    save(out / "manifest.json", dict(schema="mvsplat_renderer_bridge_v1", gaussians=n,
         model_sha256=sha(out / "model.ply"), gaussian_sha256=result["gaussian_sha256"],
         input_sha256=result["input_sha256"], source_sh_degree=4, exported_sh_degree=3,
         cameras=cameras, quality_scope="SH degree 4 truncated to 3; validate image impact separately"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from modules.contracts import RenderInput
    for cam in cameras:
        RenderInput.load(out, cam["file"])
    checks.update(complete=True, gaussians=n, cameras=len(cameras), seconds=time.perf_counter()-t)
    save(out / "adapter_validation.json", checks)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
