"""Convert world covariances and activated MVSplat outputs into frozen PLY ABI."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

from common import load_gaussians, new_directory, save, sha, validate_gaussians

# The shared renderer contract lives above the mvsplat package.  Put it on the
# import path before either the full export or deferred-camera path validates
# camera files.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _matrix_to_quaternion_wxyz(matrices):
    """Convert a batch of proper 3x3 rotations to scalar-first quaternions.

    ``scipy.spatial.transform.Rotation.from_matrix`` is convenient, but its
    per-call Python/adapter cost is visible on the ARM board during the first
    frame.  This vectorized implementation follows the standard trace/最大对角
    branch and keeps the calculation in float64 until the serialized rows are
    formed.  The returned sign is deterministic for a given matrix and is
    equivalent for covariance reconstruction.
    """
    m = np.asarray(matrices, dtype=np.float64)
    if m.ndim != 3 or m.shape[1:] != (3, 3):
        raise ValueError("Expected a batch of 3x3 rotation matrices")
    n = m.shape[0]
    q = np.empty((n, 4), dtype=np.float64)
    trace = np.trace(m, axis1=1, axis2=2)
    positive = trace > 0.0
    if np.any(positive):
        rows = np.flatnonzero(positive)
        s = 2.0 * np.sqrt(np.maximum(trace[rows] + 1.0, 1e-30))
        mm = m[rows]
        q[rows, 0] = 0.25 * s
        q[rows, 1] = (mm[:, 2, 1] - mm[:, 1, 2]) / s
        q[rows, 2] = (mm[:, 0, 2] - mm[:, 2, 0]) / s
        q[rows, 3] = (mm[:, 1, 0] - mm[:, 0, 1]) / s

    remaining = np.flatnonzero(~positive)
    if remaining.size:
        diagonals = np.stack((m[remaining, 0, 0], m[remaining, 1, 1],
                              m[remaining, 2, 2]), axis=1)
        branches = np.argmax(diagonals, axis=1)
        for branch in range(3):
            rows = remaining[branches == branch]
            if not rows.size:
                continue
            mm = m[rows]
            if branch == 0:
                s = 2.0 * np.sqrt(np.maximum(1.0 + mm[:, 0, 0] -
                                              mm[:, 1, 1] - mm[:, 2, 2], 1e-30))
                q[rows, 0] = (mm[:, 2, 1] - mm[:, 1, 2]) / s
                q[rows, 1] = 0.25 * s
                q[rows, 2] = (mm[:, 0, 1] + mm[:, 1, 0]) / s
                q[rows, 3] = (mm[:, 0, 2] + mm[:, 2, 0]) / s
            elif branch == 1:
                s = 2.0 * np.sqrt(np.maximum(1.0 - mm[:, 0, 0] +
                                              mm[:, 1, 1] - mm[:, 2, 2], 1e-30))
                q[rows, 0] = (mm[:, 0, 2] - mm[:, 2, 0]) / s
                q[rows, 1] = (mm[:, 0, 1] + mm[:, 1, 0]) / s
                q[rows, 2] = 0.25 * s
                q[rows, 3] = (mm[:, 1, 2] + mm[:, 2, 1]) / s
            else:
                s = 2.0 * np.sqrt(np.maximum(1.0 - mm[:, 0, 0] -
                                              mm[:, 1, 1] + mm[:, 2, 2], 1e-30))
                q[rows, 0] = (mm[:, 1, 0] - mm[:, 0, 1]) / s
                q[rows, 1] = (mm[:, 0, 2] + mm[:, 2, 0]) / s
                q[rows, 2] = (mm[:, 1, 2] + mm[:, 2, 1]) / s
                q[rows, 3] = 0.25 * s
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-30)
    return q


def factor_covariances(covariances, quaternion_backend="vectorized"):
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
    if quaternion_backend == "vectorized":
        wxyz = _matrix_to_quaternion_wxyz(vectors)
    elif quaternion_backend == "scipy":
        xyzw = Rotation.from_matrix(vectors).as_quat()
        wxyz = xyzw[:, [3, 0, 1, 2]]
    else:
        raise ValueError("quaternion_backend must be 'vectorized' or 'scipy'")
    return np.log(np.sqrt(values)).astype(np.float32), wxyz.astype(np.float32), int(clamped.sum())


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


def gaussian_rows(gaussians):
    g = validate_gaussians(gaussians)
    n = len(g["means"])
    scales, rotations, eigen_clamps = factor_covariances(g["covariances"])
    opacity = np.clip(g["opacities"].astype(np.float64), 1e-7, 1-1e-7)
    rows = np.zeros((n, 62), dtype="<f4")
    rows[:, :3] = g["means"]
    rows[:, 6:9] = g["harmonics"][:, :, 0]
    rows[:, 9:54] = g["harmonics"][:, :, 1:16].reshape(n, 45)
    rows[:, 54] = np.log(opacity / (1-opacity))
    rows[:, 55:58] = scales
    rows[:, 58:62] = rotations
    if not np.isfinite(rows).all():
        raise ValueError("Nonfinite exported Gaussian")
    return rows, eigen_clamps


def main(argv=None, gaussians=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--input-manifest", type=Path,
                   help="Optional validated input JSON (for the first-target snapshot)")
    p.add_argument("--append-cameras", action="store_true",
                   help="Append deferred camera files to an already validated export")
    p.add_argument("--inference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    t = time.perf_counter()
    result = json.loads((a.inference / "inference.json").read_text())
    if not result["complete"] or sha(a.inference / "gaussians.npz") != result["gaussian_sha256"]:
        raise ValueError("Inference not validated or output changed")
    if sha(a.input / "context.npz") != result["input_sha256"]:
        raise ValueError("Camera/input provenance mismatch")
    meta_path = a.input / "input.json" if a.input_manifest is None else a.input_manifest
    if meta_path.parent.resolve() != a.input.resolve():
        raise ValueError("Input manifest must be inside the prepared input directory")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("context_sha256") != result["input_sha256"]:
        raise ValueError("Input manifest/context provenance mismatch")
    if a.append_cameras:
        out = a.out
        manifest_path = out / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("Cannot append cameras to a missing export")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["input_sha256"] != result["input_sha256"] or sha(out / "model.ply") != manifest["model_sha256"]:
            raise ValueError("Existing renderer export provenance failed")
        known = {camera["file"]: camera for camera in manifest["cameras"]}
        for view in meta["views"]:
            name = Path(view["name"]).stem + ".bin"
            camera_path = out / name
            camera = {**view, "file": name}
            if name in known:
                if known[name]["sha256"] != sha(camera_path):
                    raise ValueError("Existing camera changed: " + name)
                continue
            camera_path.write_bytes(camera_bytes(view))
            camera["sha256"] = sha(camera_path)
            known[name] = camera
        # Re-open every camera through the same runtime contract used by the
        # initial export. This keeps deferred append validation equivalent to
        # the normal path instead of trusting only the file hash.
        from modules.contracts import RenderInput
        for camera in known.values():
            RenderInput.load(out, camera["file"])
        manifest["cameras"] = list(known.values())
        manifest["camera_scope"] = meta.get("target_scope", "full prepared input")
        save(manifest_path, manifest)
        checks = dict(complete=True, appended_cameras=len(manifest["cameras"]),
                      gaussians=manifest["gaussians"], cameras=len(manifest["cameras"]),
                      seconds=time.perf_counter()-t)
        save(out / "adapter_validation.json", checks)
        print(json.dumps(checks, indent=2))
        return checks
    g = load_gaussians(a.inference / "gaussians.npz") if gaussians is None else validate_gaussians(gaussians)
    rows, eigen_clamps = gaussian_rows(g)
    n = len(rows)
    props = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
    props += [f"f_rest_{i}" for i in range(45)] + ["opacity"]
    props += [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    rows_sha256 = hashlib.sha256(rows.tobytes()).hexdigest()
    out = new_directory(a.out)
    vertices = rows.view(np.dtype([(key, "<f4") for key in props])).reshape(n)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False, byte_order="<").write(out / "model.ply")
    # Validate serialized bytes, not just the pre-serialization working arrays.
    restored = PlyData.read(out / "model.ply")["vertex"].data.copy().view("<f4").reshape(n, 62)
    if not np.array_equal(restored, rows):
        raise ValueError("PLY payload differs from validated Gaussian rows")
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
    cameras = []
    for view in meta["views"]:
        name = Path(view["name"]).stem + ".bin"
        (out / name).write_bytes(camera_bytes(view))
        cameras.append({**view, "file": name, "sha256": sha(out/name)})
    save(out / "manifest.json", dict(schema="mvsplat_renderer_bridge_v1", gaussians=n,
         model_sha256=sha(out / "model.ply"), gaussian_sha256=result["gaussian_sha256"],
         rows_sha256=rows_sha256,
         input_sha256=result["input_sha256"], source_sh_degree=4, exported_sh_degree=3,
         cameras=cameras, quality_scope="SH degree 4 truncated to 3; validate image impact separately"))
    from modules.contracts import RenderInput
    for cam in cameras:
        RenderInput.load(out, cam["file"])
    checks.update(complete=True, gaussians=n, cameras=len(cameras), seconds=time.perf_counter()-t,
                  gaussian_handoff="file" if gaussians is None else "validated_memory")
    save(out / "adapter_validation.json", checks)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
