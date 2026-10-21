"""Image-level probe: render CPU and NPU Gaussians through one identical
NumPy splatter and score both against the held-out frame.

Why a new renderer instead of mvsplat/reference.py:
  * reference.py imports PIL at module level, and neither arm_env nor deps
    ships Pillow on this board.
  * reference.py always evaluates the SH4 branch with a (n,3,225) harmonics
    array against a 25-wide basis, so it raises before returning any image.
  * the renderer ABI truncates to SH3 (16 coefficients) anyway, so only the
    SH3 path is meaningful for the NPU comparison.

Both CPU and NPU Gaussians go through the exact same code path here, so any
difference in the score comes from the Gaussians, not from the renderer.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.special import sph_harm_y

from common import load_gaussians


def real_sh_basis(directions, degree=3):
    """Same convention as mvsplat/reference.py, limited to the SH3 block."""
    d = np.asarray(directions, np.float64)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    theta = np.arccos(np.clip(d[:, 2], -1, 1))
    phi = np.arctan2(d[:, 1], d[:, 0])
    values = []
    for l in range(degree + 1):
        for m in range(-l, l + 1):
            y = sph_harm_y(l, abs(m), theta, phi)
            values.append(y.real if m == 0 else np.sqrt(2) * (y.imag if m < 0 else y.real))
    return np.stack(values, axis=-1).astype(np.float32)


def project(g, view):
    w, h = view["width"], view["height"]
    pose = np.array(view["c2w"], np.float64)
    w2c = np.linalg.inv(pose).astype(np.float32)
    k = np.array(view["intrinsics_normalized"], np.float32)
    fx, fy = k[0, 0] * w, k[1, 1] * h

    means = g["means"] @ w2c[:3, :3].T + w2c[:3, 3]
    ids = np.flatnonzero(means[:, 2] > 0.2)
    t = means[ids]
    z = t[:, 2]
    center = t[:, :2] / (z[:, None] + 1e-7) * np.array([fx, fy], np.float32) \
        + np.array([(w - 1) * 0.5, (h - 1) * 0.5], np.float32)
    tx = np.clip(t[:, 0] / z, -1.3 * w / (2 * fx), 1.3 * w / (2 * fx)) * z
    ty = np.clip(t[:, 1] / z, -1.3 * h / (2 * fy), 1.3 * h / (2 * fy)) * z
    j = np.zeros((len(ids), 2, 3), np.float32)
    j[:, 0, 0], j[:, 1, 1] = fx / z, fy / z
    j[:, 0, 2], j[:, 1, 2] = -fx * tx / (z * z), -fy * ty / (z * z)
    transform = j @ w2c[:3, :3]
    cov = transform @ g["covariances"][ids] @ transform.swapaxes(-1, -2)
    cov[:, 0, 0] += 0.3
    cov[:, 1, 1] += 0.3
    a, b, c = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
    det = a * c - b * b
    if not np.isfinite(det).all() or np.any(det <= 0):
        raise ValueError("Invalid projected covariance")
    conic = np.stack([c / det, -b / det, a / det], -1)
    mid = (a + c) * 0.5
    radius = np.ceil(3 * np.sqrt(mid + np.sqrt(np.maximum(0.1, mid * mid - det))))
    basis = real_sh_basis(g["means"][ids] - pose[:3, 3])
    rgb = np.maximum(0, 0.5 + np.einsum("ncj,nj->nc",
                                        g["harmonics"][ids][:, :, :16], basis[:, :16]))
    order = np.argsort(-z)
    return dict(center=center[order], conic=conic[order],
                opacity=g["opacities"][ids][order], rgb=rgb[order],
                radius=radius[order], depth=z[order])


def render(g, view, max_radius=64.0, max_gaussians=None):
    """Far-to-near alpha compositing; identical for CPU and NPU input."""
    attrs = project(g, view)
    w, h = view["width"], view["height"]
    image = np.zeros((h, w, 3), np.float32)
    n = len(attrs["center"])
    if max_gaussians:
        n = min(n, max_gaussians)
    used = 0
    for i in range(n):
        cx, cy = attrs["center"][i]
        r = min(float(attrs["radius"][i]), max_radius)
        if not np.isfinite(r) or r < 0.5:
            continue
        x0, x1 = int(max(0, np.floor(cx - r))), int(min(w, np.ceil(cx + r) + 1))
        y0, y1 = int(max(0, np.floor(cy - r))), int(min(h, np.ceil(cy + r) + 1))
        if x0 >= x1 or y0 >= y1:
            continue
        A, B, C = attrs["conic"][i]
        xs = np.arange(x0, x1, dtype=np.float32)[None, :] - cx
        ys = np.arange(y0, y1, dtype=np.float32)[:, None] - cy
        power = -0.5 * (A * xs * xs + C * ys * ys) - B * xs * ys
        alpha = np.minimum(0.99, attrs["opacity"][i] * np.exp(np.minimum(power, 0)))
        alpha[(power > 0) | (alpha < 1 / 255)] = 0
        if not alpha.any():
            continue
        patch = image[y0:y1, x0:x1]
        patch *= (1 - alpha)[..., None]
        patch += alpha[..., None] * attrs["rgb"][i]
        used += 1
    return np.clip(image, 0, 1), used


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse <= 0 else float(10 * np.log10(1.0 / mse))


def ssim(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    scores = []
    for ch in range(a.shape[2]):
        x, y = a[:, :, ch], b[:, :, ch]
        mx = cv2.GaussianBlur(x, (11, 11), 1.5)
        my = cv2.GaussianBlur(y, (11, 11), 1.5)
        sxx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mx * mx
        syy = cv2.GaussianBlur(y * y, (11, 11), 1.5) - my * my
        sxy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mx * my
        num = (2 * mx * my + c1) * (2 * sxy + c2)
        den = (mx * mx + my * my + c1) * (sxx + syy + c2)
        scores.append(float((num / den).mean()))
    return float(np.mean(scores))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--cpu", type=Path, required=True)
    p.add_argument("--npu", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-views", type=int, default=0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((a.input / "input.json").read_text(encoding="utf-8"))
    cpu = load_gaussians(a.cpu)
    npu = load_gaussians(a.npu)
    print("gaussians cpu=%d npu=%d" % (len(cpu["means"]), len(npu["means"])), flush=True)

    targets = [v for v in meta["views"] if v["role"] == "target"]
    if a.max_views:
        targets = targets[:a.max_views]

    result = dict(scope="identical NumPy SH3 splatter for both sources; "
                       "not FPGA, not official CUDA renderer",
                  gaussians=dict(cpu=len(cpu["means"]), npu=len(npu["means"])),
                  views={})
    for view in targets:
        stem = Path(view["name"]).stem
        truth = cv2.cvtColor(cv2.imread(str(a.input / "images" / view["name"]),
                                        cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) / 255.0
        truth = truth.astype(np.float32)
        im_cpu, n_cpu = render(cpu, view)
        im_npu, n_npu = render(npu, view)
        rec = dict(
            cpu_vs_truth=dict(psnr_db=psnr(im_cpu, truth), ssim=ssim(im_cpu, truth)),
            npu_vs_truth=dict(psnr_db=psnr(im_npu, truth), ssim=ssim(im_npu, truth)),
            npu_vs_cpu=dict(psnr_db=psnr(im_npu, im_cpu), ssim=ssim(im_npu, im_cpu)),
            gaussians_drawn=dict(cpu=n_cpu, npu=n_npu),
        )
        result["views"][stem] = rec
        for tag, img in (("truth", truth), ("cpu", im_cpu), ("npu", im_npu)):
            cv2.imwrite(str(a.out / (stem + "_" + tag + ".png")),
                        cv2.cvtColor(np.rint(img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        panel = np.concatenate([np.rint(x * 255).astype(np.uint8)
                                for x in (truth, im_cpu, im_npu)], axis=1)
        cv2.imwrite(str(a.out / (stem + "_panel.png")),
                    cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        print(stem, json.dumps(rec), flush=True)

    (a.out / "image_probe.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
