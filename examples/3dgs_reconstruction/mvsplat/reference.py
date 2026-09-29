"""Independent NumPy renderer of world covariance outputs for bridge validation.

Uses ordered Gaussian alpha compositing, GraphDECO footprint/threshold semantics,
and SciPy's real SH basis. This is a CPU numerical reference, not the authors'
CUDA renderer or a speed baseline. See ../../3dgs_flicker_hw/frontend/LICENSE-GraphDECO.md
for the upstream rendering convention/license notice.
"""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from PIL import Image
from scipy.special import sph_harm_y

from common import load_gaussians, new_directory, save, sha


def real_sh_basis(directions, degree=4):
    d = np.asarray(directions, np.float64)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    theta, phi = np.arccos(np.clip(d[:, 2], -1, 1)), np.arctan2(d[:, 1], d[:, 0])
    values = []
    for l in range(degree+1):
        for m in range(-l, l+1):
            y = sph_harm_y(l, abs(m), theta, phi)
            values.append(y.real if m == 0 else np.sqrt(2) * (y.imag if m < 0 else y.real))
    return np.stack(values, axis=-1).astype(np.float32)


def project(g, view):
    w, h = view["width"], view["height"]
    pose = np.array(view["c2w"], np.float64)
    world_to_camera = np.linalg.inv(pose).astype(np.float32)
    k = np.array(view["intrinsics_normalized"], np.float32)
    fx, fy = k[0, 0]*w, k[1, 1]*h
    means = g["means"] @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    ids = np.flatnonzero(means[:, 2] > .2)
    t = means[ids]
    z = t[:, 2]
    center = t[:, :2]/(z[:, None]+1e-7)*np.array([fx, fy], np.float32) + np.array([(w-1)*.5, (h-1)*.5], np.float32)
    tx = np.clip(t[:, 0]/z, -1.3*w/(2*fx), 1.3*w/(2*fx))*z
    ty = np.clip(t[:, 1]/z, -1.3*h/(2*fy), 1.3*h/(2*fy))*z
    j = np.zeros((len(ids), 2, 3), np.float32)
    j[:, 0, 0], j[:, 1, 1] = fx/z, fy/z
    j[:, 0, 2], j[:, 1, 2] = -fx*tx/(z*z), -fy*ty/(z*z)
    transform = j @ world_to_camera[:3, :3]
    cov = transform @ g["covariances"][ids] @ transform.swapaxes(-1, -2)
    cov[:, 0, 0] += .3
    cov[:, 1, 1] += .3
    aa, bb, cc = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
    det = aa*cc-bb*bb
    if not np.isfinite(det).all() or np.any(det <= 0):
        raise ValueError("Invalid projected covariance")
    conic = np.stack([cc/det, -bb/det, aa/det], -1)
    mid = (aa+cc)*.5
    radius = np.ceil(3*np.sqrt(mid+np.sqrt(np.maximum(.1, mid*mid-det))))
    tile_dims = np.array([(w+15)//16, (h+15)//16])
    lo = np.clip(np.trunc((center-radius[:, None])/16), 0, tile_dims).astype(int)
    hi = np.clip(np.trunc((center+radius[:, None]+15)/16), 0, tile_dims).astype(int)
    basis = real_sh_basis(g["means"][ids]-pose[:3, 3])
    coefficients = g["harmonics"][ids]
    color3 = np.maximum(0, .5+np.einsum("ncj,nj->nc", coefficients[:, :, :16], basis[:, :16]))
    color4 = np.maximum(0, .5+np.einsum("ncj,nj->nc", coefficients, basis))
    order = np.lexsort((ids, z))
    return {key: value[order] for key, value in dict(center=center, conic=conic,
            opacity=g["opacities"][ids], rgb=np.concatenate([color3, color4], 1), lo=lo, hi=hi).items()}


def render(g, view):
    attrs = project(g, view)
    w, h = view["width"], view["height"]
    image = np.zeros((h, w, 6), np.float32)
    visited = accepted = 0
    for ty in range((h+15)//16):
        for tx in range((w+15)//16):
            indices = np.flatnonzero((attrs["lo"][:, 0] <= tx) & (tx < attrs["hi"][:, 0]) &
                                     (attrs["lo"][:, 1] <= ty) & (ty < attrs["hi"][:, 1]))
            yy, xx = np.mgrid[ty*16:min(h, ty*16+16), tx*16:min(w, tx*16+16)]
            pixels = np.stack([xx.reshape(-1), yy.reshape(-1)], -1).astype(np.float32)
            trans = np.ones(len(pixels), np.float32)
            rgb = np.zeros((len(pixels), 6), np.float32)
            done = np.zeros(len(pixels), bool)
            for start in range(0, len(indices), 64):
                active = np.flatnonzero(~done)
                if not len(active):
                    break
                ids = indices[start:start+64]
                d = attrs["center"][ids, None, :]-pixels[None, active, :]
                conic = attrs["conic"][ids]
                power = -.5*(conic[:, None, 0]*d[:, :, 0]**2 + conic[:, None, 2]*d[:, :, 1]**2)-conic[:, None, 1]*d[:, :, 0]*d[:, :, 1]
                alpha = np.minimum(.99, attrs["opacity"][ids, None]*np.exp(np.minimum(power, 0)))
                alpha[(power > 0) | (alpha < 1/255)] = 0
                through = np.cumprod(1-alpha, axis=0)
                next_trans = trans[active][None, :]*through
                accept = next_trans >= 1e-4
                before = np.concatenate([np.ones((1, len(active)), np.float32), through[:-1]], 0)*trans[active][None, :]
                weight = before*alpha*accept
                rgb[active] += weight.T @ attrs["rgb"][ids]
                # Keep the transmittance BEFORE the contribution that terminates
                # the ray, matching the reject-and-stop convention.
                trans[active] *= np.prod(1-np.where(accept, alpha, 0), axis=0)
                done[active] |= np.any(~accept, axis=0)
                visited += len(ids)*len(active)
                accepted += int(np.count_nonzero((alpha > 0) & accept))
            image[yy, xx] = rgb.reshape(*yy.shape, 6)
    return image[:, :, :3], image[:, :, 3:], dict(candidate_evaluations=visited, accepted_contributions=accepted)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--gaussians", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    out = new_directory(a.out)
    g = load_gaussians(a.gaussians)
    meta = json.loads((a.input/"input.json").read_text(encoding="utf-8"))
    results = {}
    for view in meta["views"]:
        if view["role"] != "target":
            continue
        start = time.perf_counter()
        im3, im4, info = render(g, view)
        stem = Path(view["name"]).stem
        np.savez(out/(stem+".npz"), sh3=im3, sh4=im4)
        for label, image in (("sh3", im3), ("sh4", im4)):
            Image.fromarray(np.rint(np.clip(image, 0, 1)*255).astype(np.uint8)).save(out/(stem+"_"+label+".png"))
        results[stem] = dict(seconds=time.perf_counter()-start, **info)
        print(stem, results[stem], flush=True)
        save(out/"reference.json", dict(gaussian_sha256=sha(a.gaussians), views=results,
             implementation="Independent NumPy world-covariance/SH reference; not official CUDA output"))


if __name__ == "__main__":
    main()
