"""RGB PSNR and valid-window Gaussian SSIM; no GT for the synthetic novel pose."""
import argparse
import json
import struct
from pathlib import Path
import cv2
import numpy as np


def quality(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid image shape or values")
    mse = float(np.mean((a-b)**2))
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)[5:-5, 5:-5]
    m1, m2 = blur(a), blur(b)
    v1, v2 = blur(a*a)-m1*m1, blur(b*b)-m2*m2
    cov = blur(a*b)-m1*m2
    ssim = ((2*m1*m2+.01**2)*(2*cov+.03**2))/((m1*m1+m2*m2+.01**2)*(v1+v2+.03**2))
    return {"psnr_db": float(-10*np.log10(mse)) if mse else None,
            "ssim_rgb_gaussian_11_sigma1p5_valid": float(ssim.mean()),
            "mae": float(np.abs(a-b).mean()), "max_abs_error": float(np.abs(a-b).max())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path)
    a = p.parse_args()
    cfg = json.loads((a.run/"config.json").read_text())
    proj = json.loads((a.run/"project.json").read_text())
    name = proj["heldout_image"]
    gt = cv2.imread(str(a.run/"project/images"/name))[:,:,::-1]/255.
    native = cv2.imread(str(a.run/"validation"/(str(cfg["iterations"])+".png")))[:,:,::-1]/255.
    result = {"heldout": name, "heldout_scope": proj["heldout_scope"],
              "image_size": list(gt.shape), "native_cpu_vs_heldout": quality(native, gt),
              "lpips": "not measured", "power": "not measured", "novel_view_ground_truth": "unavailable"}
    panels = [gt, native]
    f = a.run/"board_cpu"/Path(name).stem/"frame.bin"
    if f.exists():
        raw = f.read_bytes()
        w, h = struct.unpack_from("<II", raw, 8)
        if raw[:8] != b"GSSOUT01" or len(raw) != 16+w*h*20:
            raise ValueError("Bad framebuffer ABI")
        pix = np.frombuffer(raw, dtype=np.dtype([("rgba", "<f4", 4), ("last", "<u4")]), offset=16)
        rgb = pix["rgba"][:,:3].reshape(h,w,3)
        result["board_unclipped_diagnostic"] = {"min": float(rgb.min()), "max": float(rgb.max()),
            "fraction_channels_above_one": float(np.mean(rgb > 1)), "vs_heldout": quality(rgb, gt)}
        # OpenSplat Model::forward clamps RGB to 1; the frozen renderer applies
        # this display clamp in write_ppm, rather than in its raw framebuffer.
        displayed = np.clip(rgb, 0, 1)
        result["board_cpu_vs_heldout"] = quality(displayed, gt)
        result["board_cpu_vs_native_quantized"] = quality(displayed, native)
        result["comparison_domain"] = "RGB [0,1], same display clamp; native reference is 8-bit PNG, board is float"
        panels.append(np.clip(rgb,0,1))
    image = np.concatenate(panels, axis=1)
    cv2.imwrite(str(a.run/"comparison.png"), np.rint(image[:,:,::-1]*255).astype(np.uint8))
    (a.run/"quality.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
