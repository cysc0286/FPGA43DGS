"""Pack the best NPU-vs-CPU comparison into one compact JPEG for transfer."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

src = Path(sys.argv[1])
out = Path(sys.argv[2])
view = sys.argv[3] if len(sys.argv) > 3 else "frame_000007"

truth = cv2.imread(str(src / (view + "_truth.png")))
cpu = cv2.imread(str(src / (view + "_cpu.png")))
npu = cv2.imread(str(src / (view + "_npu.png")))
h, w = truth.shape[:2]

diff = np.abs(npu.astype(np.int16) - cpu.astype(np.int16)).max(axis=2)
amp = np.clip(diff.astype(np.float32) * 10.0, 0, 255).astype(np.uint8)
heat = cv2.applyColorMap(amp, cv2.COLORMAP_INFERNO)

pad = 6
label_h = 20
canvas = np.full((2 * (h + label_h) + pad * 3, 3 * w + pad * 4, 3), 245, np.uint8)

def put(img, row, col, text):
    y = pad + row * (h + label_h + pad)
    x = pad + col * (w + pad)
    canvas[y:y + h, x:x + w] = img
    cv2.putText(canvas, text, (x + 2, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (30, 30, 30), 1, cv2.LINE_AA)

put(truth, 0, 0, "held-out real frame")
put(cpu, 0, 1, "CPU gaussians")
put(npu, 0, 2, "NPU gaussians (to_gaussians)")
put(heat, 1, 1, "|NPU-CPU| x10")

probe = json.loads((src / "image_probe.json").read_text())
rec = probe["views"][view]
cv2.putText(canvas, "NPU vs CPU: %.1f dB  |  NPU vs truth: %.2f dB" % (
    rec["npu_vs_cpu"]["psnr_db"], rec["npu_vs_truth"]["psnr_db"]),
    (pad + 2, 2 * (h + label_h) + pad * 2 + 12),
    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)

out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
print("wrote", out, out.stat().st_size, "bytes", canvas.shape)
