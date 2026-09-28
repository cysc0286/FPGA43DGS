"""Decode and select frames. Output: frames/ and frames.json."""
from pathlib import Path
import cv2
import numpy as np
from ..common import file_sha256, save
from ..contracts import FrameSet

def frames(a):
    root = a.run
    dst = root / "frames"
    dst.mkdir()
    if not hasattr(cv2, "CAP_PROP_N_THREADS"):
        raise RuntimeError("OpenCV with configurable FFmpeg decoder threads is required")
    cap = cv2.VideoCapture(str(a.video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, a.threads])
    if not cap.isOpened():
        raise RuntimeError("Cannot decode input video")
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    if n < 3 or fps <= 0:
        raise ValueError("A decodable multi-frame video is required")
    selected = set(np.linspace(0, n - 1, min(a.frames, n)).round().astype(int).tolist())
    records = []
    for i in range(n):
        ok, im = cap.read()
        if not ok:
            raise RuntimeError(f"Decode failed at frame {i}")
        if i not in selected:
            continue
        h, w = im.shape[:2]
        scale = min(1., a.width / w)
        im = cv2.resize(im, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        name = f"frame_{i:06d}.png"
        if not cv2.imwrite(str(dst / name), im):
            raise RuntimeError("Image write failed")
        records.append({"file": name, "video_frame": i, "time_s": i / fps,
                        "width": im.shape[1], "height": im.shape[0],
                        "sharpness_laplacian_variance": float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())})
    decoder_threads = cap.get(cv2.CAP_PROP_N_THREADS)
    cap.release()
    save(root / "frames.json", {"video": str(a.video.resolve()),
         "video_sha256": file_sha256(a.video), "fps": fps,
         "source_frames": n, "decoder_threads_reported": decoder_threads,
         "selection": "uniform frame indices, no learned model", "frames": records})
    return FrameSet.load(root)
