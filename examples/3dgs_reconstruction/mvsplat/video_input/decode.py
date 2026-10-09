"""Decode selected video frames; no feature matching or pose estimation."""
import cv2


def frame_indices(count, context, targets):
    context = [i if i >= 0 else count + i for i in context]
    targets = ([count // 4, count // 2, 3 * count // 4] if targets is None
               else [i if i >= 0 else count + i for i in targets])
    selected = context + targets
    if len(context) != 2 or not targets or len(set(selected)) != len(selected):
        raise ValueError("Use two distinct context frames and at least one distinct target")
    if min(selected) < 0 or max(selected) >= count:
        raise ValueError("Selected frame outside video")
    return context, targets


def decode_selected(video, context, targets, width, threads):
    if not hasattr(cv2, "CAP_PROP_N_THREADS"):
        raise RuntimeError("OpenCV FFmpeg decoder thread control is required")
    cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, threads])
    if not cap.isOpened():
        raise ValueError("Cannot open video")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if count < 3 or fps <= 0:
        raise ValueError("Video must contain at least three timed frames")
    context, targets = frame_indices(count, context, targets)
    selected = set(context + targets)
    frames = {}
    try:
        for i in range(max(selected) + 1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Video decode failed at frame {i}")
            if i in selected:
                h, w = frame.shape[:2]
                scale = min(1., width / w)
                frames[i] = cv2.resize(frame, (round(w * scale), round(h * scale)),
                                       interpolation=cv2.INTER_AREA)
    finally:
        cap.release()
    return frames, count, fps, context, targets
