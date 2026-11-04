"""Decode selected video frames; no feature matching or pose estimation."""
import cv2
import numpy as np


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


def temporal_candidate_spans(count, fps, seconds=4.):
    """Three bounded local searches after a global selection fails; original indices."""
    length=max(3,int(round(fps*seconds)))
    if count<=length:
        return []
    starts=[count-length,(count-length)//2,0]
    return list(dict.fromkeys((start,start+length-1) for start in starts))


def decode_selected(video, context, targets, width, threads, *, calibration=None, candidate_count=0,
                    candidate_span=None):
    if not hasattr(cv2, "CAP_PROP_N_THREADS"):
        raise RuntimeError("OpenCV FFmpeg decoder thread control is required")
    cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, threads])
    if not cap.isOpened():
        raise ValueError("Cannot open video")
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if calibration is not None or candidate_count:
            # Container counts can include empty AVI slots or duration estimates.
            # General MP4 B-frame packets are in decode order with reordered PTS;
            # use actual decoded presentation frames, matching cap.read() order.
            from video_input.capture import inspect_clip
            stream, timestamps = inspect_clip(video, decoded_frames=True)
            count, fps = len(timestamps), stream["fps"]
        if count < 3 or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Video must contain at least three timed frames")
        if candidate_count:
            if not 3 <= candidate_count <= 12:
                raise ValueError("Candidate count must be 3..12")
            low,high=(0,count-1) if candidate_span is None else candidate_span
            if not 0<=low<high<count or high-low<2:
                raise ValueError('Candidate span must include at least three actual frames')
            selected = set(np.linspace(low, high, min(high-low+1, candidate_count), dtype=int).tolist())
            context, targets = [low, high], [(low+high)//2]
        else:
            context, targets = frame_indices(count, context, targets)
            selected = set(context + targets)
        frames = {}
        for i in range(max(selected) + 1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Video decode failed at frame {i}")
            if i in selected:
                if calibration is not None:
                    frames[i] = calibration.rectify(frame, width)
                else:
                    h, w = frame.shape[:2]
                    scale = min(1., width / w)
                    frames[i] = cv2.resize(frame, (round(w * scale), round(h * scale)),
                                           interpolation=cv2.INTER_AREA)
    finally:
        cap.release()
    return frames, count, fps, context, targets
