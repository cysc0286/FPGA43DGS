"""Bounded disk-backed local-video upload, independent of USB and inference.

The HTTP coordinator owns authorization and exclusive job admission. This module
only receives a new copy, verifies decoded presentation frames, and returns an
owned completed clip. It never deletes the source, starts a job or loads a model.
"""
import hashlib
import json
import math
from pathlib import Path
import secrets
import shutil
import time

from video_input.capture import inspect_clip


MAX_UPLOAD_BYTES = 256 * 1024 ** 2
RESERVE_BYTES = 64 * 1024 ** 2
VIDEO_SUFFIXES = frozenset(('.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v',
                            '.mpeg', '.mpg', '.ts', '.mts'))


def _filename(value):
    if (not isinstance(value, str) or not value or len(value) > 255
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ValueError('视频文件名无效，请重新选择文件')
    # A browser may provide C:\fakepath\clip.mp4. Only a display basename is kept;
    # no part of this name selects a directory or server-side file.
    name = value.replace('\\', '/').rsplit('/', 1)[-1]
    if name in ('', '.', '..'):
        raise ValueError('视频文件名无效，请重新选择文件')
    suffix = Path(name).suffix.lower()
    return name, suffix if suffix in VIDEO_SUFFIXES else '.video'


def _check_video(stream, times, *, min_seconds, max_seconds, min_edge, max_edge,
                 max_pixels, max_fps):
    if not isinstance(stream, dict) or not isinstance(times, (list, tuple)) or len(times) < 3:
        raise ValueError('视频需要至少三帧可解码画面')
    width, height = stream.get('width'), stream.get('height')
    if (type(width) is not int or type(height) is not int
            or min(width, height) < min_edge or max(width, height) > max_edge
            or width * height > max_pixels):
        raise ValueError(f'视频尺寸超出范围：短边至少 {min_edge}，长边最多 {max_edge}，总像素最多 {max_pixels}')
    fps = stream.get('fps')
    if (type(fps) not in (int, float) or not math.isfinite(fps) or not 0 < fps <= max_fps
            or not isinstance(stream.get('codec_name'), str) or not stream['codec_name']):
        raise ValueError(f'视频编码或帧率无效，帧率须在 0–{max_fps:g} fps 内')
    if any(type(t) not in (int, float) or not math.isfinite(t) for t in times):
        raise ValueError('视频时间戳无效')
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('视频解码时间戳不连续递增')
    duration = times[-1] - times[0]
    if not min_seconds <= duration <= max_seconds:
        raise ValueError(f'视频时长须在 {min_seconds:g}–{max_seconds:g} 秒内')
    # inspect_clip already verifies decoded PTS. The additional finite count
    # bound also protects an injected inspector from handing back a huge list.
    if len(times) > math.ceil(max_seconds * max_fps) + 2:
        raise ValueError('视频帧数超出当前板端处理范围')
    measured = (len(times) - 1) / duration
    if measured > max_fps * (1 + 1e-9):
        raise ValueError('视频实际解码帧率超出当前板端处理范围')
    return duration, measured


def _save_record(path, record):
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('x', encoding='utf-8') as target:
        json.dump(record, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write('\n')
    temporary.replace(path)


def stream_receive(stream, length, outroot, filename, *, max_bytes=MAX_UPLOAD_BYTES,
                   reserve_bytes=RESERVE_BYTES, min_seconds=1., max_seconds=120.,
                   min_edge=64, max_edge=4096, max_pixels=3840 * 2160,
                   max_fps=120., chunk_bytes=1024 ** 2, max_receive_seconds=120.,
                   inspector=inspect_clip, on_received=None):
    """Receive ``length`` raw body bytes and return a verified clip record.

    ``length`` is the parsed integer Content-Length, never chunked encoding.
    Defaults accept 1–120 s, 64-pixel minimum edge, at most 4096 on either edge,
    at most 4K UHD pixel count, and at most 120 fps. Storage is capped at 256 MiB
    and retains at least 64 MiB free. Failed partials remain diagnostic-only.
    These bounds describe ingestion; reconstruction quality is not guaranteed.
    ``on_received(monotonic, epoch)`` publishes the last-body-read timestamp
    before the final write/flush and decoded probe; it does not indicate that
    validation or durable storage has completed.
    """
    if type(max_bytes) is not int or not 0 < max_bytes <= MAX_UPLOAD_BYTES:
        raise ValueError('上传大小限制必须在 1–256 MiB 的字节范围内')
    if type(length) is not int or not 0 < length <= max_bytes:
        raise ValueError(f'上传文件不能为空，且不得超过 {max_bytes / 1024 ** 2:g} MiB')
    if (type(reserve_bytes) is not int or reserve_bytes < RESERVE_BYTES
            or type(chunk_bytes) is not int or not 0 < chunk_bytes <= 4 * 1024 ** 2
            or any(type(n) is not int or n < 1 for n in (min_edge, max_edge, max_pixels))
            or min_edge > max_edge or not callable(inspector)
            or (on_received is not None and not callable(on_received))
            or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0
                   for x in (min_seconds, max_seconds, max_fps, max_receive_seconds))
            or min_seconds > max_seconds):
        raise ValueError('上传限制配置无效')
    original_name, suffix = _filename(filename)
    root = Path(outroot)
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError('上传存储目录不可用')
    if shutil.disk_usage(root).free < length + reserve_bytes:
        raise OSError('板端空间不足：上传后需保留 64 MiB 系统空间')
    folder = root / ('upload_' + time.strftime('%Y%m%d_%H%M%S') + '_' + secrets.token_hex(4))
    folder.mkdir(exist_ok=False)
    partial, final = folder / 'video.partial', folder / ('video' + suffix)
    record = dict(schema='mvsplat_uploaded_video_v1', complete=False,
        source_kind='uploaded_video', original_filename=original_name,
        declared_bytes=length, received_bytes=0, storage_limit_bytes=max_bytes,
        reserve_bytes=reserve_bytes, max_receive_seconds=max_receive_seconds,
        timestamp_scope='decoded presentation timestamps, not exposure times',
        limits=dict(min_seconds=min_seconds, max_seconds=max_seconds, min_edge=min_edge,
                    max_edge=max_edge, max_pixels=max_pixels, max_fps=max_fps))
    started, digest = time.monotonic(), hashlib.sha256()
    try:
        with partial.open('xb') as target:
            while record['received_bytes'] < length:
                if time.monotonic() - started > max_receive_seconds:
                    raise TimeoutError('上传超过允许时长，请检查网络后重试')
                remaining = length - record['received_bytes']
                count = min(chunk_bytes, remaining)
                if shutil.disk_usage(folder).free < count + reserve_bytes:
                    raise OSError('板端可用空间不足，上传已停止并保留未完成文件')
                block = stream.read(count)
                received_monotonic, received_epoch = time.monotonic(), time.time()
                if received_monotonic - started > max_receive_seconds:
                    raise TimeoutError('上传超过允许时长，请检查网络后重试')
                if not isinstance(block, (bytes, bytearray)) or not block:
                    raise ValueError('上传提前结束，请检查连接后重新选择视频')
                if len(block) > count:
                    raise ValueError('上传流返回了超过请求范围的数据')
                if len(block) == remaining:
                    # Receipt is a transport boundary, not a storage boundary.
                    # Preserve this instant while final write/flush, hashing,
                    # video validation and processing run afterwards.
                    record.update(upload_complete_monotonic=received_monotonic,
                                  upload_complete_epoch=received_epoch,
                                  upload_seconds=received_monotonic-started)
                    if on_received is not None:
                        on_received(received_monotonic, received_epoch)
                target.write(block)
                digest.update(block)
                record['received_bytes'] += len(block)
            target.flush()
        stream_info, times = inspector(partial, decoded_frames=True)
        duration, measured = _check_video(stream_info, times, min_seconds=min_seconds,
            max_seconds=max_seconds, min_edge=min_edge, max_edge=max_edge,
            max_pixels=max_pixels, max_fps=max_fps)
        if shutil.disk_usage(folder).free < reserve_bytes:
            raise OSError('视频检查后存储余量不足，未交付处理')
        partial.rename(final)
        record.update(complete=True, path=str(final.resolve()), sha256=digest.hexdigest(),
            actual=dict(stream_info), frames=len(times), duration_seconds=duration,
            measured_fps=measured, frame_pts_seconds=times,
            stop_reason='upload_complete', stop_label='本地视频上传完成',
            preprocessing=dict(network_size=[128, 128], geometry='preserve aspect ratio then centered square crop',
                               source_aspect_ratio=stream_info['width'] / stream_info['height'],
                               source_calibration='not supplied; USB camera calibration must not be assumed'))
        _save_record(folder / 'upload.json', record)
        return record
    except BaseException as exc:
        record.update(complete=False, error=type(exc).__name__ + ': ' + str(exc),
                      elapsed_seconds=time.monotonic() - started)
        try:
            _save_record(folder / 'upload.json', record)
        except OSError:
            pass
        raise
