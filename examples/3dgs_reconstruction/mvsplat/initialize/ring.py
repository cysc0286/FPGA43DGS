"""Coordinate the explicit sparse fixed-centre candidate using existing modules."""
import time
from pathlib import Path

from common import save, sha


def prepare_ring_scene(video, out, runtime, renderer, events, *, groups=2, candidates=24,
                       max_gaussians=65536, size=128, prepare_threads=1,
                       focal_ratio=.9, calibration=None, max_pairs=8, min_pairs=1,
                       coverage_target=.95, feature_contrast=.04):
    from video_input.ring import prepare_ring
    from gaussian_generation.ring import generate_ring, persist_ring_archive
    from gaussian_generation.adapter import main as export, camera_bytes, gaussian_rows
    out = Path(out)
    started = time.monotonic()
    def progress(item):
        temporary = out/'progress.tmp'
        save(temporary, dict(item, elapsed_seconds=time.monotonic()-started))
        temporary.replace(out/'progress.json')
    progress(dict(stage='rotation_preparation', pair=0, of=max_pairs if groups=='auto' else groups))
    prepared = prepare_ring(video, out/'input', groups=groups, candidates=candidates,
                            size=size, threads=prepare_threads, focal_ratio=focal_ratio,
                            calibration=calibration, max_pairs=max_pairs, min_pairs=min_pairs,
                            coverage_target=coverage_target, feature_contrast=feature_contrast)
    actual_groups = len(prepared['pairs'])
    inference, gaussians = generate_ring(runtime, prepared, out/'inference',
                                         max_gaussians=max_gaussians, progress=progress,
                                         defer_archive=True)
    progress(dict(stage='scene_load', pair=actual_groups, of=actual_groups))
    target = next(v for v in prepared['views'] if v['role'] == 'target')
    rows, _ = gaussian_rows(gaussians)
    resident = renderer.load_rows(rows)
    events.emit('SCENE_READY', gaussian_sha256=inference['gaussian_sha256'],
                gaussian_identity_kind=inference['gaussian_identity_kind'],
                resident_scene=resident, scope='sparse fixed-centre rotation-only candidate')
    frame_dir = out/'board/first_frame'
    frame = renderer.render_camera(camera_bytes(target), frame_dir)
    if not frame['complete'] or frame['frame_sha256'] != sha(frame_dir/'frame.bin'):
        raise ValueError('Incomplete sparse panorama first frame')
    events.emit('FRAME_COMPLETE', frame_sha256=frame['frame_sha256'], camera=target['name'])
    progress(dict(stage='archive', pair=actual_groups, of=actual_groups))
    persist_ring_archive(inference, gaussians, out/'inference')
    export(['--input', str(out/'input'), '--inference', str(out/'inference'),
            '--out', str(out/'renderer_input')], gaussians=gaussians)
    progress(dict(stage='complete', pair=actual_groups, of=actual_groups))
    return dict(frame=frame, inference=inference,
                scheduling=dict(ring_total_with_archive_seconds=time.monotonic()-started,
                                video_prepare_stages=prepared['timing']),
                coverage=prepared['coverage'], result_scope=prepared['quality_scope'])
