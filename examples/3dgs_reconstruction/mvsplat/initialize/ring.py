"""Coordinate the explicit sparse fixed-centre candidate using existing modules."""
import time
from pathlib import Path

from common import gaussian_content_sha256, save, sha


def calibrate_ring_appearance(gaussians,inference,prepared,renderer,out,strength,transition_degrees):
    """Use only source images and resident FPGA frames; retain the parent identity."""
    import numpy as np
    import cv2
    from gaussian_generation.adapter import camera_bytes,gaussian_rows
    from gaussian_generation.appearance import bake_context_residual
    from gaussian_generation.blending import AngularFeathering
    if not callable(getattr(renderer,'render_linear',None)):
        raise ValueError('Appearance candidate requires the native linear framebuffer interface')
    out = Path(out)
    started = time.monotonic()
    if (inference.get('gaussian_identity_kind')!='array_content_v1'
            or inference.get('archive_verified') is not False):
        raise ValueError('Appearance requires an unarchived in-memory ring scene')
    views = {v['index']:v for v in prepared['views']}
    cameras = [views[i] for pair in prepared['pairs'] for i in pair['context']]
    feather = AngularFeathering(prepared['coverage']['directions'],cameras,transition_degrees)
    parent = gaussian_content_sha256(gaussians)
    if parent!=inference['gaussian_content_sha256']:
        raise ValueError('Appearance parent Gaussian identity changed')
    rows,_ = gaussian_rows(gaussians)
    renderer.load_rows(rows)
    del rows
    frames = []
    def read_context(view):
        if Path(view['name']).name!=view['name']:
            raise ValueError('Invalid appearance source filename')
        image = out/'input/images'/view['name']
        if sha(image)!=view['prepared_sha256']:
            raise ValueError('Appearance context changed after preparation')
        pixels = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if pixels is None:
            raise ValueError('Cannot decode verified appearance context')
        return np.asarray(cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB),np.float64)/255.
    def render_context(view):
        rgb,coverage,frame = renderer.render_linear(camera_bytes(view))
        if not frame.get('complete'):
            raise ValueError('Incomplete internal appearance frame')
        frames.append(dict(source_index=view['index'],**frame))
        return rgb,coverage
    result,record = bake_context_residual(gaussians,cameras,feather,read_context,
                                          render_context,strength=strength)
    record.update(total_seconds=time.monotonic()-started,render_frames=frames,
                  parent_gaussian_content_sha256=parent)
    content = gaussian_content_sha256(result)
    updated = dict(inference,gaussian_sha256=content,gaussian_content_sha256=content,
                   appearance=record,process_seconds=inference.get('process_seconds',0.)+record['total_seconds'])
    save(out/'inference/inference.json',updated)
    return updated,result


def prepare_ring_scene(video, out, runtime, renderer, events, *, groups=2, candidates=24,
                       max_gaussians=65536, size=128, prepare_threads=1,
                       focal_ratio=.9, calibration=None, max_pairs=8, min_pairs=1,
                       coverage_target=.95, feature_contrast=.04,transition_degrees=8.,
                       appearance_strength=0., selection_policy='balanced'):
    import math
    if (type(appearance_strength) not in (int,float) or not math.isfinite(appearance_strength)
            or not 0<=appearance_strength<=1 or (appearance_strength and not transition_degrees)):
        raise ValueError('Appearance strength must be in [0,1] with observed overlap enabled')
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
                            coverage_target=coverage_target, feature_contrast=feature_contrast,
                            selection_policy=selection_policy)
    actual_groups = len(prepared['pairs'])
    inference, gaussians = generate_ring(runtime, prepared, out/'inference',
                                         max_gaussians=max_gaussians, progress=progress,
                                         defer_archive=True,transition_degrees=transition_degrees)
    if appearance_strength:
        progress(dict(stage='appearance_calibration',pair=actual_groups,of=actual_groups))
        inference,gaussians = calibrate_ring_appearance(gaussians,inference,prepared,renderer,
            out,appearance_strength,transition_degrees)
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
