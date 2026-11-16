"""One retained inference session, bounded angular support, no depth alignment."""
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from common import gaussian_content_sha256, load_gaussians, save, sha, validate_gaussians
from gaussian_generation.blending import AngularFeathering, SOURCE_FEATHER, attenuate_opacity
from gaussian_generation.sampling import (BALANCED_SAMPLING, FOOTPRINT_COMPENSATION, MOMENT_MERGE,
                                          SPATIAL_SAMPLING, balanced_sample_indices,compensate_covariances,
                                          merge_retained_neighbours,spatial_sample_indices)


def angular_owned_indices(gaussians, pair_directions, all_directions, global_view_indices,
                          max_per_view, *, source_shape=None, sampling='legacy_linear',
                          return_counts=False,contribution_weights=None):
    """Select hard owners or explicitly weighted source support within a budget.

    Optional contribution weights retain observed overlap; neither branch fits
    a shared depth across independently predicted source views.
    """
    g = validate_gaussians(gaussians)
    if len(g['means']) % 2 or type(max_per_view) is not int or max_per_view < 1:
        raise ValueError('Expected equal two-view Gaussian blocks and a positive budget')
    if sampling not in ('spatial', 'legacy_linear','balanced'):
        raise ValueError('Unknown ring sampling policy')
    if type(return_counts) is not bool:
        raise ValueError('Ownership counts must be explicitly requested')
    directions = np.asarray(all_directions, np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3 or not np.isfinite(directions).all():
        raise ValueError('Invalid source directions')
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms < .999) or np.any(norms > 1.001):
        raise ValueError('Source axes must be unit length')
    if len(global_view_indices) != 2 or len(set(global_view_indices)) != 2:
        raise ValueError('Invalid view ownership')
    if not np.allclose(directions[global_view_indices], pair_directions, atol=1e-6):
        raise ValueError('Pair axes differ from global frame')
    points = np.asarray(g['means'], np.float64)
    lengths = np.linalg.norm(points, axis=1)
    valid = lengths > 1e-8
    if contribution_weights is None:
        rays = points/np.maximum(lengths[:, None], 1e-8)
        owner = np.argmax(rays @ directions.T, axis=1)
    else:
        weights = np.asarray(contribution_weights)
        if (weights.shape!=(len(points),) or not np.isfinite(weights).all()
                or np.any(weights<0) or np.any(weights>1)):
            raise ValueError('Invalid source contribution weights')
        valid &= weights>0
        owner = None
    per_view = len(points)//2
    if source_shape is None:
        side = math.isqrt(per_view)
        # The locked MVSplat model emits one row-major Gaussian per source pixel.
        # Non-image test fixtures retain an explicit one-row fallback.
        source_shape = (side, side) if side*side == per_view else (1, per_view)
    if (not isinstance(source_shape, (tuple, list)) or len(source_shape) != 2
            or any(type(x) is not int or x < 1 for x in source_shape)
            or math.prod(source_shape) != per_view):
        raise ValueError('Source image shape differs from one-Gaussian-per-pixel output')
    selected,owned_counts = [],[]
    for local, global_index in enumerate(global_view_indices):
        index = np.arange(local*per_view, (local+1)*per_view)
        index = index[valid[index] & ((owner[index] == global_index) if owner is not None else True)]
        owned_counts.append(len(index))
        if len(index) > max_per_view:
            if sampling == 'legacy_linear':
                index = index[np.linspace(0, len(index)-1, max_per_view, dtype=int)]
            else:
                start = local*per_view
                select = balanced_sample_indices if sampling=='balanced' else spatial_sample_indices
                index = start+select(index-start,max_per_view,source_shape)
        selected.append(index)
    retained = np.concatenate(selected)
    return (retained,owned_counts) if return_counts else retained


def persist_ring_archive(record, gaussians, out):
    """Archive the rendered arrays after FRAME_COMPLETE, without changing identity.

    A deferred scene's Gaussian identity is the canonical array-content digest.
    The NPZ file digest is a separate field and must not replace that identity.
    Existing verified archives are checked and reused, never blindly overwritten.
    """
    out = Path(out)
    g = validate_gaussians(gaussians)
    content = gaussian_content_sha256(g)
    kind = record.get('gaussian_identity_kind')
    if (not record.get('complete') or kind not in ('array_content_v1', 'npz_file_v1')
            or content != record.get('gaussian_content_sha256')
            or (kind == 'array_content_v1' and record.get('gaussian_sha256') != content)):
        raise ValueError('Ring Gaussian content identity changed before archival')
    archive = out/'gaussians.npz'
    if record.get('archive_verified'):
        if not archive.is_file() or sha(archive) != record.get('gaussian_archive_sha256'):
            raise ValueError('Existing ring Gaussian archive changed')
        save(out/'inference.json', record)
        return record
    if archive.exists():
        raise FileExistsError('Refusing to overwrite an unverified ring archive')
    temporary = out/'.gaussians.pending.npz'
    created = False
    started = time.monotonic()
    try:
        with temporary.open('xb') as stream:
            created = True
            np.savez(stream, **g)
        digest = sha(temporary)
        restored = load_gaussians(temporary)
        if any(not np.array_equal(restored[key], g[key]) for key in g):
            raise ValueError('Ring NPZ differs from the rendered Gaussian arrays')
        if gaussian_content_sha256(restored) != content:
            raise ValueError('Ring archive changed canonical Gaussian content')
        del restored
        if archive.exists():
            raise FileExistsError('Ring archive appeared during deferred validation')
        temporary.replace(archive)
    except BaseException:
        if created:
            temporary.unlink(missing_ok=True)
        raise
    record.update(archive_verified=True, gaussian_archive_sha256=digest,
                  archive_bytes=archive.stat().st_size,
                  archive_seconds=time.monotonic()-started)
    if kind == 'npz_file_v1':
        record['gaussian_sha256'] = digest
    save(out/'inference.json', record)
    return record


def generate_ring(runtime, prepared, out, *, max_gaussians=65536, progress=None,
                  defer_archive=False, sampling='balanced', footprint_compensation=True,
                  reduction='moments',transition_degrees=0.,source_edge_margin=.04):
    if (type(max_gaussians) is not int or not 1024 <= max_gaussians <= 131072
            or prepared.get('depth_observable') is not False):
        raise ValueError('Invalid bounded rotation-only generation contract')
    if (type(footprint_compensation) is not bool or sampling not in ('legacy_linear','spatial','balanced')
            or reduction not in ('footprint','moments')):
        raise ValueError('Invalid ring retention policy')
    if (type(transition_degrees) not in (int,float) or not np.isfinite(transition_degrees)
            or not 0<=transition_degrees<=12
            or type(source_edge_margin) not in (int,float) or not np.isfinite(source_edge_margin)
            or not 0<=source_edge_margin<=.25):
        raise ValueError('Invalid ring transition width')
    out = Path(out)
    out.mkdir(exist_ok=False)
    pairs = prepared['pairs']
    if not 1 <= len(pairs) <= 8:
        raise ValueError('Unsupported pair count')
    per_view = max_gaussians//(2*len(pairs))
    directions = prepared['coverage']['directions']
    shapes = dict(means=(3,), covariances=(3, 3), harmonics=(3, 25), opacities=())
    accumulated = {k:np.empty((max_gaussians, *shape), np.float32) for k, shape in shapes.items()}
    records, position, generated_payload_bytes = [], 0, 0
    source_views = {view['index']:view for view in prepared.get('views', [])}
    feather = None
    if transition_degrees:
        try:
            cameras = [source_views[i] for pair in pairs for i in pair['context']]
        except KeyError as exc:
            raise ValueError('Source feathering requires actual camera metadata for every context') from exc
        feather = AngularFeathering(directions,cameras,transition_degrees,source_edge_margin)
    started = time.monotonic()
    for n, pair in enumerate(pairs):
        folder = Path(pair['directory'])
        if sha(folder/'context.npz') != pair['context_sha256']:
            raise ValueError('Panorama pair changed')
        if progress:
            progress(dict(stage='inference', pair=n+1, of=len(pairs)))
        inference = runtime.infer(folder, out/f'pair_{n:02d}', pair['context_sha256'],
                                  memory_only=True)
        g = runtime.take_gaussians(inference['gaussian_sha256'])
        if (not inference.get('complete')
                or inference.get('gaussian_identity_kind') != 'array_content_v1'
                or inference.get('archive_verified') is not False
                or inference.get('gaussian_archive', {}).get('policy') != 'memory_only'):
            raise ValueError('Ring pair did not use the explicit memory-only Gaussian contract')
        if any(value.dtype != np.float32 for value in g.values()):
            raise ValueError('Ring accumulation requires original FP32 Gaussian fields')
        payload_bytes = sum(value.nbytes for value in g.values())
        generated_payload_bytes += payload_bytes
        global_indices = [2*n, 2*n+1]
        contribution_weights = None
        if feather is not None:
            if not len(g['means']) or len(g['means']) % 2:
                raise ValueError('Feathering requires two equal Gaussian source blocks')
            weights = feather.weights(g['means'])
            local_source = np.arange(len(g['means']))//(len(g['means'])//2)
            contribution_weights = weights[np.arange(len(g['means'])),2*n+local_source]
            g = dict(g,opacities=attenuate_opacity(g['opacities'],contribution_weights))
            del weights,local_source
        source_shape = None
        if source_views:
            shapes_in = [(source_views[i]['height'],source_views[i]['width']) for i in pair['context']]
            if len(shapes_in) != 2 or shapes_in[0] != shapes_in[1]:
                raise ValueError('Ring contexts must have the same source image dimensions')
            source_shape = shapes_in[0]
        index,owned_counts = angular_owned_indices(g, np.asarray(directions)[global_indices], directions,
                                      global_indices, per_view, source_shape=source_shape,
                                      sampling=sampling, return_counts=True,
                                      contribution_weights=contribution_weights)
        if not len(index):
            raise ValueError('No angularly owned Gaussians in a requested pair')
        end = position+len(index)
        corrected_covariance,corrected_fields,retention = None,None,None
        if footprint_compensation:
            cameras = [source_views.get(i) for i in pair['context']]
            if reduction=='moments':
                owned = angular_owned_indices(g,np.asarray(directions)[global_indices],directions,
                    global_indices,len(g['means']),source_shape=source_shape,
                    contribution_weights=contribution_weights)
                corrected_fields,retention = merge_retained_neighbours(g,index,owned,cameras)
                del owned
            else:
                corrected_covariance,retention = compensate_covariances(g,index,cameras,owned_counts)
        for key in shapes:
            accumulated[key][position:end] = (corrected_fields[key] if corrected_fields is not None
                else corrected_covariance if key=='covariances' and corrected_covariance is not None
                else g[key][index])
        records.append(dict(context=pair['context'], context_sha256=pair['context_sha256'],
                            gaussian_sha256=inference['gaussian_sha256'], generated=len(g['means']),
                            gaussian_identity_kind=inference['gaussian_identity_kind'],
                            archive_policy='memory_only', generated_payload_bytes=payload_bytes,
                            retained=len(index), pre_thinning_owned_counts=owned_counts,
                            footprint_retention=retention,
                            inference_seconds=inference['inference_seconds'],
                            backend=inference['backend'], npu_executed=inference['npu_executed']))
        position = end
        del g,index,corrected_covariance,corrected_fields,contribution_weights
        gc.collect()
    result = {k:v[:position] for k, v in accumulated.items()}
    validate_gaussians(result)
    content = gaussian_content_sha256(result)
    record = dict(complete=True, schema='mvsplat_rotation_ring_inference_v1',
                  backend=records[0]['backend'], npu_executed=all(r['npu_executed'] for r in records),
                  input_sha256=prepared['context_sha256'], gaussian_content_sha256=content,
                  gaussian_identity_kind='array_content_v1' if defer_archive else 'npz_file_v1',
                  gaussian_sha256=content if defer_archive else None,
                  gaussian_archive_sha256=None, archive_verified=False,
                  archive_policy='deferred_after_first_frame' if defer_archive else 'immediate',
                  gaussians=position, max_gaussians=max_gaussians, per_view_budget=per_view,
                  inference_seconds=sum(r['inference_seconds'] for r in records),
                  process_seconds=time.monotonic()-started, pairs=records,
                  source_manifest_sha256=hashlib.sha256(json.dumps(records, sort_keys=True,
                      separators=(',', ':')).encode()).hexdigest(),
                  storage_io_estimate=dict(kind='logical_payload_estimate_not_ddr_measurement',
                      omitted_pair_npz_write_bytes=generated_payload_bytes,
                      omitted_pair_npz_verification_read_bytes=generated_payload_bytes,
                      omitted_pair_npz_hash_read_bytes=generated_payload_bytes,
                      omitted_total_pair_storage_payload_bytes=3*generated_payload_bytes,
                      scope='Excludes NPZ headers, filesystem/cache effects, content hashing and final aggregate archive'),
                  coordinate_frame='common_rotation_origin_zero_translation',
                  learned_depth_observable=False, full_3d_reconstruction=False,
                  overlap_policy=(SOURCE_FEATHER if feather is not None else
                                  'nearest source optical axis; no averaging of unobservable depths'),
                  transition_degrees=transition_degrees,source_edge_margin=source_edge_margin,
                  point_sampling=({'spatial':SPATIAL_SAMPLING,'balanced':BALANCED_SAMPLING,
                                   'legacy_linear':'legacy_linear'}[sampling]),
                  footprint_compensation=(FOOTPRINT_COMPENSATION if reduction=='footprint' else MOMENT_MERGE)
                      if footprint_compensation else 'none',
                  retention_reduction=reduction if footprint_compensation else 'none',
                  quality_scope=('Fixed-centre angular candidate; local reduction may smooth details; depth remains unobservable'
                      if footprint_compensation else
                      'Fixed-centre angular candidate; uncompensated thinning may lose pixel coverage; depth remains unobservable'))
    save(out/'inference.json', record)
    if not defer_archive:
        persist_ring_archive(record, result, out)
        record['process_seconds'] = time.monotonic()-started
        save(out/'inference.json', record)
    return record, result
