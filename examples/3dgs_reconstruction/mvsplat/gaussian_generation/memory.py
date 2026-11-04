"""Generate an isolated upstream source variant with bounded cost-volume scratch.

The 128 depth hypotheses, channel reduction, pretrained layers and weights stay
unchanged. Only warp + dot-product scheduling is split along independent depths.
"""
from pathlib import Path
import difflib
import shutil

from common import save, sha

OLD = '''                feat01_warped = warp_with_pose_depth_candidates(
                    feat10,
                    intr_curr,
                    pose_curr,
                    1.0 / disp_candi_curr.repeat([1, 1, *feat10.shape[-2:]]),
                    warp_padding_mode="zeros",
                )  # [B, C, D, H, W]
                # calculate similarity
                raw_correlation_in = (feat01.unsqueeze(2) * feat01_warped).sum(
                    1
                ) / (
                    c**0.5
                )  # [vB, D, H, W]'''

HELPER = '''

def heterogs_chunked_correlation(feat01, feat10, intrinsics, pose, disparities, chunk):
    # Independent depth hypotheses; keep upstream channel summation and order.
    inference = not torch.is_grad_enabled()
    pieces = [] if not inference else None
    output = None
    for first in range(0, disparities.shape[1], chunk):
        depth = 1.0 / disparities[:, first:first + chunk].repeat([1, 1, *feat10.shape[-2:]])
        warped = warp_with_pose_depth_candidates(feat10, intrinsics, pose, depth,
                                                  warp_padding_mode="zeros")
        if inference and warped.dtype == feat01.dtype:
            # grid_sample owns this disposable result. Reusing it avoids a
            # second [B,C,chunk,H,W] allocation without changing the reduction.
            warped.mul_(feat01.unsqueeze(2))
            correlation = warped.sum(1) / (feat01.shape[1] ** 0.5)
        else:
            # Preserve the original differentiable path and type promotion.
            correlation = (feat01.unsqueeze(2) * warped).sum(1) / (feat01.shape[1] ** 0.5)
        if inference:
            if output is None:
                output = correlation.new_empty((correlation.shape[0], disparities.shape[1],
                                                *correlation.shape[2:]))
            output[:, first:first + correlation.shape[1]].copy_(correlation)
        else:
            pieces.append(correlation)
        del warped, depth, correlation
    return output if inference else torch.cat(pieces, dim=1)
'''


def prepare_chunked_source(vendor, destination, chunk, *, corr_depth_pipeline=False,
                           refine_heads_pipeline=False):
    if type(corr_depth_pipeline) is not bool:
        raise ValueError('corr_depth_pipeline must be an explicit boolean')
    if type(refine_heads_pipeline) is not bool:
        raise ValueError('refine_heads_pipeline must be an explicit boolean')
    if (corr_depth_pipeline or refine_heads_pipeline) and type(chunk) is not int:
        raise ValueError('Continuous pipeline requires an integer depth chunk within 1..128')
    if not 1 <= chunk <= 128:
        raise ValueError("Depth chunk must be within 1..128")
    vendor, destination = Path(vendor), Path(destination)
    target = Path("src/model/encoder/costvolume/depth_predictor_multiview.py")
    source = (vendor/target).read_text(encoding="utf-8")
    if source.count(OLD) != 1:
        raise ValueError("Upstream cost-volume source changed; review patch")
    replacement = f'''                raw_correlation_in = heterogs_chunked_correlation(
                    feat01, feat10, intr_curr, pose_curr, disp_candi_curr, {chunk})'''
    changed = source.replace(OLD, replacement) + HELPER
    seams = []
    if corr_depth_pipeline:
        from gaussian_generation.npu_branch import corr_depth
        seams.append(('corr_depth_pipeline', corr_depth, 'mvsplat_corr_depth_generated_source_v1'))
    if refine_heads_pipeline:
        from gaussian_generation.npu_branch import refine_heads
        seams.append(('refine_heads_pipeline', refine_heads, 'mvsplat_refine_heads_generated_source_v1'))
    transformed = []
    for name, tool, schema in seams:
        previous = changed
        changed, metadata = tool.patch_source(previous)
        seam_diff = "".join(difflib.unified_diff(previous.splitlines(True), changed.splitlines(True),
            fromfile="previous/"+target.as_posix(), tofile=name+"/"+target.as_posix()))
        transformed.append((name, tool, schema, metadata, seam_diff))
    shutil.copytree(vendor, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    (destination/target).write_text(changed, encoding="utf-8")
    diff = "".join(difflib.unified_diff(source.splitlines(True), changed.splitlines(True),
                                    fromfile="upstream/"+target.as_posix(), tofile="bounded/"+target.as_posix()))
    (destination.parent/"cost_volume_chunk.patch").write_text(diff, encoding="utf-8")
    record = dict(depth_chunk=chunk, depth_candidates=128, original_sha256=sha(vendor/target),
                  modified_sha256=sha(destination/target), changed_file=target.as_posix(),
                  inference_scratch="reuse_warp_and_preallocate_correlation",
                  gradient_schedule="original_multiply_and_concatenate",
                  effect="warp and channel dot-product per depth chunk; reuse disposable warp in inference; no candidate pruning or precision change")
    for name, tool, schema, metadata, seam_diff in transformed:
        seam_record = dict(schema=schema, enabled=True,
            changed_file=target.as_posix(), original_file_sha256=sha(vendor/target),
            generated_file_sha256=sha(destination/target),
            tool_sha256=sha(Path(tool.__file__)), transform=metadata,
            source_written='isolated runtime_source only; upstream vendor unchanged',
            npu_executed=False, qualified=False)
        (destination.parent/(name+'.patch')).write_text(seam_diff, encoding='utf-8')
        save(destination.parent/(name+'.json'), seam_record)
        record[name] = dict(enabled=True,
            metadata=name+'.json', patch=name+'.patch',
            tool_sha256=seam_record['tool_sha256'])
    save(destination.parent/"memory_schedule.json", record)
    return destination
