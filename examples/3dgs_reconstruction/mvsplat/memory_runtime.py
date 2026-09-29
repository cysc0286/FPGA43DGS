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
    pieces = []
    for first in range(0, disparities.shape[1], chunk):
        depth = 1.0 / disparities[:, first:first + chunk].repeat([1, 1, *feat10.shape[-2:]])
        warped = warp_with_pose_depth_candidates(feat10, intrinsics, pose, depth,
                                                  warp_padding_mode="zeros")
        correlation = (feat01.unsqueeze(2) * warped).sum(1) / (feat01.shape[1] ** 0.5)
        pieces.append(correlation)
        del warped, depth
    return torch.cat(pieces, dim=1)
'''


def prepare_chunked_source(vendor, destination, chunk):
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
    shutil.copytree(vendor, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    (destination/target).write_text(changed, encoding="utf-8")
    diff = "".join(difflib.unified_diff(source.splitlines(True), changed.splitlines(True),
                                    fromfile="upstream/"+target.as_posix(), tofile="bounded/"+target.as_posix()))
    (destination.parent/"cost_volume_chunk.patch").write_text(diff, encoding="utf-8")
    record = dict(depth_chunk=chunk, depth_candidates=128, original_sha256=sha(vendor/target),
                  modified_sha256=sha(destination/target), changed_file=target.as_posix(),
                  effect="warp and channel dot-product per depth chunk; no candidate pruning or precision change")
    save(destination.parent/"memory_schedule.json", record)
    return destination
