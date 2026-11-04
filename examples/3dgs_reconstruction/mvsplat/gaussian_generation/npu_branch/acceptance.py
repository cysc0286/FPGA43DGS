"""Artifact-bound native-image qualification, distinct from tensor equivalence.

Receipts are local experiment evidence, not signed security attestations or
proof of quality on arbitrary unseen scenes. No diagnostic-only receipt passes.
"""
import json
import math
from pathlib import Path

from common import FULL_WEIGHT_SHA256, sha
from gaussian_generation.npu_branch.catalog import validate_selection

SCHEMA = 'mvsplat_npu_native_acceptance_v1'
QUALITY_LIMITS = dict(minimum_psnr_db=40., minimum_ssim=.99,
                      maximum_truth_psnr_drop_db=.2, maximum_truth_ssim_drop=.005)
PROVENANCE_KEYS = ('bundle_sha256', 'oracle_manifest_sha256', 'bridge_sha256')


def check_views(views):
    """Require at least three distinct held-out views through the same FPGA path."""
    if len(views) < 3:
        raise ValueError('NPU image qualification needs three held-out target views')
    for name, view in views.items():
        q = view['npu_vs_cpu']
        values = (q['psnr_db'], q['ssim'], view['truth_psnr_drop_db'], view['truth_ssim_drop'])
        if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in values):
            raise ValueError('Nonfinite image qualification metric: ' + name)
        if (q['psnr_db'] < QUALITY_LIMITS['minimum_psnr_db'] or
                q['ssim'] < QUALITY_LIMITS['minimum_ssim'] or
                view['truth_psnr_drop_db'] > QUALITY_LIMITS['maximum_truth_psnr_drop_db'] or
                view['truth_ssim_drop'] > QUALITY_LIMITS['maximum_truth_ssim_drop']):
            raise ValueError('NPU native image quality gate failed: ' + name)


def check_receipt(record):
    names = record.get('partitions', [])
    validate_selection(names)
    if (record.get('schema') != SCHEMA or record.get('complete') is not True or
            record.get('diagnostic_only', True) is not False or
            record.get('npu_executed') is not True or record.get('fpga_executed') is not True or
            record.get('gaussian_repeat_exact') is not True or
            record.get('weight_sha256') != FULL_WEIGHT_SHA256 or
            record.get('quality_limits') != QUALITY_LIMITS):
        raise ValueError('Incomplete or diagnostic NPU qualification receipt')
    if not record.get('runtime_identity', {}).get('sdk_libraries'):
        raise ValueError('NPU qualification lacks SDK identity')
    if not record.get('source_files'):
        raise ValueError('NPU qualification lacks fixed model source identity')
    if not record.get('oracle_checks') or not all(c.get('repeat_exact') is True for c in record['oracle_checks']):
        raise ValueError('NPU qualification lacks stable oracle outputs')
    checked = [c['partition'] for c in record['oracle_checks']]
    if len(checked) != len(names) or set(checked) != set(names):
        raise ValueError('NPU qualification must check every selected partition exactly once')
    check_views(record['views'])
    return record


def validate_model(path, model):
    record = load_receipt(path)
    if record['source_files'] != model.source_hashes:
        raise ValueError('Qualified model source changed; repeat native image qualification')


def load_receipt(path):
    return check_receipt(json.loads(Path(path).read_text(encoding='utf-8')))


def validate_renderer(path, renderer, binary):
    record = load_receipt(path)
    from rendering.resident_runtime import frozen_renderer_module
    package = frozen_renderer_module(Path(renderer)).verify()
    from rendering.runtime import load_mainline_profile
    profile = load_mainline_profile()
    if (record['renderer']['binary_sha256'] != sha(binary) or
            record['renderer']['package'] != package or
            record['renderer']['profile'] != profile or
            sha(record['renderer']['boot_image']) != profile['required_hardware']['accepted_boot_sha256']):
        raise ValueError('NPU qualified renderer/BOOT changed; requalify before READY')
    return record


def validate_worker(path, worker, checks):
    record = load_receipt(path)
    if (record['partitions'] != list(worker.parts) or
            record['runtime_identity'] != worker.readiness.get('runtime_identity') or
            any(record['provenance'][k] != worker.provenance[k] for k in PROVENANCE_KEYS)):
        raise ValueError('NPU qualification artifact/runtime identity mismatch')
    expected = {c['partition']: c for c in record['oracle_checks']}
    if set(expected) != set(worker.parts) or len(expected) != len(record['oracle_checks']):
        raise ValueError('NPU qualification oracle partition mismatch')
    checked = [c['partition'] for c in checks]
    if len(checked) != len(expected) or set(checked) != set(expected):
        raise ValueError('Current NPU oracle checks omit or duplicate a selected partition')
    for check in checks:
        old = expected[check['partition']]
        if (not check['repeat_exact'] or check['output_sha256'] != old['output_sha256'] or
                check['oracle_sha256'] != old['oracle_sha256']):
            raise ValueError('NPU oracle output changed since native-image qualification')
    return dict(receipt_sha256=sha(path), policy='native_fpga_image',
                scope='qualified held-out views and fixed oracle repeat; unseen-scene quality is not guaranteed',
                quality_limits=QUALITY_LIMITS)
