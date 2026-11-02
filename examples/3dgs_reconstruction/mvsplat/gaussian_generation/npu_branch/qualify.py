"""Qualify a retained NPU partition set through the frozen FPGA renderer.

This command deliberately records failed strict tensor checks. It can issue a
native-image receipt only after finite/stable NPU output, Gaussian repeat,
artifact checks and three held-out FPGA image comparisons all pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

from common import FULL_WEIGHT_SHA256, new_directory, save, sha
from gaussian_generation.npu_branch.acceptance import (
    SCHEMA, QUALITY_LIMITS, PROVENANCE_KEYS, check_receipt)
from gaussian_generation.npu_branch.config import add_arguments, BackendConfig


def read_rgb(path):
    # Import after Torch on the board; OpenCV is already a runtime dependency.
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot read held-out image: ' + str(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb(path, image):
    import cv2
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise ValueError('Cannot write comparison image: ' + str(path))


def image_metrics(actual, expected):
    from evaluate import quality
    q = quality(actual, expected)
    # Identical images have unbounded PSNR. Use a finite documented display cap.
    return dict(psnr_db=120. if q['psnr_db'] is None else min(120., q['psnr_db']),
                ssim=q['ssim_rgb_gaussian_11_sigma1p5_valid'],
                mae=q['mae'], max_abs_error=q['max_abs_error'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='Prepared context + at least three held-out targets')
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--vendor', type=Path, required=True)
    p.add_argument('--renderer', type=Path, required=True)
    p.add_argument('--live-renderer', type=Path, required=True)
    p.add_argument('--boot-image', type=Path, required=True, help='Actual mounted boot partition BOOT.bin')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--out', type=Path, required=True)
    add_arguments(p)
    a = p.parse_args(argv)
    try:
        config = BackendConfig.from_args(a)
        if config.backend != 'npu' or config.npu_acceptance:
            raise ValueError('Qualification requires --backend npu and no existing acceptance receipt')
        if a.threads < 1 or a.threads > 4:
            raise ValueError('Qualification thread count must be 1..4')
    except ValueError as exc:
        p.error(str(exc))
    if platform.machine().lower() not in ('aarch64', 'arm64'):
        p.error('Native NPU + FPGA qualification requires the ARM board')
    out = new_directory(a.out)
    record = dict(schema=SCHEMA, complete=False, diagnostic_only=False,
                  npu_executed=False, fpga_executed=False, quality_limits=QUALITY_LIMITS,
                  partitions=list(config.partitions), weight_sha256=sha(a.weights),
                  views={}, power='not measured', resource_utilization='no FPGA/BOOT change',
                  scope='Fixed prepared input: CPU vs selected retained NPU partitions, same native FPGA renderer; '
                        'not video-to-first-frame timing or proof of quality for every unseen scene')
    try:
        from rendering.runtime import load_mainline_profile
        profile = load_mainline_profile()
        if sha(a.boot_image) != profile['required_hardware']['accepted_boot_sha256']:
            raise ValueError('Mounted BOOT differs from accepted renderer hardware')
        if record['weight_sha256'] != FULL_WEIGHT_SHA256:
            raise ValueError('Checkpoint mismatch')
        meta = json.loads((a.input / 'input.json').read_text(encoding='utf-8'))
        targets = [v for v in meta['views'] if v['role'] == 'target']
        names = [v['name'] for v in targets]
        if len(set(names)) < 3 or len(set(names)) != len(names):
            raise ValueError('Need at least three distinct held-out target views')
        if set(meta.get('context_indices', [])) & set(meta.get('target_indices', [])):
            raise ValueError('Context and target views overlap')
        digest = sha(a.input / 'context.npz')
        if meta['context_sha256'] != digest:
            raise ValueError('Prepared input context checksum mismatch')
        # Torch before OpenCV, including the evaluator's lazy imports.
        from gaussian_generation.runtime import ModelRuntime
        model = ModelRuntime(a.weights, a.vendor, a.threads, 16, out / 'initialize')
        record['source_files'] = model.source_hashes
        import numpy as np
        import cv2
        cv2.setNumThreads(1)
        from gaussian_generation.adapter import camera_bytes, gaussian_rows
        from gaussian_generation.npu_branch.runtime import PartitionRuntime
        from gaussian_generation.npu_branch.readiness import probe_oracles
        from initialize.session import render_environment, worker_environment, protect_renderer_workspace
        from rendering.pipeline_adapter import PipelineRenderer
        cpu = model.infer(a.input, out / 'cpu', digest)
        cpu_g = model.take_gaussians(cpu['gaussian_sha256'])
        cpu_rows, _ = gaussian_rows(cpu_g)
        del cpu_g
        graphs = config.oracle_graphs or config.partition_bundle.parent / 'graphs'
        with PipelineRenderer(a.renderer, a.live_renderer, render_environment(), out / 'renderer.log') as renderer:
            protect_renderer_workspace(renderer, 'npu')
            with PartitionRuntime('npu', config.partition_bundle, config.partitions, out / 'partition_runtime',
                library=config.npu_library, worker_python=config.worker_python, environment=worker_environment(),
                buffer_policy=config.buffer_policy, buffer_limit_mib=config.buffer_limit_mib) as worker:
                record['renderer'] = dict(binary_sha256=sha(a.live_renderer), package=renderer.manifest,
                                          profile=profile, boot_image=str(a.boot_image.resolve()))
                first = probe_oracles(worker, graphs, out / 'oracle_before.json')
                model.attach_partitions(worker)
                hybrid = model.infer(a.input, out / 'npu', digest)
                hybrid_g = model.take_gaussians(hybrid['gaussian_sha256'])
                hybrid_rows, _ = gaussian_rows(hybrid_g)
                cpu_images = {}
                renderer.load_rows(cpu_rows)
                for view in targets:
                    frame = renderer.runtime.render_camera(camera_bytes(view), include_raw=True)
                    cpu_images[view['name']] = frame.rgb.copy()
                del cpu_rows
                renderer.load_rows(hybrid_rows)
                del hybrid_rows
                for index, view in enumerate(targets):
                    name = view['name']
                    from gaussian_generation.npu_branch.artifacts import relative_file
                    truth_path = relative_file(a.input / 'images', name)
                    if sha(truth_path) != view['prepared_sha256']:
                        raise ValueError('Held-out image changed')
                    truth = np.asarray(read_rgb(truth_path), np.float64) / 255.
                    frame = renderer.runtime.render_camera(camera_bytes(view), include_raw=True)
                    frame.archive(out / ('view_' + str(index)))
                    actual, reference = frame.rgb.astype(np.float64)/255., cpu_images[name].astype(np.float64)/255.
                    q_cpu, q_npu = image_metrics(reference, truth), image_metrics(actual, truth)
                    record['views'][name] = dict(npu_vs_cpu=image_metrics(actual, reference),
                        cpu_vs_truth=q_cpu, npu_vs_truth=q_npu,
                        truth_psnr_drop_db=q_cpu['psnr_db']-q_npu['psnr_db'],
                        truth_ssim_drop=q_cpu['ssim']-q_npu['ssim'],
                        target_sha256=sha(truth_path), camera_sha256=hashlib.sha256(camera_bytes(view)).hexdigest(),
                        live_frame_ms=frame.metadata['wall_ms'])
                    panel = np.concatenate((np.rint(truth*255).astype(np.uint8), cpu_images[name], frame.rgb), axis=1)
                    write_rgb(out / ('view_' + str(index) + '_comparison.png'), panel)
                # Reuse after FPGA work, not only two adjacent NPU calls.
                repeat = model.infer(a.input, out / 'npu_repeat', digest)
                repeat_g = model.take_gaussians(repeat['gaussian_sha256'])
                record['gaussian_repeat_exact'] = all(np.array_equal(repeat_g[k], hybrid_g[k]) for k in hybrid_g)
                del repeat_g, hybrid_g
                last = probe_oracles(worker, graphs, out / 'oracle_after.json')
                before = {c['partition']: c['output_sha256'] for c in first['checks'] if c['round'] == 1}
                checks = [c for c in last['checks'] if c['round'] == 1]
                if any(c['output_sha256'] != before[c['partition']] for c in checks):
                    raise ValueError('NPU oracle changed after inference/FPGA execution')
                record.update(npu_executed=hybrid['npu_executed'] and repeat['npu_executed'],
                    fpga_executed=renderer.runtime.configuration['backend'] == 'fpga',
                    oracle_checks=checks, strict_tensor_passed=last['strict_passed'],
                    runtime_identity=worker.readiness['runtime_identity'],
                    provenance={k: worker.provenance[k] for k in PROVENANCE_KEYS},
                    context_sha256=digest, input_manifest_sha256=sha(a.input/'input.json'),
                    cpu=cpu, npu=hybrid, npu_repeat=repeat,
                    prepared_inference_saving_seconds=cpu['inference_seconds']-hybrid['inference_seconds'],
                    qualified_at_unix=time.time(), psnr_infinity_display_cap_db=120.)
                record['complete'] = True
                check_receipt(record)
    except BaseException as exc:
        record.update(complete=False, error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        save(out / 'acceptance.json', record)
    print(json.dumps(dict(complete=True, acceptance=str(out / 'acceptance.json'),
                          prepared_inference_saving_seconds=record['prepared_inference_saving_seconds'])), flush=True)


if __name__ == '__main__':
    main()
