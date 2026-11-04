"""Alternate full board inference paths on one retained model.

This is a timing diagnostic, not a native image acceptance receipt or a video
benchmark. Each case owns only its selected NPU worker; closure restores every
CPU module before a subsequent case creates any device sessions.
"""
import argparse
import gc
import hashlib
import math
from pathlib import Path
import platform
import time

from common import new_directory, save, sha
from gaussian_generation.npu_branch.catalog import semantic_coverage, validate_selection


# Original encoder execution order, independent of CLI/session creation order.
MODEL_ORDER = ('backbone_cnn', *('transformer_ffn_'+str(i) for i in range(6)),
               'corr_depth_pipeline', 'corr_refine_net', 'regressor_residual', 'depth_head', 'upsampler',
               'refine_heads_pipeline', 'proj_feature', 'refine_unet', 'fused_heads', 'to_gaussians', 'to_disparity')


def comparison_cases(partitions, baseline, rounds):
    """Reject a bad comparison before constructing a model or device worker."""
    expanded = validate_selection(partitions)
    if type(rounds) is not int or rounds < 2:
        raise ValueError('Requires at least two alternating rounds')
    if baseline is None:
        if 'backbone_cnn' not in expanded:
            raise ValueError('Paired CPU/backbone comparison requires backbone_cnn')
        cases = [('cpu', []), ('backbone_only', ['backbone_cnn']), ('expanded', expanded)]
    else:
        baseline = validate_selection(baseline)
        # A fused graph changes identity while covering the same operations.
        # Permit that explicit comparison, but reject lost coverage or a
        # permutation of the same selection masquerading as an expansion.
        if not semantic_coverage(baseline) <= semantic_coverage(expanded) or set(baseline) == set(expanded):
            raise ValueError('Baseline semantic coverage must be preserved by a distinct expanded selection')
        cases = [('baseline', baseline), ('expanded', expanded)]
    return [(index, label, list(enabled)) for index in range(rounds)
            for label, enabled in (cases if index % 2 == 0 else list(reversed(cases)))]


def expected_calls(parts):
    """Expand actual adapter/tile calls, without assuming one call per module."""
    from gaussian_generation.npu_branch.ffn_contract import validate_adapter
    from gaussian_generation.npu_branch.spatial import validate_tiling
    if set(parts) - set(MODEL_ORDER):
        raise ValueError('No model execution order for selected partition')
    if parts:
        validate_selection(list(parts))
    expected = []
    for name in MODEL_ORDER:
        if name not in parts:
            continue
        meta = parts[name]['meta']
        count = 1
        spatial = validate_tiling(meta)
        if spatial:
            batch, _, height, width = spatial['input_shape']
            core = spatial['core_size']
            count = batch * math.ceil(height/core) * math.ceil(width/core)
        elif meta.get('ffn_adapter'):
            full = validate_adapter(meta)['full_input_shape']
            tile = meta['input_shape']
            count = full[0]//tile[0] * math.ceil(full[2]/tile[2]) * math.ceil(full[3]/tile[3])
        expected.extend([name] * count)
    return expected


def check_calls(calls, parts, first, identity, inference_seconds):
    """Check model order, tile counts, sequence and all charged SDK passes."""
    expected = expected_calls(parts)
    if [c.get('name') for c in calls] != expected:
        raise ValueError('Actual NPU call order/count differs from model adapters')
    def seconds(value):
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid measured NPU/inference seconds')
        return value
    elapsed = seconds(inference_seconds)
    sdk_total, partition_total, passes = 0., 0., 0
    for offset, call in enumerate(calls):
        name, meta = call['name'], parts[call['name']]['meta']
        expected_passes = 2 if (meta.get('ffn_adapter') or name in ('corr_refine_net', 'refine_unet', 'corr_depth_pipeline', 'refine_heads_pipeline') or
            (meta.get('spatial_tiling') and name in ('to_gaussians', 'to_disparity', 'fused_heads'))) else (
                identity.get('backbone_settle_passes', 1) if name == 'backbone_cnn' else 1)
        if ((call.get('event'), call.get('backend'), call.get('npu_executed'), call.get('sequence')) !=
                ('COMPLETE', 'npu', True, first+offset) or
                call.get('sdk_execution_passes') != expected_passes):
            raise ValueError('NPU sequence/backend/charged SDK pass contract failed')
        names, measured = call.get('sdk_stage_names'), call.get('sdk_seconds', [])
        if names == ['input_write', 'forward_submit', 'wait', 'output_convert', 'sdk_total']:
            if len(measured) != 5:
                raise ValueError('Detailed SDK bucket count differs')
            values = [seconds(v) for v in measured]
            cost = values[4]
            if abs(cost-sum(values[:4])) > 1e-5:
                raise ValueError('Detailed SDK total omits charged stages')
        elif names == ['input_write', 'execute_wait', 'output_convert']:
            if len(measured) != 3:
                raise ValueError('Legacy SDK bucket count differs')
            cost = sum(seconds(v) for v in measured)
        else:
            raise ValueError('Unknown SDK timing buckets')
        total = seconds(call.get('total_seconds'))
        if cost > total + 1e-5:
            raise ValueError('Partition wall time omits charged SDK cost')
        sdk_total += cost
        partition_total += total
        passes += expected_passes
    if partition_total > elapsed + 1e-5:
        raise ValueError('Full inference wall time omits partition work')
    return dict(logical_calls=len(calls), calls_per_partition={n: expected.count(n) for n in parts},
                sdk_forward_calls=passes, sdk_seconds=sdk_total, partition_seconds=partition_total)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'weights', 'vendor', 'bundle', 'graphs', 'library', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--partitions', nargs='+', required=True)
    p.add_argument('--baseline-partitions', nargs='+',
                   help='Explicit subset for baseline/expanded ABBA; otherwise CPU/backbone/expanded')
    p.add_argument('--worker-timeout', type=float, default=90,
                   help='Per-worker response deadline, including initial session loading')
    p.add_argument('--threads', type=int, default=3)
    p.add_argument('--rounds', type=int, default=2)
    a = p.parse_args(argv)
    try:
        cases = comparison_cases(a.partitions, a.baseline_partitions, a.rounds)
        if not math.isfinite(a.worker_timeout) or a.worker_timeout <= 0 or not 1 <= a.threads <= 4:
            raise ValueError('Use a positive worker timeout and 1..4 CPU threads')
    except ValueError as exc:
        p.error(str(exc))
    if platform.machine().lower() not in ('aarch64', 'arm64'):
        p.error('Requires the ARM board')
    out = new_directory(a.out)
    record = dict(complete=False, diagnostic_only=True, acceptance_receipt=False, cases=[],
                  threads=a.threads, worker_timeout=a.worker_timeout,
                  scope='Full network wall time includes remaining CPU compute, layout/IPC and every SDK pass; '
                        'fixed prepared input, no video preparation or FPGA rendering; '
                        'worker loading and pre/post oracle probes recorded outside inference')
    try:
        from gaussian_generation.runtime import ModelRuntime
        from gaussian_generation.npu_branch.runtime import PartitionRuntime
        from gaussian_generation.npu_branch.readiness import probe_oracles
        from initialize.session import worker_environment
        model_options = ({'corr_depth_pipeline': True}
            if any('corr_depth_pipeline' in enabled for _, _, enabled in cases) else {})
        if any('refine_heads_pipeline' in enabled for _, _, enabled in cases):
            model_options['refine_heads_pipeline'] = True
        model = ModelRuntime(a.weights, a.vendor, a.threads, 16, out/'initialize', **model_options)
        np, torch = model.np, model.torch
        record['model_load_seconds'] = getattr(model, 'load_seconds', None)
        record['source_files'] = getattr(model, 'source_hashes', None)
        record['context_sha256'] = sha(a.input/'context.npz')
        with np.load(a.input/'context.npz', allow_pickle=False) as data:
            context = {k: torch.from_numpy(data[k].copy()).unsqueeze(0) for k in data.files}
        for ordinal, (index, label, enabled) in enumerate(cases):
            if model.partition_runtime is not None:
                raise RuntimeError('Previous case still owns NPU sessions')
            folder = new_directory(out/(str(ordinal)+'_'+label))
            result = dict(round=index, label=label, enabled=list(enabled), complete=False)
            record['cases'].append(result)
            worker = None
            try:
                if enabled:
                    start = time.perf_counter()
                    worker = PartitionRuntime('npu', a.bundle, enabled, folder/'worker',
                        library=a.library, environment=worker_environment(), timeout=a.worker_timeout)
                    result['worker_load_seconds'] = time.perf_counter()-start
                    result['provenance'] = worker.provenance
                    result['runtime_identity'] = worker.readiness['runtime_identity']
                    start = time.perf_counter()
                    result['oracle_before'] = probe_oracles(worker, a.graphs, folder/'oracle_before.json')
                    result['oracle_before_seconds'] = time.perf_counter()-start
                    start = time.perf_counter()
                    model.attach_partitions(worker)
                    result['attach_seconds'] = time.perf_counter()-start
                first = worker.sequence if worker else 0
                start = time.perf_counter()
                with torch.inference_mode():
                    g = model.model(context, global_step=300000, deterministic=True)
                elapsed = time.perf_counter()-start
                arrays = {k: getattr(g,k).detach().numpy() for k in
                          ('means','covariances','harmonics','opacities')}
                finite = all(np.isfinite(v).all() for v in arrays.values())
                result.update(inference_seconds=elapsed, finite=finite,
                    gaussian_hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()})
                del g, arrays
                if not finite:
                    raise ValueError('Gaussian finite check failed')
                calls = worker.report(first)['calls'] if worker else []
                result.update(calls=calls, npu_executed=bool(calls))
                result['call_accounting'] = check_calls(
                    calls, worker.parts if worker else {}, first,
                    worker.readiness['runtime_identity'] if worker else {}, elapsed)
                if worker:
                    start = time.perf_counter()
                    result['oracle_after'] = probe_oracles(worker, a.graphs, folder/'oracle_after.json')
                    result['oracle_after_seconds'] = time.perf_counter()-start
                    before, after = result['oracle_before'], result['oracle_after']
                    def hashes(probe):
                        checks = [c for c in probe['checks'] if c['round'] == 1]
                        if (not probe['complete'] or len(checks) != len(enabled) or
                                {c['partition'] for c in checks} != set(enabled)):
                            raise ValueError('Incomplete pre/post oracle probe')
                        return {c['partition']: c['output_sha256'] for c in checks}
                    result['strict_tensor_passed'] = before['strict_passed'] and after['strict_passed']
                    result['oracle_output_stable'] = hashes(before) == hashes(after)
                    if not result['oracle_output_stable']:
                        raise ValueError('NPU oracle changed after full inference')
                result['complete'] = True
            except BaseException as exc:
                result['error'] = type(exc).__name__+': '+str(exc)
                raise
            finally:
                start = time.perf_counter()
                try:
                    if worker:
                        if model.partition_runtime is worker:
                            model.detach_partitions()
                        else:
                            worker.close()
                        if model.partition_runtime is not None:
                            raise RuntimeError('Model restoration left an attached NPU worker')
                    gc.collect()
                except BaseException as exc:
                    result.update(complete=False, close_error=type(exc).__name__+': '+str(exc))
                    raise
                finally:
                    result['close_seconds'] = time.perf_counter()-start
                    save(out/'paired.json', record)
            print(label, index, elapsed, flush=True)
        record['repeat_exact'] = all(len({tuple(sorted(c['gaussian_hashes'].items()))
            for c in record['cases'] if c['label'] == label}) == 1 for label in {c[1] for c in cases})
        if not record['repeat_exact']:
            raise ValueError('Repeated full network output changed')
        if sha(a.input/'context.npz') != record['context_sha256']:
            raise ValueError('Prepared context changed during paired inference')
        record['strict_tensor_passed'] = all(c.get('strict_tensor_passed', True) for c in record['cases'])
        record['complete'] = True
    except BaseException as exc:
        record['error'] = type(exc).__name__+': '+str(exc)
        raise
    finally:
        save(out/'paired.json', record)


if __name__ == '__main__':
    main()
