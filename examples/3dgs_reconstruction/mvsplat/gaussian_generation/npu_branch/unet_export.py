"""Export full depth UNets through a temporary inference-only checkpoint bypass.

This is an explicit candidate exporter, not a production graph replacement.
The original weights, AttentionBlock._forward, normalization and residual paths
are retained. All instance forward methods are restored on leaving the scope.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import gc
from pathlib import Path
import time
import types

import numpy as np

from common import SOURCE_COMMIT, new_directory, save, sha


MODULES = ('corr_refine_net', 'refine_unet')


@contextmanager
def bypass_attention_checkpoint(module):
    """Temporarily call the same AttentionBlock implementation, without autograd.

    Patch only known AttentionBlock instances on an eval-only model. Do not
    mutate class methods, _forward, parameters, use_checkpoint flags or globals.
    """
    children = list(module.named_modules())
    if any(child.training for _, child in children):
        raise ValueError('Checkpoint bypass is restricted to eval-only export')
    selected = [(name, child) for name, child in children
                if type(child).__name__ == 'AttentionBlock'
                and callable(getattr(child, '_forward', None))]
    if not selected:
        raise ValueError('Full UNet has no checkpoint-backed AttentionBlock')
    previous = []

    def direct(self, *args, **kwargs):
        return self._forward(*args, **kwargs)

    try:
        for name, child in selected:
            previous.append((child, 'forward' in child.__dict__, child.__dict__.get('forward')))
            child.forward = types.MethodType(direct, child)
        yield [name for name, _ in selected]
    finally:
        for child, had_instance_method, value in reversed(previous):
            if had_instance_method:
                child.forward = value
            else:
                delattr(child, 'forward')


def output_wrapper(module, layout):
    """Make ONNX result axis order explicit; logical runtime remains NCHW."""
    import torch
    if layout not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported export output axis order')

    class Unary(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, value):
            result = self.inner(value)
            return result.permute(0, 2, 3, 1).contiguous() if layout == 'NHWC' else result

    return Unary(module).eval()


def numerical_comparison(actual, expected, *, rtol, atol):
    """Reject nonfinite/shape mismatches and retain strict failure evidence."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError('UNet output shape mismatch')
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    if not finite:
        return dict(passed=False, finite=False, exact=False,
                    tolerance=dict(rtol=rtol, atol=atol))
    delta = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    failed = delta > atol+rtol*np.abs(expected)
    return dict(passed=not bool(failed.any()), finite=True,
                exact=bool(np.array_equal(actual, expected)),
                max_abs_error=float(delta.max(initial=0)),
                mean_abs_error=float(delta.mean()),
                rmse=float(np.sqrt(np.mean(delta*delta))),
                outside_tolerance_fraction=float(failed.mean()),
                tolerance=dict(rtol=rtol, atol=atol))


def _fixtures(path, name, torch):
    with np.load(Path(path)/(name+'.npz'), allow_pickle=False) as fixture:
        x = torch.from_numpy(fixture['input'].copy())
        expected = fixture['expected'].copy()
    if (x.ndim != 4 or x.dtype != torch.float32 or not np.isfinite(x.numpy()).all()
            or expected.ndim != 4 or expected.dtype != np.float32
            or not np.isfinite(expected).all()):
        raise ValueError('Expected finite unary FP32 NCHW fixture')
    return x, expected


def capture_inputs(runtime, context_path, names):
    """One real board reference forward; callers may provide cached fixtures."""
    torch = runtime.torch
    captured, handles = {}, []
    def capture(name):
        def hook(module, args, output):
            if name in captured or len(args) != 1 or not isinstance(output, torch.Tensor):
                raise ValueError('UNet is not a one-call unary partition: '+name)
            captured[name] = (args[0].detach().clone(), output.detach().numpy().copy())
        return hook
    for name in names:
        handles.append(getattr(runtime.model.depth_predictor, name).register_forward_hook(capture(name)))
    try:
        with np.load(context_path, allow_pickle=False) as data:
            context = {key:torch.from_numpy(data[key].copy()).unsqueeze(0) for key in data.files}
        with torch.inference_mode():
            g = runtime.model(context, global_step=300000, deterministic=True)
        del g, context
    finally:
        for handle in handles:
            handle.remove()
    return captured


def export_module(module, x, expected, folder, *, output_layout, probe_original, second=None):
    import torch
    import onnx
    import onnxruntime as ort
    record = dict(complete=False, npu_executed=False, full_module=True,
                  input_shape=list(x.shape), output_shape=list(expected.shape),
                  logical_layout='NCHW', onnx_output_layout=output_layout,
                  dtype='float32', returns_list=False,
                  original_parameters=sum(p.numel() for p in module.parameters()),
                  production_forward_changed=False,
                  onnx_reference_worker_compatible=output_layout == 'NCHW')
    save(folder/'export_result.json', record)
    wrapped = output_wrapper(module, output_layout)
    with torch.inference_mode():
        original = module(x).detach().numpy()
    record['fixture_reference'] = numerical_comparison(original, expected, rtol=1e-5, atol=1e-6)
    save(folder/'export_result.json', record)
    if not record['fixture_reference']['passed']:
        raise ValueError('Captured fixture is not this original UNet output')
    if probe_original:
        original_graph = folder/'original_checkpoint_probe.onnx'
        try:
            with torch.inference_mode():
                torch.onnx.export(wrapped, (x,), str(original_graph), input_names=['features'],
                                  output_names=['result'], opset_version=17, dynamo=False)
            onnx.checker.check_model(str(original_graph))
            record['original_export_probe'] = dict(exported=True, graph_sha256=sha(original_graph))
        except Exception as exc:
            record['original_export_probe'] = dict(exported=False, error=type(exc).__name__+': '+str(exc))
        save(folder/'export_result.json', record)
    with bypass_attention_checkpoint(module) as patched:
        record['bypassed_attention_blocks'] = patched
        checks = []
        for label, value, reference in [('A', x, original)]+([] if second is None else [('B', *second)]):
            with torch.inference_mode():
                actual = module(value).detach().numpy()
            check = numerical_comparison(actual, reference, rtol=1e-5, atol=1e-6)
            checks.append(dict(input=label, **check))
            if not check['passed']:
                record['bypass_checks'] = checks
                save(folder/'export_result.json', record)
                raise ValueError('Checkpoint bypass differs on input '+label)
        record['bypass_checks'] = checks
        save(folder/'export_result.json', record)
        graph = folder/'model.onnx'
        started = time.perf_counter()
        with torch.inference_mode():
            torch.onnx.export(wrapped, (x,), str(graph), input_names=['features'],
                              output_names=['result'], opset_version=17, dynamo=False)
        record['export_seconds'] = time.perf_counter()-started
    record['checkpoint_scope_restored'] = True
    onnx.checker.check_model(str(graph))
    model = onnx.load(str(graph), load_external_data=False)
    record['ops'] = dict(Counter((node.domain+':' if node.domain else '')+node.op_type
                                 for node in model.graph.node))
    del model
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(graph), sess_options=options, providers=['CPUExecutionProvider'])
    started = time.perf_counter()
    wire = session.run(None, {'features':x.numpy()})[0]
    record['host_onnx_seconds'] = time.perf_counter()-started
    record['onnx_output_shape'] = list(wire.shape)
    actual = wire.transpose(0,3,1,2) if output_layout == 'NHWC' else wire
    check = numerical_comparison(actual, expected, rtol=2e-3, atol=2e-4)
    del session
    gc.collect()
    record['host_onnx_check'] = check
    record['host_onnx_passed'] = check['passed']
    np.savez(folder/'oracle.npz', input=x.numpy(), expected=expected)
    record.update(complete=check['passed'], parameters=record['original_parameters'],
                  graph_sha256=sha(graph), oracle_sha256=sha(folder/'oracle.npz'),
                  max_abs_error=check.get('max_abs_error'), mean_abs_error=check.get('mean_abs_error'),
                  tolerance=dict(rtol=2e-3, atol=2e-4),
                  status='exported_host_verified_not_compiled' if check['passed'] else 'host_onnx_failed')
    save(folder/'export_result.json', record)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--context', required=True, type=Path, help='Actual context.npz; provenance even with fixtures')
    p.add_argument('--weights', required=True, type=Path)
    p.add_argument('--vendor', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--fixtures', type=Path, help='Cached MODULE.npz with input/expected; avoids another full forward')
    p.add_argument('--fixtures-b', type=Path, help='Optional second actual context fixtures, not a generated perturbation')
    p.add_argument('--threads', type=int, default=3)
    p.add_argument('--modules', nargs='+', choices=MODULES, default=list(MODULES))
    p.add_argument('--output-layout', choices=('NCHW','NHWC'), default='NHWC')
    p.add_argument('--probe-original-export', action='store_true')
    a = p.parse_args(argv)
    if len(set(a.modules)) != len(a.modules):
        p.error('Select each UNet at most once')
    out = new_directory(a.out)
    from gaussian_generation.runtime import ModelRuntime
    runtime = ModelRuntime(a.weights, a.vendor, a.threads, 16, out/'runtime')
    with np.load(a.context, allow_pickle=False) as data:
        input_shape = [1, *data['image'].shape]
    manifest = dict(schema='mvsplat_partitions_v1', diagnostic_only=True, npu_executed=False,
        weight_sha256=runtime.weight_hash, context_sha256=sha(a.context),
        source_commit=SOURCE_COMMIT, exporter_sha256=sha(Path(__file__)),
        input_shape=input_shape, source_hashes=runtime.source_hashes,
        bypass_scope='only temporary AttentionBlock.forward calls existing _forward in eval export',
        output_contract='NCHW logical oracle; explicit ONNX output order recorded per partition',
        partitions={})
    captured = {} if a.fixtures else capture_inputs(runtime, a.context, a.modules)
    for name in a.modules:
        folder = new_directory(out/name)
        record = dict(complete=False, full_module=True, npu_executed=False,
                      path='depth_predictor.'+name)
        manifest['partitions'][name] = record
        save(out/'manifest.json', manifest)
        try:
            x, expected = _fixtures(a.fixtures, name, runtime.torch) if a.fixtures else captured.pop(name)
            second = _fixtures(a.fixtures_b, name, runtime.torch) if a.fixtures_b else None
            module = getattr(runtime.model.depth_predictor, name).eval()
            item = export_module(module, x, expected, folder,
                output_layout=a.output_layout, probe_original=a.probe_original_export, second=second)
            record.update(item)
            del x, expected, second
        except Exception as exc:
            partial = folder/'export_result.json'
            if partial.is_file():
                import json
                record.update(json.loads(partial.read_text()))
            record.update(complete=False, status='export_failed', error=type(exc).__name__+': '+str(exc))
        finally:
            gc.collect()
            save(folder/'export_result.json', record)
            save(out/'manifest.json', manifest)
            import json
            print(json.dumps(dict(module=name, complete=record['complete'],
                                  status=record.get('status'),error=record.get('error'))),flush=True)
    if not all(item['complete'] for item in manifest['partitions'].values()):
        raise RuntimeError('Full UNet export candidate failed; see per-module manifest')


if __name__ == '__main__':
    main()
