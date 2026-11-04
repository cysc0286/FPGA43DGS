"""Export six original-weight FFNs as 1x1 graphs, with real captured oracles.

Run full-model capture on the board, or supply --fixtures from a single board
reference capture. This is export/host-equivalence evidence, never NPU acceptance.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from common import new_directory, save, sha
from gaussian_generation.npu_branch.ffn import (
    FFN_PARTITIONS, PointwiseFFN, adapter_spec, locate_ffn, token_to_image,
)


def capture_inputs(runtime, input_dir):
    """One unchanged CPU reference forward, retaining only six FFN boundaries."""
    samples, hooks = {}, []
    for name in FFN_PARTITIONS:
        def capture(module, args, result, name=name):
            if name in samples or len(args) != 1:
                raise ValueError('FFN must execute exactly once with one token input')
            samples[name] = (args[0].detach().clone(), result.detach().clone())
        hooks.append(locate_ffn(runtime.model, name).register_forward_hook(capture))
    try:
        with runtime.np.load(Path(input_dir)/'context.npz', allow_pickle=False) as f:
            context = {k:runtime.torch.from_numpy(f[k].copy()).unsqueeze(0) for k in f.files}
        with runtime.torch.inference_mode():
            result = runtime.model(context, global_step=300000, deterministic=True)
        del result, context
    finally:
        for hook in hooks:
            hook.remove()
    if set(samples) != set(FFN_PARTITIONS):
        raise ValueError('Expected all six FFN boundaries in the actual forward')
    return samples


def _image(torch, value, height, width):
    tensor = torch.from_numpy(value.copy()) if isinstance(value, np.ndarray) else value
    if tensor.ndim == 3:
        return token_to_image(tensor, height, width, tensor.shape[2])
    if tensor.ndim != 4 or list(tensor.shape[2:]) != [height, width] or tensor.dtype != torch.float32:
        raise ValueError('Fixture is neither the reviewed token nor NCHW boundary')
    return tensor


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='Real prepared context, binding context SHA256')
    p.add_argument('--fixtures', type=Path, help='Six transformer_ffn_i.npz files with input/expected arrays')
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--vendor', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--threads', type=int, default=3)
    p.add_argument('--batch', type=int, choices=(1,2), default=2)
    p.add_argument('--tile-size', type=int, default=32)
    p.add_argument('--output-layout', choices=('NCHW','NHWC'), default='NHWC')
    p.add_argument('--partitions', nargs='+', choices=list(FFN_PARTITIONS), default=list(FFN_PARTITIONS))
    a = p.parse_args(argv)
    if not 1 <= a.tile_size <= 32 or len(set(a.partitions)) != len(a.partitions):
        p.error('Need unique selected FFNs and a tile size in 1..32')
    with np.load(a.input/'context.npz', allow_pickle=False) as context:
        if list(context['image'].shape) != [2,3,128,128]:
            raise ValueError('FFN export is bound to two 128x128 RGB context views')
    from gaussian_generation.runtime import ModelRuntime
    out = new_directory(a.out)
    runtime = ModelRuntime(a.weights, a.vendor, a.threads, 16, out/'runtime')
    torch = runtime.torch
    import onnx
    import onnxruntime as ort
    captured = None if a.fixtures else capture_inputs(runtime, a.input)
    manifest = dict(schema='mvsplat_partitions_v1', npu_executed=False,
        weight_sha256=runtime.weight_hash, context_sha256=sha(a.input/'context.npz'),
        source_commit=__import__('common').SOURCE_COMMIT, input_shape=[1,2,3,128,128],
        source_files=runtime.source_hashes, torch_version=torch.__version__, onnx_version=onnx.__version__,
        onnxruntime_version=ort.__version__,
        scope='Original checkpoint FFN export and host equivalence only; no NPU acceptance', partitions={})

    class ExportLayout(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, value):
            result = self.inner(value)
            return result.permute(0,2,3,1) if a.output_layout == 'NHWC' else result

    for name in a.partitions:
        folder = new_directory(out/name)
        item = dict(path=FFN_PARTITIONS[name], complete=False, returns_list=False,
                    dtype='float32', logical_layout='NCHW', onnx_output_layout=a.output_layout)
        try:
            if a.fixtures:
                fixture = a.fixtures/(name+'.npz')
                with np.load(fixture, allow_pickle=False) as f:
                    full_in = _image(torch, f['input'], 32, 32)
                    full_out = _image(torch, f['expected'], 32, 32)
                item['fixture_sha256'] = sha(fixture)
            else:
                tokens, expected_tokens = captured[name]
                full_in = _image(torch, tokens, 32, 32)
                full_out = _image(torch, expected_tokens, 32, 32)
            if list(full_in.shape) != [2,256,32,32] or list(full_out.shape) != [2,128,32,32]:
                raise ValueError('Unexpected actual FFN dimensions')
            x = full_in[:a.batch, :, :a.tile_size, :a.tile_size].contiguous()
            expected = full_out[:a.batch, :, :a.tile_size, :a.tile_size].contiguous()
            pointwise = PointwiseFFN(locate_ffn(runtime.model, name))
            with torch.inference_mode():
                converted = pointwise(x)
            if not torch.allclose(converted, expected, rtol=2e-3, atol=2e-4):
                raise ValueError('Original-weight 1x1 conversion differs from captured CPU FFN')
            item['pointwise_max_abs_error'] = float((converted-expected).abs().max())
            module = ExportLayout(pointwise).eval()
            graph = folder/'model.onnx'
            with torch.inference_mode():
                torch.onnx.export(module, (x,), str(graph), input_names=['features'],
                    output_names=['result'], opset_version=17, dynamo=False)
            onnx.checker.check_model(str(graph))
            options = ort.SessionOptions()
            options.intra_op_num_threads = a.threads
            session = ort.InferenceSession(str(graph), sess_options=options, providers=['CPUExecutionProvider'])
            started = time.perf_counter()
            actual = session.run(None, {'features':x.numpy()})[0]
            elapsed = time.perf_counter()-started
            if a.output_layout == 'NHWC':
                actual = actual.transpose(0,3,1,2)
            delta = actual-expected.numpy()
            passed = bool(np.isfinite(actual).all() and np.allclose(actual, expected.numpy(), rtol=2e-3, atol=2e-4))
            np.savez(folder/'oracle.npz', input=x.numpy(), expected=expected.numpy())
            # Input-switch probe uses the other real view. For batch two,
            # reverse view order: a true pointwise FFN is permutation-equivariant.
            alternate_in = full_in.flip(0)[:a.batch,:, :a.tile_size,:a.tile_size].contiguous()
            alternate_out = full_out.flip(0)[:a.batch,:, :a.tile_size,:a.tile_size].contiguous()
            if torch.equal(alternate_in, x):
                raise ValueError('Real FFN fixture cannot distinguish input switches')
            np.savez(folder/'oracle_alternate.npz', input=alternate_in.numpy(), expected=alternate_out.numpy())
            item.update(complete=passed, input_shape=list(x.shape), output_shape=list(expected.shape),
                ffn_adapter=adapter_spec(name, full_in.shape, full_out.shape, x.shape, expected.shape),
                graph_sha256=sha(graph), oracle_sha256=sha(folder/'oracle.npz'),
                alternate_oracle_sha256=sha(folder/'oracle_alternate.npz'),
                parameters=sum(v.numel() for v in pointwise.parameters()),
                host_onnx_seconds=elapsed, max_abs_error=float(np.abs(delta).max()),
                mean_abs_error=float(np.abs(delta).mean()), host_onnx_passed=passed,
                tolerance=dict(rtol=2e-3,atol=2e-4), npu_executed=False)
            del x, expected, converted, full_in, full_out, pointwise, module, session
        except Exception as exc:
            item['error'] = type(exc).__name__+': '+str(exc)
        manifest['partitions'][name] = item
        save(out/'manifest.json', manifest)
        print(json.dumps(dict(name=name, **item)), flush=True)
    if not all(v['complete'] for v in manifest['partitions'].values()):
        raise RuntimeError('One or more FFN exports failed; see manifest.json')


if __name__ == '__main__':
    main()
