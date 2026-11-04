"""Capture and export original U-Net boundaries for isolated NPU diagnosis.

This tool never installs a production partition. It preserves both views,
normalization domains, attention, weights and skip connections. Block graphs
consume the actual captured block input; prefix graphs start at the original
network input and execute every preceding dependency, including skip branches.
Run the pretrained reference only on the board, under measure.py protections.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import fnmatch
import gc
import json
from pathlib import Path
import time

import numpy as np

from common import SOURCE_COMMIT, new_directory, save, sha
from .unet_export import MODULES, bypass_attention_checkpoint, numerical_comparison


@dataclass(frozen=True)
class Boundary:
    name: str
    path: str
    group: str
    prefix: bool = True


def boundary_plan(module, include_layers=False):
    """Only accept the actual unconditional unary sequential MVSplat topology."""
    import torch
    if not isinstance(module, torch.nn.Sequential):
        raise ValueError('Expected a top-level Sequential depth network')
    matches = [(i, child) for i, child in enumerate(module)
               if all(hasattr(child, key) for key in
                      ('input_blocks', 'middle_block', 'output_blocks', 'out', 'dtype'))]
    if len(matches) != 1:
        raise ValueError('Expected exactly one U-Net in the depth network')
    index, unet = matches[0]
    if (getattr(unet, 'cross_attn_condition', False)
            or getattr(unet, 'num_classes', None) is not None
            or getattr(unet, 'predict_codebook_ids', False)):
        raise ValueError('Conditional/codebook U-Net needs an explicit multi-input contract')
    plan = [Boundary('stem_%02d' % i, str(i), 'stem') for i in range(index)]
    for group, attr in [('down', 'input_blocks'), ('middle', 'middle_block'),
                        ('up', 'output_blocks'), ('out', 'out')]:
        for i, _ in enumerate(getattr(unet, attr)):
            plan.append(Boundary(group+'_%02d' % i, f'{index}.{attr}.{i}', group))
    plan.append(Boundary('unet_complete', str(index), 'unet'))
    plan.extend(Boundary('tail_%02d' % i, str(i), 'tail')
                for i in range(index+1, len(module)))
    plan.append(Boundary('full', '', 'full'))
    if include_layers:
        # Attention uses rank-three qkv/proj/GroupNorm internally. Do not pass
        # those through the current rank-four ABI or silently reshape them.
        attention_paths = [path for path, child in module.named_modules()
                           if type(child).__name__ == 'AttentionBlock']
        existing = {node.path for node in plan}
        kinds = (torch.nn.Conv2d, torch.nn.GroupNorm, torch.nn.GELU, torch.nn.SiLU)
        for path, child in module.named_modules():
            if (path not in existing and isinstance(child, kinds)
                    and not any(path == a or path.startswith(a+'.') for a in attention_paths)):
                plan.append(Boundary('layer_'+path.replace('.', '_'), path, 'layer', prefix=False))
    return index, plan


def select_boundaries(plan, patterns):
    if not patterns:
        return list(plan)
    selected = [node for node in plan
                if any(fnmatch.fnmatchcase(node.name, pattern) for pattern in patterns)]
    unmatched = [pattern for pattern in patterns
                 if not any(fnmatch.fnmatchcase(node.name, pattern) for node in plan)]
    if unmatched:
        raise ValueError('Unknown boundary pattern: '+', '.join(unmatched))
    return selected


def node_module(module, node):
    return module.get_submodule(node.path) if node.path else module


def prefix_module(module, target):
    """Replay the source forward up to one boundary, with all skip dependencies."""
    import torch
    index, plan = boundary_plan(module)
    if target not in {node.name for node in plan}:
        raise ValueError('Unknown U-Net prefix endpoint: '+target)

    class Prefix(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.original = module

        def forward(self, value):
            h = value
            for i in range(index):
                h = self.original[i](h)
                if target == 'stem_%02d' % i:
                    return h
            unet = self.original[index]
            unet_input_dtype = h.dtype
            h = h.type(unet.dtype)
            skips = []
            for i, block in enumerate(unet.input_blocks):
                h = block(h)
                skips.append(h)
                if target == 'down_%02d' % i:
                    return h
            for i, block in enumerate(unet.middle_block):
                h = block(h)
                if target == 'middle_%02d' % i:
                    return h
            for i, block in enumerate(unet.output_blocks):
                h = torch.cat((h, skips.pop()), dim=1)
                h = block(h)
                if target == 'up_%02d' % i:
                    return h
            h = h.type(unet_input_dtype)
            for i, block in enumerate(unet.out):
                h = block(h)
                if target == 'out_%02d' % i:
                    return h
            if target == 'unet_complete':
                return h
            for i in range(index+1, len(self.original)):
                h = self.original[i](h)
                if target == 'tail_%02d' % i:
                    return h
            return h

    return Prefix().eval()


def checked_array(value, name):
    import torch
    if not isinstance(value, torch.Tensor):
        raise ValueError('Non-tensor boundary '+name)
    if value.ndim != 4 or value.dtype != torch.float32:
        raise ValueError('Boundary needs finite FP32 NCHW rank four: '+name)
    array = value.detach().cpu().numpy()
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite boundary '+name)
    return array


def fixture_path(root, name):
    root = Path(root)
    choices = [root] if root.is_file() else [root/(name+'.npz'), root/name/'oracle.npz']
    found = [path for path in choices if path.is_file()]
    if len(found) != 1:
        raise ValueError('Expected exactly one input/expected fixture for '+name)
    return found[0]


def read_fixture(path):
    with np.load(path, allow_pickle=False) as data:
        value, expected = data['input'].copy(), data['expected'].copy()
    for name, array in [('input', value), ('expected', expected)]:
        if array.ndim != 4 or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError('Invalid FP32 NCHW fixture '+name)
    return value, expected


def capture_boundaries(module, value, expected, out, *, label, plan=None):
    """Stream each hook's input/output to disk; never retain all tensors in RAM."""
    import torch
    if any(child.training for child in module.modules()):
        raise ValueError('Capture requires eval mode')
    _, complete_plan = boundary_plan(module)
    plan = complete_plan if plan is None else plan
    out = Path(out)
    records, handles = {}, []

    def hook(node):
        def record(inner, args, result):
            if node.name in records or len(args) != 1:
                raise ValueError('Boundary must be called exactly once: '+node.name)
            x = checked_array(args[0], node.name+' input')
            y = checked_array(result, node.name+' output')
            folder = out/node.name
            folder.mkdir(parents=True, exist_ok=True)
            path = folder/('oracle_'+label.lower()+'.npz')
            if path.exists():
                raise FileExistsError('Do not overwrite captured oracle: '+str(path))
            np.savez(path, input=x, expected=y)
            records[node.name] = dict(**asdict(node), input_shape=list(x.shape),
                output_shape=list(y.shape), oracle=path.relative_to(out).as_posix(),
                oracle_sha256=sha(path), call_index=len(records), dtype='float32',
                layout='NCHW', preserves_full_view_batch=True)
        return record

    try:
        for node in plan:
            handles.append(node_module(module, node).register_forward_hook(hook(node)))
        started = time.perf_counter()
        with torch.inference_mode():
            result = checked_array(module(torch.from_numpy(value)), 'final output')
        reference = numerical_comparison(result, expected, rtol=1e-5, atol=1e-6)
        if len(records) != len(plan):
            raise ValueError('Not all requested U-Net boundaries were visited')
        if not reference['passed']:
            raise ValueError('Input fixture does not match original complete network')
        return dict(complete=True, label=label, seconds=time.perf_counter()-started,
                    reference=reference, boundaries=records)
    finally:
        for handle in handles:
            handle.remove()


def explicit_output(module, channels, layout, identity_conv_tail):
    """Real optional identity Conv restores a compiler sink, never a layout tag."""
    import torch
    if layout not in ('NCHW', 'NHWC') or channels < 1:
        raise ValueError('Invalid explicit output contract')

    class Sink(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.original = module
            self.tail = None
            if identity_conv_tail:
                self.tail = torch.nn.Conv2d(channels, channels, 1, bias=True)
                with torch.no_grad():
                    self.tail.weight.zero_()
                    self.tail.weight[:, :, 0, 0].copy_(torch.eye(channels))
                    self.tail.bias.zero_()
                self.tail.requires_grad_(False)

        def forward(self, value):
            result = self.original(value)
            if self.tail is not None:
                result = self.tail(result)
            return result.permute(0, 2, 3, 1).contiguous() if layout == 'NHWC' else result

    return Sink().eval()


def checkpoint_scope(module):
    if any(type(child).__name__ == 'AttentionBlock'
           and callable(getattr(child, '_forward', None)) for child in module.modules()):
        return bypass_attention_checkpoint(module)
    return nullcontext([])


def export_candidate(module, full_module, samples, folder, *, output_layout='NHWC',
                     identity_conv_tail=False):
    """Prove PyTorch and ORT A/B against captured original boundaries."""
    import onnx
    import onnxruntime as ort
    import torch
    folder = Path(folder)
    x, expected = samples['A']
    channels = expected.shape[1]
    wrapped = explicit_output(module, channels, output_layout, identity_conv_tail)
    record = dict(complete=False, diagnostic_only=True, production_qualified=False,
        npu_executed=False, input_shape=list(x.shape), output_shape=list(expected.shape),
        dtype='float32', logical_layout='NCHW', onnx_output_layout=output_layout,
        returns_list=False, original_parameters=sum(p.numel() for p in module.parameters()),
        identity_conv_tail=identity_conv_tail,
        identity_conv_multiply_count=int(np.prod(expected.shape)*channels) if identity_conv_tail else 0,
        full_view_batch_preserved=True, source_weight_or_normalization_changed=False,
        torch_checks={}, host_onnx_checks={})
    save(folder/'export_result.json', record)
    graph = folder/'model.onnx'
    with checkpoint_scope(full_module) as patched:
        record['temporary_checkpoint_bypass'] = patched
        for label, (value, reference) in samples.items():
            if value.shape != x.shape or reference.shape != expected.shape:
                raise ValueError('Static graph requires matching A/B shapes')
            with torch.inference_mode():
                wire = wrapped(torch.from_numpy(value)).detach().numpy()
            actual = wire.transpose(0, 3, 1, 2) if output_layout == 'NHWC' else wire
            check = numerical_comparison(actual, reference, rtol=1e-5, atol=1e-6)
            record['torch_checks'][label] = check
            save(folder/'export_result.json', record)
            if not check['passed']:
                raise ValueError('Original boundary/prefix PyTorch mismatch: '+label)
        started = time.perf_counter()
        with torch.inference_mode():
            torch.onnx.export(wrapped, (torch.from_numpy(x),), str(graph),
                input_names=['features'], output_names=['result'], opset_version=17, dynamo=False)
        record['export_seconds'] = time.perf_counter()-started
    record['checkpoint_scope_restored'] = True
    del wrapped
    gc.collect()
    onnx.checker.check_model(str(graph))
    inspected = onnx.load(str(graph), load_external_data=False)
    record['ops'] = dict(Counter(node.op_type for node in inspected.graph.node))
    del inspected
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = None
    try:
        session = ort.InferenceSession(str(graph), sess_options=options,
                                      providers=['CPUExecutionProvider'])
        for label, (value, reference) in samples.items():
            started = time.perf_counter()
            wire = session.run(None, {'features':value})[0]
            actual = wire.transpose(0, 3, 1, 2) if output_layout == 'NHWC' else wire
            check = numerical_comparison(actual, reference, rtol=2e-3, atol=2e-4)
            check['seconds'] = time.perf_counter()-started
            record['host_onnx_checks'][label] = check
            record['onnx_output_shape'] = list(wire.shape)
            path = folder/('oracle.npz' if label == 'A' else 'oracle_b.npz')
            np.savez(path, input=value, expected=reference)
            record['oracle_sha256' if label == 'A' else 'oracle_b_sha256'] = sha(path)
            save(folder/'export_result.json', record)
    finally:
        del session
        gc.collect()
    if 'B' in samples:
        difference = numerical_comparison(samples['A'][1], samples['B'][1],
                                          rtol=2e-3, atol=2e-4)
        record['input_switch_reference'] = dict(
            input_exactly_equal=bool(np.array_equal(samples['A'][0], samples['B'][0])),
            expected_distinguishable_at_tolerance=not difference['passed'],
            expected_difference=difference)
    passed = all(check['passed'] for check in record['host_onnx_checks'].values())
    record.update(complete=passed, host_onnx_passed=passed, graph_sha256=sha(graph),
                  parameters=record['original_parameters'],
                  status='host_verified_not_npu_executed' if passed else 'host_onnx_failed')
    save(folder/'export_result.json', record)
    return record


def load_captured_samples(capture_root, module_name, node, kind):
    """Hash-check persisted capture; prefixes use original network input."""
    root = Path(capture_root)
    manifest = json.loads((root/'capture_manifest.json').read_text(encoding='utf-8'))
    result = {}
    for label, trace in manifest['modules'][module_name]['captures'].items():
        if not trace['complete']:
            raise ValueError('Incomplete original capture: '+label)
        captured = trace['boundaries'][node.name]
        if captured['path'] != node.path:
            raise ValueError('Capture path disagrees with actual module topology')
        path = root/module_name/captured['oracle']
        if sha(path) != captured['oracle_sha256']:
            raise ValueError('Captured boundary oracle hash mismatch')
        value, expected = read_fixture(path)
        if kind == 'prefix':
            source = root/module_name/trace['source_fixture']
            if sha(source) != trace['source_fixture_sha256']:
                raise ValueError('Captured source fixture hash mismatch')
            value, _ = read_fixture(source)
        result[label] = value, expected
    if 'A' not in result:
        raise ValueError('Missing original A capture')
    return manifest, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', required=True, type=Path, help='Real context provenance; no full forward')
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--vendor', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--modules', nargs='+', choices=MODULES, default=list(MODULES))
    parser.add_argument('--phase', choices=('capture', 'export', 'both', 'list'), default='both')
    parser.add_argument('--fixtures', type=Path, help='Original stage input/expected fixtures')
    source_b = parser.add_mutually_exclusive_group()
    source_b.add_argument('--fixtures-b', type=Path, help='Original second real stage fixtures')
    source_b.add_argument('--zero-b', action='store_true', help='Explicit synthetic zero-stage-input B')
    parser.add_argument('--capture-root', type=Path, help='Previously completed capture root for export phase')
    parser.add_argument('--nodes', nargs='+', help='Boundary names/globs; default capture/export all')
    parser.add_argument('--layers', action='store_true',
                        help='Also expose unary rank-four Conv/GroupNorm/activation inside blocks; block-only')
    parser.add_argument('--mode', choices=('block', 'prefix', 'both'), default='block')
    parser.add_argument('--output-layout', choices=('NHWC', 'NCHW'), default='NHWC')
    parser.add_argument('--identity-conv-tail', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    if len(args.modules) != len(set(args.modules)):
        parser.error('Each module may be selected once')
    if args.phase in ('capture', 'both') and not args.fixtures:
        parser.error('Capture requires --fixtures from a real original full-stage reference')
    if args.phase == 'export' and not args.capture_root:
        parser.error('Export requires --capture-root')
    out = new_directory(args.out)
    from gaussian_generation.runtime import ModelRuntime
    runtime = ModelRuntime(args.weights, args.vendor, args.threads, 16, out/'runtime')
    with np.load(args.context, allow_pickle=False) as data:
        context_shape = [1, *data['image'].shape]
    base = dict(weight_sha256=runtime.weight_hash, context_sha256=sha(args.context),
        source_commit=SOURCE_COMMIT, source_hashes=runtime.source_hashes,
        exporter_sha256=sha(Path(__file__)), input_shape=context_shape,
        diagnostic_only=True, npu_executed=False, production_qualified=False,
        protection='Run under measure.py --pss-mib 700 --reserve-mib 96; no guard relaxation')
    capture_manifest = dict(**base, schema='mvsplat_unet_boundaries_v1', modules={})
    manifest = dict(**base, schema='mvsplat_partitions_v1', partitions={})
    for name in args.modules:
        module = getattr(runtime.model.depth_predictor, name).eval()
        _, plan = boundary_plan(module, include_layers=args.layers)
        selected = select_boundaries(plan, args.nodes)
        if args.phase == 'list':
            print(json.dumps(dict(module=name, boundaries=[asdict(node) for node in plan])), flush=True)
            continue
        capture_root = out if args.phase != 'export' else args.capture_root
        if args.phase in ('capture', 'both'):
            folder = new_directory(out/name)
            record = dict(boundaries=[asdict(node) for node in selected], captures={})
            capture_manifest['modules'][name] = record
            source = fixture_path(args.fixtures, name)
            value, expected = read_fixture(source)
            samples = {'A':(value, expected, str(source.resolve()))}
            if args.fixtures_b:
                source_b_path = fixture_path(args.fixtures_b, name)
                samples['B'] = (*read_fixture(source_b_path), str(source_b_path.resolve()))
            elif args.zero_b:
                zero = np.zeros_like(value)
                with runtime.torch.inference_mode():
                    zero_expected = checked_array(module(runtime.torch.from_numpy(zero)), 'zero B').copy()
                samples['B'] = (zero, zero_expected, 'synthetic zero original-stage input; not a real second scene')
            for label, (value, expected, origin) in samples.items():
                source_path = folder/('source_'+label.lower()+'.npz')
                np.savez(source_path, input=value, expected=expected)
                try:
                    trace = capture_boundaries(module, value, expected, folder, label=label, plan=selected)
                    trace.update(source_fixture=source_path.name, source_fixture_sha256=sha(source_path),
                                 source_origin=origin)
                    record['captures'][label] = trace
                except Exception as exc:
                    record['captures'][label] = dict(complete=False, error=type(exc).__name__+': '+str(exc))
                save(out/'capture_manifest.json', capture_manifest)
                print(json.dumps(dict(module=name, capture=label,
                    complete=record['captures'][label]['complete'])), flush=True)
            del samples, value, expected
            gc.collect()
        if args.phase == 'capture':
            continue
        kinds = ('block', 'prefix') if args.mode == 'both' else (args.mode,)
        for node in selected:
            for kind in kinds:
                if kind == 'prefix' and not node.prefix:
                    continue
                key = name+'__'+node.name+'__'+kind
                folder = new_directory(out/key)
                item = dict(complete=False, module=name, boundary=asdict(node), kind=kind,
                            diagnostic_only=True, npu_executed=False, production_qualified=False)
                manifest['partitions'][key] = item
                try:
                    captured_manifest, samples = load_captured_samples(capture_root, name, node, kind)
                    if (captured_manifest['weight_sha256'] != runtime.weight_hash
                            or captured_manifest['source_hashes'] != runtime.source_hashes
                            or captured_manifest['context_sha256'] != sha(args.context)):
                        raise ValueError('Capture weight/context/execution source identity mismatch')
                    item['capture_manifest_sha256'] = sha(Path(capture_root)/'capture_manifest.json')
                    candidate = node_module(module, node) if kind == 'block' else prefix_module(module, node.name)
                    item.update(export_candidate(candidate, module, samples, folder,
                        output_layout=args.output_layout, identity_conv_tail=args.identity_conv_tail))
                    del candidate, samples
                except Exception as exc:
                    partial = folder/'export_result.json'
                    if partial.is_file():
                        item.update(json.loads(partial.read_text(encoding='utf-8')))
                    item.update(complete=False, status='export_failed', error=type(exc).__name__+': '+str(exc))
                finally:
                    gc.collect()
                    save(folder/'export_result.json', item)
                    save(out/'manifest.json', manifest)
                print(json.dumps(dict(partition=key, status=item['status'], error=item.get('error'))), flush=True)
    if args.phase in ('capture', 'both') and any(not trace['complete']
            for item in capture_manifest['modules'].values() for trace in item['captures'].values()):
        raise RuntimeError('One or more original boundary captures failed; inspect capture_manifest.json')
    if args.phase in ('export', 'both') and not any(item['complete'] for item in manifest['partitions'].values()):
        raise RuntimeError('No boundary candidate passed original PyTorch and ONNX checks')


if __name__ == '__main__':
    main()
