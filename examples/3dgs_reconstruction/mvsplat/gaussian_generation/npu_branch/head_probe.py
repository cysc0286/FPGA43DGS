"""Checkpoint-backed Gaussian-head candidates and single-session NPU diagnosis.

This does not replace a production partition or issue an acceptance receipt.
The batch-one and aligned graphs are candidates for the CPU's largest remaining
convolution head; prefixes/suffixes isolate the first numerical deviation.
"""
import argparse
import copy
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from common import new_directory, save, sha


def compare_output(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError('Output shape differs from oracle')
    finite = np.isfinite(actual)
    if not np.isfinite(expected).all():
        raise ValueError('Nonfinite reference output')
    count = int(actual.size - np.count_nonzero(finite))
    result = dict(passed=False, nonfinite_count=count,
                  output_sha256=hashlib.sha256(actual.tobytes()).hexdigest())
    if not count:
        from gaussian_generation.npu_branch.verify import errors
        result.update(errors(actual, expected, 2e-3, 2e-4))
    return result


def _head(model):
    from onnx import numpy_helper
    nodes = list(model.graph.node)
    conv = [i for i, n in enumerate(nodes) if n.op_type == 'Conv']
    if (len(conv) != 2 or conv[0] != 0 or conv[1] != len(nodes)-1 or
            len(model.graph.input) != 1 or len(model.graph.output) != 1):
        raise ValueError('Expected exactly Conv / pointwise activation / Conv')
    # Only channel-independent, zero-preserving activation paths can safely
    # gain zero channels. In particular, never pad a Softmax/normalization.
    allowed = {'Constant', 'Div', 'Erf', 'Add', 'Mul', 'Tanh', 'Relu', 'Identity'}
    if any(n.op_type not in allowed for n in nodes[1:-1]):
        raise ValueError('Only a pointwise activation is supported')
    weights = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}
    first, last = nodes[0], nodes[-1]
    for node in (first, last):
        if len(node.input) != 3 or any(a.name == 'group' and a.i != 1 for a in node.attribute):
            raise ValueError('Expected biased, ungrouped convolutions')
    w0, w1 = weights[first.input[1]], weights[last.input[1]]
    if w0.ndim != 4 or w1.ndim != 4 or w0.shape[0] != w1.shape[1]:
        raise ValueError('Convolution channel contract differs')
    return nodes, weights, w0.shape[1], w0.shape[0], w1.shape[0]


def aligned_model(model, multiple=16):
    """Explicit zero-padding, keeping every original parameter and channel.

    This changes physical channel counts, not learned weights. CPU equivalence
    is required after rewriting. Hardware precision/traffic still need testing.
    """
    from onnx import numpy_helper
    if type(multiple) is not int or multiple < 1:
        raise ValueError('Channel multiple must be positive')
    model = copy.deepcopy(model)
    nodes, values, inputs, hidden, outputs = _head(model)
    sizes = [((n+multiple-1)//multiple)*multiple for n in (inputs, hidden, outputs)]
    ni, nh, no = sizes
    for node, new_shape in ((nodes[0], (nh, ni)), (nodes[-1], (no, nh))):
        name = node.input[1]
        old = values[name]
        padded = np.zeros((*new_shape, *old.shape[2:]), dtype=old.dtype)
        padded[:old.shape[0], :old.shape[1]] = old
        values[name] = padded
        bias = values[node.input[2]]
        values[node.input[2]] = np.pad(bias, (0, new_shape[0]-len(bias)))
    for initializer in model.graph.initializer:
        initializer.CopyFrom(numpy_helper.from_array(values[initializer.name], initializer.name))
    model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = ni
    model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = no
    del model.graph.value_info[:]
    return model


def make_cases(model):
    """Full control, two prefixes, final convolution, batch-one, aligned head."""
    import onnx
    nodes, _, _, _, _ = _head(model)
    inferred = onnx.shape_inference.infer_shapes(model)
    types = {v.name: v for v in (*inferred.graph.input, *inferred.graph.value_info,
                                *inferred.graph.output)}
    cases = {'full_b2': copy.deepcopy(model)}
    for name, selected, input_name, output_name in (
        ('conv_in_b2', nodes[:1], nodes[0].input[0], nodes[0].output[0]),
        ('conv_gelu_b2', nodes[:-1], nodes[0].input[0], nodes[-1].input[0]),
        ('conv_out_b2', nodes[-1:], nodes[-1].input[0], nodes[-1].output[0])):
        candidate = copy.deepcopy(model)
        del candidate.graph.node[:]
        candidate.graph.node.extend(copy.deepcopy(selected))
        del candidate.graph.input[:]
        candidate.graph.input.append(copy.deepcopy(types[input_name]))
        del candidate.graph.output[:]
        candidate.graph.output.append(copy.deepcopy(types[output_name]))
        # Standardize the boundary names used by the existing ICraft CLI.
        for node in candidate.graph.node:
            for i, value in enumerate(node.input):
                if value == input_name:
                    node.input[i] = 'features'
            for i, value in enumerate(node.output):
                if value == output_name:
                    node.output[i] = 'result'
        candidate.graph.input[0].name = 'features'
        candidate.graph.output[0].name = 'result'
        used = {v for n in candidate.graph.node for v in n.input}
        parameters = [v for v in candidate.graph.initializer if v.name in used]
        del candidate.graph.initializer[:]
        candidate.graph.initializer.extend(parameters)
        del candidate.graph.value_info[:]
        cases[name] = candidate
    one = copy.deepcopy(model)
    for v in (*one.graph.input, *one.graph.output):
        v.type.tensor_type.shape.dim[0].dim_value = 1
    del one.graph.value_info[:]
    cases['full_b1'] = one
    cases['aligned_b1'] = aligned_model(one)
    for candidate in cases.values():
        onnx.checker.check_model(candidate)
    return cases


def _session(model):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    return ort.InferenceSession(model.SerializeToString(), sess_options=options,
                               providers=['CPUExecutionProvider'])


def export(graphs, out):
    import onnx
    from gaussian_generation.npu_branch.artifacts import validate_bundle
    metadata, parts = validate_bundle(graphs, ['to_gaussians'], 'onnx_reference')
    source = Path(parts['to_gaussians']['graph'])
    oracle = Path(graphs)/'to_gaussians/oracle.npz'
    if sha(oracle) != metadata['partitions']['to_gaussians']['oracle_sha256']:
        raise ValueError('Source oracle checksum changed')
    with np.load(oracle, allow_pickle=False) as data:
        value, expected = data['input'], data['expected']
    if value.ndim != 4 or value.shape[0] != 2 or not np.isfinite(value).all():
        raise ValueError('Expected two finite, real context-view inputs')
    model = onnx.load(str(source))
    original = _session(model).run(None, {'features': value})[0]
    source_check = compare_output(original, expected)
    if not source_check['passed']:
        raise ValueError('Original ONNX no longer matches the saved CPU oracle')
    cases = make_cases(model)
    hidden = _session(cases['conv_gelu_b2']).run(None, {'features': value})[0]
    out = new_directory(out)
    manifest = dict(schema='mvsplat_head_probe_v1', diagnostic_only=True, npu_executed=False,
        weight_sha256=metadata['weight_sha256'], input_shape=metadata['input_shape'],
        source_manifest_sha256=sha(Path(graphs)/'manifest.json'),
        source_graph_sha256=sha(source), source_oracle_sha256=sha(oracle),
        original_onnx_vs_saved_cpu=source_check, partitions={})
    for name, candidate in cases.items():
        folder = new_directory(out/name)
        x = hidden if name == 'conv_out_b2' else value
        if name.endswith('_b1'):
            a, b = x[:1], x[1:2]
        else:
            a, b = x, np.zeros_like(x)
        if name == 'aligned_b1':
            width = candidate.graph.input[0].type.tensor_type.shape.dim[1].dim_value
            a, b = [np.pad(v, ((0,0), (0,width-v.shape[1]), (0,0), (0,0))) for v in (a,b)]
        session = _session(candidate)
        ya, yb = [session.run(None, {'features': v})[0] for v in (a,b)]
        # The full and suffix graphs compare to the original saved CPU result;
        # prefix oracles come from the unmodified ONNX path, not invented data.
        check = dict(passed=True, scope='unmodified ONNX prefix')
        if name.startswith(('full_', 'aligned_')) or name == 'conv_out_b2':
            ref = expected[:1] if name.endswith('_b1') else expected
            check = compare_output(ya[:, :ref.shape[1]], ref)
            if name.endswith('_b1'):
                alternate = compare_output(yb[:, :ref.shape[1]], expected[1:2])
                if not alternate['passed']:
                    raise ValueError('Second real view changed after graph rewrite: '+name)
                check['second_view'] = alternate
            if not check['passed'] or not np.isfinite(yb).all():
                raise ValueError('Head transformation changed the real oracle: '+name)
            if name == 'aligned_b1' and (np.any(ya[:, ref.shape[1]:]) or np.any(yb[:, ref.shape[1]:])):
                raise ValueError('Padded output channels must remain exactly zero')
        if not np.isfinite(ya).all() or not np.isfinite(yb).all():
            raise ValueError('Nonfinite host oracle: '+name)
        onnx.save(candidate, str(folder/'model.onnx'))
        np.savez(folder/'oracle.npz', input=a, expected=ya, alternate_input=b, alternate_expected=yb)
        manifest['partitions'][name] = dict(complete=True, diagnostic_only=True,
            input_shape=list(a.shape), output_shape=list(ya.shape), returns_list=False,
            graph_sha256=sha(folder/'model.onnx'), oracle_sha256=sha(folder/'oracle.npz'),
            host_onnx_passed=True, cpu_reference_check=check,
            logical_io_bytes=a.nbytes+ya.nbytes,
            alternate_kind='second real view' if name.endswith('_b1') else 'zero-input freshness control',
            npu_executed=False)
        save(out/'manifest.json', manifest)
        print(name, 'host oracle passed', a.shape, ya.shape, flush=True)
    return manifest


def run(graphs, bundle, library, name, out, passes=1, save_outputs=False):
    """Exactly one graph/session per process; changed inputs must be current.

    Input write, submit, wait, conversion and entire caller duration are kept.
    A failed value or freshness check is never an accepted performance result.
    """
    if platform.machine().lower() not in ('aarch64', 'arm64'):
        raise RuntimeError('Actual ARM NPU required; host replay is not hardware execution')
    import fcntl
    from gaussian_generation.npu_branch.artifacts import relative_file, to_wire, from_wire
    from gaussian_generation.npu_branch.compile_graphs import audit
    from gaussian_generation.npu_branch.protocol import check_bridge, check_direct_sfb
    if passes not in (1, 2):
        raise ValueError('Choose one SDK pass or an explicitly timed two-pass control')
    metadata = json.loads((graphs/'manifest.json').read_text())
    compiled = json.loads((bundle/'manifest.json').read_text())
    if (metadata.get('schema') != 'mvsplat_head_probe_v1' or
            not metadata.get('diagnostic_only') or
            compiled['source_manifest_sha256'] != sha(graphs/'manifest.json')):
        raise ValueError('Diagnostic source/compiled provenance differs')
    item = compiled['partitions'][name]
    if not item['complete'] or item['export'] != metadata['partitions'][name]:
        raise ValueError('Compiled diagnostic metadata differs')
    graph, raw = [relative_file(bundle, item[k]) for k in ('graph', 'raw')]
    placement = audit(graph, raw, item['export'])
    if placement != item['audit']:
        raise ValueError('Compiled ABI or placement changed')
    oracle = relative_file(graphs, name+'/oracle.npz')
    if sha(oracle) != item['export']['oracle_sha256']:
        raise ValueError('Diagnostic oracle changed')
    out = new_directory(out)
    result = dict(schema='mvsplat_head_probe_result_v1', diagnostic_only=True,
        passed=False, npu_executed=False, case=name, sdk_execution_passes=passes,
        library_sha256=sha(library), source_manifest_sha256=sha(graphs/'manifest.json'),
        compiled_manifest_sha256=sha(bundle/'manifest.json'), checks=[],
        scope='one real NPU session; not full model, FPGA quality or end-to-end speed')
    lock = open('/run/lock/fpga43dgs-npu.lock', 'a')
    ctx, lib = None, None
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Preserve the existing renderer/NPU device-address separation.
        os.environ.setdefault('MGS_NPU_DDR_GUARD_MIB', '64')
        lib = ct.CDLL(str(library.resolve()))
        check_bridge(lib)
        if os.environ.get('MGS_NPU_DIRECT_SFB') == '1':
            check_direct_sfb(lib)
        lib.mgs_error.restype = ct.c_char_p
        lib.mgs_create.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_size_t, ct.c_size_t]
        lib.mgs_create.restype = ct.c_void_p
        lib.mgs_destroy.argtypes = [ct.c_void_p]
        lib.mgs_bindings.argtypes = [ct.c_void_p]
        forward = lib.mgs_forward_detailed
        forward.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_void_p,
                            ct.c_size_t, ct.POINTER(ct.c_double)]
        forward.restype = ct.c_int
        lib.mgs_memory_layout.argtypes = [ct.c_void_p]
        lib.mgs_memory_layout.restype = ct.c_char_p
        with np.load(oracle, allow_pickle=False) as samples:
            x = np.empty(placement['input']['shape'], np.float32)
            y = np.empty(placement['output']['shape'], np.float32)
            ctx = lib.mgs_create(str(graph).encode(), str(raw).encode(), x.nbytes, y.nbytes)
            if not ctx:
                raise RuntimeError(lib.mgs_error().decode())
            if lib.mgs_bindings(ctx) < 1:
                raise ValueError('No real NPU compute binding')
            layout = json.loads(lib.mgs_memory_layout(ctx) or 'null')
            guard = int(os.environ['MGS_NPU_DDR_GUARD_MIB'])*1024**2
            if guard < 64*1024**2 or not layout or min(layout, key=lambda c:c['begin'])['bytes'] != guard:
                raise ValueError('Unverified renderer/NPU device memory separation')
            result['memory_layout'] = layout
            try:
                precheck = lib.mgs_precheck
            except AttributeError:
                result['sdk_precheck'] = dict(available=False, passed=False,
                                             reason='Rebuild the diagnostic bridge to expose SDK precheck')
            else:
                precheck.argtypes = [ct.c_void_p]
                precheck.restype = ct.c_int
                status = precheck(ctx)
                result['sdk_precheck'] = dict(available=True, passed=status == 1, status=status,
                    error=lib.mgs_error().decode() if status < 0 else None)
            # Continue only inside this diagnostic to localize the first bad
            # layer. A failed/unavailable SDK precheck cannot produce a pass.
            for step, label in enumerate(('A', 'A', 'B', 'B', 'A', 'A')):
                value, expected = (samples['input'], samples['expected']) if label == 'A' else (
                    samples['alternate_input'], samples['alternate_expected'])
                started = time.perf_counter()
                to_wire(value, placement['input'], out=x, blocked=x.nbytes >= 4*1024**2)
                packed = time.perf_counter()
                stages = np.zeros(5, np.float64)
                for _ in range(passes):
                    times = (ct.c_double*5)()
                    if forward(ctx, x.ctypes.data, x.nbytes, y.ctypes.data, y.nbytes, times):
                        raise RuntimeError(lib.mgs_error().decode())
                    result['npu_executed'] = True
                    stages += np.asarray(times)
                received = time.perf_counter()
                # Preserve nonfinite output evidence; never send it downstream.
                actual = (from_wire(y, placement['output']) if np.isfinite(y).all() else
                          np.array(y.transpose(0,3,1,2) if placement['output']['layout'] == 'NHWC' else y, copy=True))
                finished = time.perf_counter()
                check = compare_output(actual, expected)
                check.update(step=step, input_label=label, pack_seconds=packed-started,
                    sdk_call_seconds=received-packed, unpack_seconds=finished-received,
                    total_seconds=finished-started, sdk_seconds=stages.tolist(),
                    sdk_stage_names=['input_write','forward_submit','wait','output_convert','sdk_total'],
                    logical_io_bytes=x.nbytes+y.nbytes)
                result['checks'].append(check)
                if save_outputs:
                    np.save(out/f'{step}_{label}.npy', actual, allow_pickle=False)
                save(out/'result.json', result)
                print(json.dumps(check), flush=True)
            hashes = [c['output_sha256'] for c in result['checks']]
            result['freshness_passed'] = bool(hashes[0] == hashes[1] == hashes[4] == hashes[5] and
                                               hashes[2] == hashes[3] and hashes[1] != hashes[2])
            result['passed'] = (result['sdk_precheck']['passed'] and result['freshness_passed'] and
                                all(c['passed'] for c in result['checks']))
    except BaseException as exc:
        result['error'] = type(exc).__name__+': '+str(exc)
        raise
    finally:
        if ctx:
            lib.mgs_destroy(ctx)
        lock.close()
        save(out/'result.json', result)
    if not result['passed']:
        raise RuntimeError('Head numerical/freshness gate failed; not a speed result')
    return result


def compare_icraft(graphs, bundle, out, runner):
    """Compare ICraft Host layer dumps before/after quantization, never speed.

    This separates compiler-host arithmetic from the real device failure. Only
    the original batch-two control is used so its three saved layer oracles
    match one-for-one. A Host success cannot qualify any hardware candidate.
    """
    from gaussian_generation.npu_branch.artifacts import relative_file
    metadata = json.loads((graphs/'manifest.json').read_text())
    compiled = json.loads((bundle/'manifest.json').read_text())
    if (metadata.get('schema') != 'mvsplat_head_probe_v1' or
            compiled['source_manifest_sha256'] != sha(graphs/'manifest.json')):
        raise ValueError('Compiler-host diagnostic provenance differs')
    source = relative_file(graphs, 'full_b2/oracle.npz')
    if sha(source) != metadata['partitions']['full_b2']['oracle_sha256']:
        raise ValueError('Full-head oracle changed')
    out = new_directory(out)
    result = dict(scope='ICraft Host per-layer oracle, not NPU or model performance',
                  npu_executed=False, runner_sha256=sha(runner), stages={})
    references = {}
    for key, case in (('Conv2d_0', 'conv_in_b2'), ('GELU_0', 'conv_gelu_b2'),
                      ('Conv2d_1', 'full_b2')):
        oracle = relative_file(graphs, case+'/oracle.npz')
        if sha(oracle) != metadata['partitions'][case]['oracle_sha256']:
            raise ValueError('Layer oracle changed')
        with np.load(oracle, allow_pickle=False) as samples:
            references[key] = samples['expected']
    with np.load(source, allow_pickle=False) as samples:
        value = samples['input']
    value.transpose(0,2,3,1).copy().tofile(out/'input.ftmp')
    for stage in ('optimized', 'quantized'):
        folder = new_directory(out/stage)
        network_path = bundle/'full_b2'/('full_b2_'+stage+'.json')
        raw = network_path.with_suffix('.raw')
        network = json.loads(network_path.read_text())
        command = [str(runner.resolve()), '--json', str(network_path.resolve()),
                   '--raw', str(raw.resolve()), '--input', str((out/'input.ftmp').resolve()),
                   '--backends', 'Host', '--dump_format', 'SFB', '--log_path', str(folder/'logs')]
        record = result['stages'][stage] = dict(command=command, layers={}, complete=False,
            graph_sha256=sha(network_path), raw_sha256=sha(raw))
        try:
            with (folder/'run.log').open('wb') as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120)
            record['exit_code'] = process.returncode
            if process.returncode:
                raise RuntimeError('ICraft Host stage failed; inspect run.log')
            counts = {}
            dumps = folder/'logs'/network['name']/'ftmpSFB'
            for op in network['ops']:
                kind = op['_type_key'].rsplit('::',1)[-1]
                index = counts.get(kind, 0)
                counts[kind] = index+1
                key = kind+'_'+str(index)
                if kind != 'Input' and key not in references:
                    continue
                for tensor in op.get('outputs', []):
                    dtype = tensor['dtype']
                    if dtype['layout'] != '@layout(NHWC)':
                        raise ValueError('Unreviewed ICraft Host dump layout')
                    actual = np.fromfile(dumps/(str(tensor['v_id'])+'.ftmp'), np.float32)
                    if actual.size != int(np.prod(dtype['shape'])):
                        raise ValueError('ICraft SFB layer length differs')
                    actual = actual.reshape(dtype['shape']).transpose(0,3,1,2).copy()
                    record['layers'][key] = compare_output(actual, value if kind == 'Input' else references[key])
            if len(record['layers']) != 4:
                raise ValueError('Missing ICraft input or head layer dump')
            record['complete'] = True
            record['first_strict_failure'] = next((key for key, check in record['layers'].items()
                                                   if not check['passed']), None)
        except Exception as exc:
            record['error'] = type(exc).__name__+': '+str(exc)
        save(out/'result.json', result)
        print(stage, json.dumps(record), flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('export', 'run', 'compare-icraft'))
    p.add_argument('--graphs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--bundle', type=Path)
    p.add_argument('--library', type=Path)
    p.add_argument('--runner', type=Path)
    p.add_argument('--case')
    p.add_argument('--sdk-passes', type=int, choices=(1,2), default=1)
    p.add_argument('--save-outputs', action='store_true')
    a = p.parse_args(argv)
    if a.action == 'export':
        export(a.graphs, a.out)
    elif a.action == 'compare-icraft':
        if not a.bundle or not a.runner:
            p.error('compare-icraft requires bundle and runner')
        record = compare_icraft(a.graphs, a.bundle, a.out, a.runner)
        if not all(v['complete'] for v in record['stages'].values()):
            raise SystemExit(1)
    else:
        if not all((a.bundle, a.library, a.case)):
            p.error('run requires bundle, library and case')
        run(a.graphs, a.bundle, a.library, a.case, a.out, a.sdk_passes, a.save_outputs)


if __name__ == '__main__':
    main()
