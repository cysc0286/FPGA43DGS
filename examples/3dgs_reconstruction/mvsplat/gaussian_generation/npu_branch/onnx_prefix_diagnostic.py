"""Diagnose unregistered source-ONNX-derived prefixes; never issue a receipt.

Unlike diagnostic_runner's ARM intermediate hook contract, these expected
intermediates are derived from the verified original full ONNX graph. Original
ARM full-stage references bind the real A/B inputs and remain separate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from common import save, sha
from .artifacts import from_wire, relative_file, to_wire
from .catalog import PARTITIONS
from .compile_graphs import audit
from .diagnostic_runner import BridgeSession, array_sha, compare


SCHEMA = 'mvsplat_source_onnx_prefix_diagnostic_v1'
ORACLE = 'source_onnx_derived'
SEQUENCES = {'aabbaa': ('A', 'A', 'B', 'B', 'A', 'A'),
             'abba': ('A', 'B', 'B', 'A')}
MANIFEST_KEYS = ('export', 'compiled', 'full_source', 'arm_fixtures')


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _logical_shape(value):
    if (not isinstance(value, list) or len(value) != 4
            or any(type(x) is not int or x <= 0 for x in value) or value[0] != 2):
        raise ValueError('Prefix must preserve a static rank-four two-view batch')
    return value


def _check_root_input_shapes(source, compiled, original):
    shape = original.get('input_shape')
    if (not isinstance(shape, list) or len(shape) != 5
            or any(type(x) is not int or x <= 0 for x in shape)
            or shape[:3] != [1, 2, 3]
            or source.get('input_shape') != shape or compiled.get('input_shape') != shape):
        raise ValueError('Original/export/compiled root input shapes must match one two-view RGB context')
    return shape


def _check_capture_shapes(stage, original_meta, prefix_input_shape):
    if (stage.get('input_shape') != prefix_input_shape
            or stage.get('input_shape') != original_meta.get('input_shape')
            or stage.get('output_shape') != original_meta.get('output_shape')):
        raise ValueError('Capture input/output shape metadata differs from original full stage')
    _logical_shape(stage['input_shape'])
    _logical_shape(stage['output_shape'])


def _check_lineage(full_path, candidate_path, meta):
    """Verify real nodes/weights/tail operations, never a compiled layout tag."""
    import onnx
    from onnx import numpy_helper

    full = onnx.load(str(full_path), load_external_data=False)
    candidate = onnx.load(str(candidate_path), load_external_data=False)
    allowed = {'first_conv': ('Conv', '/inner/inner.0/Conv'),
               'first_groupnorm_affine': ('Add', '/inner/inner.1/Add'),
               'first_groupnorm_gelu': ('Mul', '/inner/inner.2/Mul_1')}
    expected_endpoint = allowed.get(meta.get('prefix_label'))
    if expected_endpoint != (meta.get('endpoint_op'), meta.get('endpoint_node')):
        raise ValueError('Unknown explicit source ONNX prefix endpoint')
    matches = [i for i, node in enumerate(full.graph.node)
               if (node.op_type, node.name) == expected_endpoint]
    if len(matches) != 1:
        raise ValueError('Source endpoint is absent or ambiguous')
    original_nodes = list(full.graph.node[:matches[0] + 1])
    actual_nodes = list(candidate.graph.node)
    if [n.SerializeToString() for n in actual_nodes[:len(original_nodes)]] != [
            n.SerializeToString() for n in original_nodes]:
        raise ValueError('Candidate does not preserve the exact original prefix nodes')
    endpoint = original_nodes[-1].output[0]
    if endpoint != meta.get('endpoint_tensor'):
        raise ValueError('Source endpoint tensor differs from export metadata')
    if list(full.opset_import) != list(candidate.opset_import):
        raise ValueError('Candidate changed original ONNX opsets')
    needed = {x for node in original_nodes for x in node.input}
    original_initializers = {x.name: x for x in full.graph.initializer if x.name in needed}
    actual_initializers = {x.name: x for x in candidate.graph.initializer}
    if len(actual_initializers) != len(candidate.graph.initializer):
        raise ValueError('Duplicate initializer name')
    for name, value in original_initializers.items():
        if (name not in actual_initializers or actual_initializers[name].SerializeToString()
                != value.SerializeToString() or value.data_location == onnx.TensorProto.EXTERNAL):
            raise ValueError('Original initializer bytes changed or externalized: ' + name)
    extras = set(actual_initializers) - set(original_initializers)
    tail = actual_nodes[len(original_nodes):]
    identity = bool((meta.get('output_tail_control') or {}).get('identity_conv'))
    added_parameters = 0
    if identity:
        if len(tail) != 2 or tail[0].op_type != 'Conv':
            raise ValueError('Identity candidate lacks a genuine extra Conv node')
        conv = tail[0]
        names = ('__mgs_output_control_weight', '__mgs_output_control_bias')
        attrs = {x.name: onnx.helper.get_attribute_value(x) for x in conv.attribute}
        if (extras != set(names) or list(conv.input) != [endpoint, *names]
                or attrs != dict(kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0])):
            raise ValueError('Unexpected identity Conv topology/initializer names')
        channels = meta['output_shape'][1]
        weight = numpy_helper.to_array(actual_initializers[names[0]])
        bias = numpy_helper.to_array(actual_initializers[names[1]])
        if (weight.dtype != np.float32 or bias.dtype != np.float32
                or not np.array_equal(weight, np.eye(channels, dtype=np.float32).reshape(channels, channels, 1, 1))
                or not np.array_equal(bias, np.zeros(channels, dtype=np.float32))):
            raise ValueError('Extra Conv is not exactly the declared identity')
        added_parameters = channels * channels + channels
        endpoint = conv.output[0]
        if meta.get('injected_identity_parameters') != added_parameters:
            raise ValueError('Identity parameter cost is missing or incorrect')
        tail = tail[1:]
    elif extras:
        raise ValueError('Unexpected additional prefix initializers')
    if len(tail) != 1 or tail[0].op_type != 'Transpose':
        raise ValueError('Expected an explicit terminal NHWC transpose')
    trans = tail[0]
    attrs = {x.name: onnx.helper.get_attribute_value(x) for x in trans.attribute}
    if (attrs != {'perm': [0, 2, 3, 1]} or list(trans.input) != [endpoint]
            or len(candidate.graph.output) != 1
            or list(trans.output) != [candidate.graph.output[0].name]):
        raise ValueError('Candidate terminal transpose/axis contract changed')
    input_values = [x for x in candidate.graph.input if x.name not in actual_initializers]
    if len(input_values) != 1 or input_values[0].name != meta.get('input_name'):
        raise ValueError('Expected exactly the declared original public input')
    for value, expected in ((input_values[0], meta['input_shape']),
                             (candidate.graph.output[0], meta['onnx_output_shape'])):
        dtype = value.type.tensor_type
        if dtype.elem_type != onnx.TensorProto.FLOAT or [d.dim_value for d in dtype.shape.dim] != expected:
            raise ValueError('Actual ONNX input/output differs from declared static FP32 ABI')
    return dict(original_node_count=len(original_nodes),
                original_initializer_count=len(original_initializers),
                original_initializer_bytes_exact=True, identity_conv_tail=identity,
                injected_identity_parameters=added_parameters,
                injected_identity_multiplications=int(np.prod(meta['output_shape'])) * meta['output_shape'][1] if identity else 0)


def validate_candidate(graphs, bundle, partition, full_source, arm_fixtures, pins):
    """Validate distinct diagnostic identities without accepting formal names."""
    if (not isinstance(partition, str) or not partition or partition in ('.', '..')
            or any(x in partition for x in ('/', '\\', ':'))
            or any(ord(x) < 32 for x in partition) or Path(partition).name != partition):
        raise ValueError('Partition must be a single portable path component')
    graphs, bundle = Path(graphs).resolve(), Path(bundle).resolve()
    full_source, arm_fixtures = Path(full_source).resolve(), Path(arm_fixtures).resolve()
    paths = dict(export=graphs / 'manifest.json', compiled=bundle / 'manifest.json',
                 full_source=full_source / 'manifest.json', arm_fixtures=arm_fixtures / 'capture_manifest.json')
    if set(pins) != set(MANIFEST_KEYS) or any(sha(paths[k]) != pins[k] for k in MANIFEST_KEYS):
        raise ValueError('Pinned diagnostic/source/ARM fixture manifest identity changed')
    source, compiled, original, capture = (_json(paths[k]) for k in MANIFEST_KEYS)
    if partition in PARTITIONS:
        raise ValueError('A diagnostic prefix cannot use a registered production partition identity')
    root_input_shape = _check_root_input_shapes(source, compiled, original)
    if (source.get('schema') != 'mvsplat_partitions_v1' or not source.get('diagnostic_only')
            or source.get('production_qualified') is not False or source.get('npu_executed') is not False
            or source.get('source_manifest_sha256') != pins['full_source']
            or source.get('original_arm_fixture_manifest_sha256') != pins['arm_fixtures']):
        raise ValueError('Expected explicitly source-ONNX-derived diagnostic manifest lineage')
    meta = source.get('partitions', {}).get(partition)
    item = compiled.get('partitions', {}).get(partition)
    if (not meta or not item or not meta.get('complete') or not meta.get('host_onnx_passed')
            or not meta.get('diagnostic_only') or meta.get('production_qualified') is not False
            or meta.get('npu_executed') is not False or meta.get('kind') != 'onnx_prefix'
            or meta.get('module') not in ('corr_refine_net', 'refine_unet')
            or 'ONNX' not in meta.get('intermediate_oracle_provenance', '')
            or meta.get('logical_layout') != 'NCHW' or meta.get('onnx_output_layout') != 'NHWC'
            or meta.get('dtype') != 'float32' or not meta.get('full_view_batch_preserved')
            or compiled.get('schema') != 'mvsplat_compiled_v1'
            or compiled.get('source_manifest_sha256') != pins['export']
            or compiled.get('npu_executed') is not False or not item.get('complete')
            or item.get('export') != meta):
        raise ValueError('Prefix metadata, compiled identity or oracle provenance differs')
    module = meta['module']
    original_meta = original.get('partitions', {}).get(module) or {}
    weight = source.get('weight_sha256')
    if (not original_meta.get('complete') or not original_meta.get('full_module')
            or not original_meta.get('host_onnx_passed')
            or not weight or any(x.get('weight_sha256') != weight for x in (compiled, original, capture))
            or capture.get('schema') != 'mvsplat_real_context_stage_fixtures_v1'
            or not capture.get('complete') or capture.get('npu_executed') is not False
            or capture.get('reference_device') != 'board ARM CPU'):
        raise ValueError('Original full ONNX/checkpoint/ARM capture identity mismatch')
    input_shape, output_shape = _logical_shape(meta['input_shape']), _logical_shape(meta['output_shape'])
    if input_shape != original_meta['input_shape']:
        raise ValueError('Prefix input differs from original complete stage')
    full_graph = relative_file(full_source, Path(module) / 'model.onnx')
    model = relative_file(graphs, Path(partition) / 'model.onnx')
    if (sha(full_graph) != source.get('source_model_sha256')
            or sha(full_graph) != original_meta.get('graph_sha256') or sha(model) != meta.get('graph_sha256')):
        raise ValueError('Original/prefix source ONNX hash changed')
    lineage = _check_lineage(full_graph, model, meta)
    graph, raw = relative_file(bundle, item['graph']), relative_file(bundle, item['raw'])
    checked = audit(graph, raw, meta)
    if checked != item.get('audit') or checked.get('host_compute_ops') != 0 or checked.get('npu_executed') is not False:
        raise ValueError('Compiled graph/weights/ABI audit changed')
    for name, shape in (('input', input_shape), ('output', output_shape)):
        spec = checked[name]
        if spec.get('layout') not in ('NCHW', 'NHWC') or spec.get('logical_shape') != shape:
            raise ValueError('Compiled wire ABI does not preserve logical tensor shape')
    samples, provenance = {}, {}
    for label, filename, hash_key in (('A', 'oracle.npz', 'oracle_sha256'), ('B', 'oracle_b.npz', 'oracle_b_sha256')):
        context = capture.get('contexts', {}).get(label) or {}
        stage = context.get('modules', {}).get(module) or {}
        _check_capture_shapes(stage, original_meta, input_shape)
        ref = source.get('original_arm_references', {}).get(label) or {}
        observed = ref.get('derived_intermediates', {}).get(partition) or {}
        arm_path = relative_file(arm_fixtures, Path(label) / stage.get('fixture', 'missing'))
        oracle = relative_file(graphs, Path(partition) / filename)
        if (context.get('context_views') != root_input_shape[1] or not stage.get('complete')
                or sha(arm_path) != stage.get('fixture_sha256') or sha(oracle) != meta.get(hash_key)
                or ref.get('context_sha256') != context.get('context_sha256')
                or ref.get('fixture_sha256') != sha(arm_path)
                or observed.get('source_output_tensor') != meta.get('endpoint_tensor')
                or 'ONNX' not in observed.get('provenance', '')):
            raise ValueError('Real A/B full-stage and ONNX-derived oracle linkage differs: ' + label)
        with np.load(arm_path, allow_pickle=False) as z:
            if set(z.files) != {'input', 'expected'}:
                raise ValueError('Original ARM fixture fields changed')
            original_input, original_expected = z['input'].copy(), z['expected'].copy()
        with np.load(oracle, allow_pickle=False) as z:
            if set(z.files) != {'input', 'expected'}:
                raise ValueError('Derived oracle fields changed')
            value, expected = z['input'].copy(), z['expected'].copy()
        if (array_sha(original_input) != stage.get('input_sha256')
                or list(original_input.shape) != stage['input_shape']
                or list(original_expected.shape) != stage['output_shape']
                or original_expected.dtype != np.float32 or not np.isfinite(original_expected).all()
                or array_sha(original_expected) != stage.get('expected_sha256')
                or array_sha(original_expected) != ref.get('original_arm_full_expected_sha256')
                or array_sha(value) != ref.get('input_sha256')
                or array_sha(expected) != observed.get('sha256')
                or not np.array_equal(value, original_input)):
            raise ValueError('Original ARM input or derived intermediate oracle array changed: ' + label)
        if (value.dtype != np.float32 or expected.dtype != np.float32
                or list(value.shape) != input_shape or list(expected.shape) != output_shape
                or not np.isfinite(value).all() or not np.isfinite(expected).all()
                or not np.count_nonzero(value)):
            raise ValueError('Need finite, nonzero, FP32 real A/B tensors with declared shapes')
        full_check = (source.get('original_arm_full_graph_checks') or {}).get(label) or {}
        host_check = (meta.get('host_onnx_checks') or {}).get(label) or {}
        if not full_check.get('passed') or not full_check.get('finite') or not host_check.get('passed'):
            raise ValueError('Missing pinned full-ARM or derived-prefix host check: ' + label)
        samples[label] = value, expected
        provenance[label] = dict(context_sha256=context['context_sha256'],
            original_arm_fixture_sha256=sha(arm_path), original_arm_input_sha256=array_sha(original_input),
            original_arm_full_expected_sha256=array_sha(original_expected),
            derived_oracle_sha256=sha(oracle), derived_intermediate_sha256=array_sha(expected),
            oracle_provenance=ORACLE, independent_arm_intermediate=False)
    if (provenance['A']['context_sha256'] == provenance['B']['context_sha256']
            or np.array_equal(samples['A'][0], samples['B'][0]) or compare(samples['A'][1], samples['B'][1])['passed']):
        raise ValueError('Need genuinely distinct real A/B inputs and distinguishable expected outputs')
    return dict(name=partition, graph=graph, raw=raw,
        input_spec=checked['input'], output_spec=checked['output'], samples=samples,
        provenance=provenance, lineage=lineage, audit=checked,
        identities=dict(manifests={k: dict(path=str(paths[k]), sha256=sha(paths[k])) for k in paths},
            source_model_sha256=sha(full_graph), prefix_model_sha256=sha(model),
            compiled_graph_sha256=sha(graph), compiled_raw_sha256=sha(raw), weight_sha256=weight))


def run_sequence(session, samples, input_spec, output_spec, *, sequence='aabbaa', sdk_passes=1):
    """Charge each bridge/SDK call, including stale settle passes and failures."""
    if sequence not in SEQUENCES or type(sdk_passes) is not int or sdk_passes not in (1, 2):
        raise ValueError('Select a fixed sequence and one/two explicit charged SDK passes')
    labels = SEQUENCES[sequence]
    if set(samples) != {'A', 'B'}:
        raise ValueError('Exactly A and B samples required')
    for value, expected in samples.values():
        if (value.dtype != np.float32 or expected.dtype != np.float32
                or not np.isfinite(value).all() or not np.isfinite(expected).all()):
            raise ValueError('Sequence requires finite FP32 samples')
    if np.array_equal(samples['A'][0], samples['B'][0]) or compare(samples['A'][1], samples['B'][1])['passed']:
        raise ValueError('Sequence requires distinct inputs and distinguishable expected outputs')
    rows, requests, outputs, pass_outputs = [], [], {}, {}
    error = None
    t0 = time.perf_counter()
    for index, label in enumerate(labels):
        started = time.perf_counter()
        value, expected = samples[label]
        packed_at = time.perf_counter()
        wire = to_wire(value, input_spec)
        packing_seconds = time.perf_counter() - packed_at
        final = None
        for pass_index in range(sdk_passes):
            output_wire = np.full(output_spec['shape'], np.nan, dtype=np.float32)
            row = dict(request=index, label=label, pass_index=pass_index,
                is_final_pass=pass_index == sdk_passes - 1, input_sha256=array_sha(value),
                sdk_attempted=True, sdk_completed=False, finite=False,
                oracle_provenance=ORACLE)
            call_started = time.perf_counter()
            try:
                timing = session.forward_into(wire, output_wire)
                row['sdk_completed'] = True
                row['wire_output_sha256'] = array_sha(output_wire)
                row['finite'] = bool(np.isfinite(output_wire).all())
                if (not isinstance(timing, dict) or any(
                        type(v) not in (int, float) or not np.isfinite(v) or v < 0
                        for v in timing.values())):
                    raise ValueError('Invalid bridge timing measurements')
                row['bridge_timing_seconds'] = timing
                if not row['finite']:
                    raise ValueError('Bridge left nonfinite or unwritten output')
                unpack_started = time.perf_counter()
                actual = from_wire(output_wire, output_spec)
                row['output_unpack_seconds'] = time.perf_counter() - unpack_started
                row['output_sha256'] = array_sha(actual)
                row['source_onnx_reference'] = compare(actual, expected)
                other = 'B' if label == 'A' else 'A'
                row['matches_other_reference'] = compare(actual, samples[other][1])['passed']
                row['exact_previous_other_output'] = other in outputs and np.array_equal(actual, outputs[other][0])
                row['stale_output_suspected'] = bool(not row['source_onnx_reference']['passed']
                    and (row['matches_other_reference'] or row['exact_previous_other_output']))
                pass_key = (label, pass_index)
                row['same_input_same_pass_repeat_exact'] = (None if pass_key not in pass_outputs
                    else bool(np.array_equal(actual, pass_outputs[pass_key])))
                pass_outputs.setdefault(pass_key, actual.copy())
                final = actual
            except Exception as exc:
                error = type(exc).__name__ + ': ' + str(exc)
                row['error'] = error
            finally:
                row['call_wall_seconds'] = time.perf_counter() - call_started
                rows.append(row)
            if error:
                break
        if error:
            break
        repeat = None if label not in outputs else bool(np.array_equal(final, outputs[label][0]))
        outputs.setdefault(label, (final.copy(), []))[1].append(repeat)
        requests.append(dict(index=index, label=label, sdk_passes=sdk_passes,
            input_pack_seconds=packing_seconds, request_wall_seconds=time.perf_counter() - started,
            final_output_sha256=array_sha(final), source_onnx_reference=compare(final, expected),
            repeat_exact=repeat))
    completed = len(requests) == len(labels)
    repeat_exact = completed and all(x['repeat_exact'] is not False for x in requests)
    actual_distinct = set(outputs) == {'A', 'B'} and not compare(outputs['A'][0], outputs['B'][0])['passed']
    final_pass = completed and all(x['source_onnx_reference']['passed'] for x in requests)
    summary = dict(complete=completed, diagnostic_only=True, acceptance=False,
        production_qualified=False, oracle_provenance=ORACLE, sequence=list(labels),
        explicit_sdk_passes_per_request=sdk_passes, sdk_forward_attempted=len(rows),
        sdk_forward_completed=sum(x['sdk_completed'] for x in rows),
        all_sdk_passes_strict_pass=bool(rows) and all((x.get('source_onnx_reference') or {}).get('passed', False) for x in rows),
        final_outputs_strict_pass=final_pass, repeat_exact=repeat_exact,
        all_sdk_same_input_same_pass_repeats_exact=all(
            x.get('same_input_same_pass_repeat_exact') is not False for x in rows),
        actual_ab_distinguishable=bool(actual_distinct),
        stale_output_detected=any(x.get('stale_output_suspected', False) for x in rows),
        stale_pass_count=sum(x.get('stale_output_suspected', False) for x in rows),
        stale_final_pass_count=sum(x.get('stale_output_suspected', False) and x['is_final_pass'] for x in rows),
        nonfinite_or_unwritten_count=sum(x['sdk_completed'] and not x['finite'] for x in rows),
        all_call_wall_seconds=sum(x['call_wall_seconds'] for x in rows),
        bridge_component_sums_seconds={k: sum(x.get('bridge_timing_seconds', {}).get(k, 0.) for x in rows)
            for k in sorted({k for x in rows for k in x.get('bridge_timing_seconds', {})})},
        wall_seconds=time.perf_counter() - t0)
    summary['diagnostic_pass'] = completed and final_pass and repeat_exact and bool(actual_distinct)
    summary['timing_scope'] = 'All attempted bridge calls and explicit passes; SDK total overlaps its components and must not be added twice'
    return dict(complete=completed, summary=summary, requests=requests, sdk_calls=rows,
                **({'error': error} if error else {}))


def execute(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = dict(schema=SCHEMA, complete=False, diagnostic_only=True, acceptance=False,
        production_qualified=False, formal_receipt=False, npu_executed=False,
        oracle_provenance=ORACLE, independent_arm_intermediate=False,
        production_runtime_modified=False, partition=args.partition,
        scope='Dedicated source-ONNX-derived prefix diagnostic; never production or arbitrary-scene qualification')
    started = time.perf_counter()
    try:
        pins = {k: getattr(args, k + '_manifest_sha256') for k in MANIFEST_KEYS}
        spec = validate_candidate(args.graphs, args.bundle, args.partition,
                                  args.full_source, args.arm_fixtures, pins)
        record.update(assets=spec['identities'], provenance=spec['provenance'],
                      lineage=spec['lineage'], host_audit=spec['audit'], validation_passed=True)
        if args.preflight_only:
            record.update(complete=True, preflight_only=True, execution_performed=False)
            return record
        if not args.library or not args.library_sha256 or sha(args.library) != args.library_sha256:
            raise ValueError('Execution requires the explicitly pinned ARM bridge library')
        if os.environ.get('MGS_NPU_DDR_GUARD_MIB') != '64':
            raise ValueError('Keep the explicit 64 MiB device DDR reservation')
        if any(os.environ.get(k) not in (None, '0') for k in ('MGS_NPU_DIRECT_SFB', 'MGS_NPU_REAPPLY', 'MGS_NPU_FORCE_WAIT', 'MGS_NPU_DISABLE_ETM')):
            raise ValueError('This controlled diagnostic excludes extra bridge switches')
        if os.environ.get('MGS_NPU_FRESH_INPUT') not in (None, '1'):
            raise ValueError('Source-prefix diagnostic requires fresh input tensors')
        os.environ['MGS_NPU_FRESH_INPUT'] = '1'
        record['bridge_sha256'] = sha(args.library)
        record['execution_environment'] = {k: os.environ.get(k) for k in (
            'MGS_NPU_FRESH_INPUT', 'MGS_NPU_FRESH_OUTPUT', 'MGS_NPU_DDR_GUARD_MIB',
            'MGS_NPU_DIRECT_SFB', 'MGS_NPU_REAPPLY', 'MGS_NPU_FORCE_WAIT', 'MGS_NPU_DISABLE_ETM')}
        setup_started = time.perf_counter()
        with BridgeSession(spec, args.library) as session:
            record['session_setup_seconds'] = time.perf_counter() - setup_started
            record['session'] = dict(one_session=True, bindings=session.bindings, memory_layout=session.memory_layout)
            run = run_sequence(session, spec['samples'], spec['input_spec'], spec['output_spec'],
                               sequence=args.sequence, sdk_passes=args.sdk_passes)
            record.update(run)
            record['npu_executed'] = run['summary']['sdk_forward_completed'] > 0
            record['execution_performed'] = True
        return record
    except BaseException as exc:
        record['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        record['total_wall_seconds_including_validation_and_session'] = time.perf_counter() - started
        save(out / 'prefix_diagnostic_result.json', record)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('graphs', 'bundle', 'full-source', 'arm-fixtures', 'out'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--partition', required=True)
    for key in MANIFEST_KEYS:
        p.add_argument('--' + key.replace('_', '-') + '-manifest-sha256', required=True)
    p.add_argument('--library', type=Path)
    p.add_argument('--library-sha256')
    p.add_argument('--sequence', choices=tuple(SEQUENCES), default='aabbaa')
    p.add_argument('--sdk-passes', type=int, choices=(1, 2), default=1,
                   help='Every SDK pass is separately recorded; no hidden settling or discarded timing')
    p.add_argument('--preflight-only', action='store_true', help='Validate local files without loading the bridge or opening the NPU')
    args = p.parse_args(argv)
    record = execute(args)
    print(json.dumps(dict(complete=record['complete'], diagnostic_only=True,
        acceptance=False, production_qualified=False, oracle_provenance=ORACLE,
        preflight_only=args.preflight_only, npu_executed=record['npu_executed'],
        diagnostic_pass=(record.get('summary') or {}).get('diagnostic_pass')), allow_nan=False))
    if not args.preflight_only and not (record.get('summary') or {}).get('diagnostic_pass'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
