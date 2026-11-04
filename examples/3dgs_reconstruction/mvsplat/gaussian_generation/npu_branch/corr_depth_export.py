"""Compose unchanged cost refinement, residual and depth-head ONNX graphs.

The candidate returns low-resolution depth logits; softmax, dynamic geometric
warping and the remaining model stay outside. No production module is replaced.
Its new A/B oracles are source-ONNX-derived, not independent ARM depth captures.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np

from common import new_directory, save, sha
from gaussian_generation.npu_branch.unet_export import numerical_comparison


NAME = 'corr_depth_pipeline'


def _shape(value):
    import onnx
    tensor = value.type.tensor_type
    shape = [d.dim_value for d in tensor.shape.dim]
    if tensor.elem_type != onnx.TensorProto.FLOAT or len(shape) != 4 or min(shape) < 1:
        raise ValueError('Expected static positive rank-four FP32 ONNX boundary')
    return shape


def _unary(model):
    import onnx
    if (len(model.graph.input) != 1 or len(model.graph.output) != 1
            or model.functions or model.graph.sparse_initializer):
        raise ValueError('Expected a self-contained unary graph without local functions')
    if any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
           for node in model.graph.node for a in node.attribute):
        raise ValueError('Nested control-flow graphs need an explicit renaming contract')
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer):
        raise ValueError('External tensor data must first be materialized and hash-bound')
    return _shape(model.graph.input[0]), _shape(model.graph.output[0])


def _logical_corr(model, layout):
    """Remove a real terminal layout conversion, never relabel its tensor."""
    import onnx
    result = copy.deepcopy(model)
    incoming, outgoing = _unary(result)
    if layout == 'NCHW':
        return result, incoming, outgoing
    if layout != 'NHWC' or not result.graph.node:
        raise ValueError('Unsupported correlation source output layout')
    last = result.graph.node[-1]
    perm = next((list(a.ints) for a in last.attribute if a.name == 'perm'), None)
    if (last.op_type != 'Transpose' or last.domain not in ('', 'ai.onnx')
            or perm != [0, 2, 3, 1] or len(last.input) != 1
            or list(last.output) != [result.graph.output[0].name]):
        raise ValueError('NHWC source requires the exact terminal NCHW-to-NHWC Transpose')
    logical = [outgoing[i] for i in (0, 3, 1, 2)]
    output = onnx.helper.make_tensor_value_info(last.input[0], onnx.TensorProto.FLOAT, logical)
    del result.graph.node[-1]
    result.graph.output[0].CopyFrom(output)
    onnx.checker.check_model(result)
    return result, incoming, logical


def _branch(model, prefix, input_alias):
    result = copy.deepcopy(model)
    incoming = result.graph.input[0].name
    def rename(value):
        return input_alias if value == incoming else prefix+value if value else ''
    for node in result.graph.node:
        node.name = prefix+node.name
        for i, value in enumerate(node.input):
            node.input[i] = rename(value)
        for i, value in enumerate(node.output):
            node.output[i] = rename(value)
    for tensor in result.graph.initializer:
        tensor.name = rename(tensor.name)
    return list(result.graph.node), list(result.graph.initializer), rename(result.graph.output[0].name)


def compose(correlation, residual, head, *, corr_output_layout='NHWC'):
    """Keep both views and every trained value; wire x -> head(corr(x)+skip(x))."""
    import onnx
    from onnx import helper, TensorProto
    correlation, incoming, refined = _logical_corr(correlation, corr_output_layout)
    residual_in, residual_out = _unary(residual)
    head_in, outgoing = _unary(head)
    if (incoming != residual_in or refined != residual_out or refined != head_in
            or incoming[0] != outgoing[0] or incoming[2:] != outgoing[2:]):
        raise ValueError('Correlation/residual/head batch, channels or spatial boundary mismatch')
    models = (correlation, residual, head)
    versions = [{item.domain: item.version for item in model.opset_import} for model in models]
    if any(v != versions[0] for v in versions[1:]):
        raise ValueError('Source ONNX operator-set versions differ')
    branches = [_branch(correlation, 'corr/', 'features'),
                _branch(residual, 'residual/', 'features'),
                _branch(head, 'head/', 'combined')]
    nodes = branches[0][0]+branches[1][0]+[
        helper.make_node('Add', [branches[0][2], branches[1][2]], ['combined'],
                         name='original_residual_add')]+branches[2][0]
    nodes.append(helper.make_node('Transpose', [branches[2][2]], ['result_nhwc'],
                                  name='explicit_logits_nchw_to_nhwc', perm=[0, 2, 3, 1]))
    graph = helper.make_graph(nodes, 'mvsplat_corr_residual_depth_pipeline',
        [helper.make_tensor_value_info('features', TensorProto.FLOAT, incoming)],
        [helper.make_tensor_value_info('result_nhwc', TensorProto.FLOAT,
                                      [outgoing[i] for i in (0, 2, 3, 1)])],
        [t for _, initializers, _ in branches for t in initializers])
    model = helper.make_model(graph, opset_imports=list(correlation.opset_import),
                              producer_name='mvsplat_corr_depth_candidate')
    model.ir_version = max(m.ir_version for m in models)
    onnx.checker.check_model(model)
    elements = lambda shape: int(np.prod(shape))*4
    # Static logical boundaries only: neither actual DDR traffic nor SDK passes.
    all_npu = sum(elements(s) for s in (incoming, refined, incoming, refined, refined, outgoing))
    split_corr_head = sum(elements(s) for s in (incoming, refined, refined, outgoing))
    fused = elements(incoming)+elements(outgoing)
    return model, dict(input_shape=incoming, output_shape=outgoing,
        onnx_output_shape=[outgoing[i] for i in (0, 2, 3, 1)],
        onnx_output_layout='NHWC', logical_layout='NCHW', dtype='float32',
        source_parameters={label: sum(int(np.prod(t.dims)) for t in original.graph.initializer)
                           for label, original in zip(('corr', 'residual', 'head'), models)},
        floating_point_operations_changed=False, trained_parameters_changed=False,
        whole_view_batch_preserved=True, geometry_warping_included=False,
        softmax_included=False, output_meaning='low-resolution depth logits before softmax',
        logical_io_bytes=dict(three_npu_boundaries=all_npu,
            corr_head_npu_residual_cpu=split_corr_head, continuous=fused,
            saved_vs_three=all_npu-fused, saved_vs_corr_head=split_corr_head-fused),
        logical_io_scope='static FP32 shape bytes once per logical call; not actual DDR traffic')


def _session(model):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(model.SerializeToString(), sess_options=options,
                                providers=['CPUExecutionProvider'])


def _run(session, value, layout):
    wire = session.run(None, {session.get_inputs()[0].name: value})[0]
    result = wire.transpose(0, 3, 1, 2) if layout == 'NHWC' else wire
    if result.dtype != np.float32 or result.ndim != 4 or not np.isfinite(result).all():
        raise ValueError('Source or composed graph returned invalid FP32 values')
    return np.ascontiguousarray(result)


def _source(folder, name):
    import onnx
    manifest_path = Path(folder)/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema') != 'mvsplat_partitions_v1' or manifest.get('npu_executed') is not False:
        raise ValueError('Expected unchanged source-only export manifest')
    meta = manifest['partitions'][name]
    graph, oracle = Path(folder)/name/'model.onnx', Path(folder)/name/'oracle.npz'
    if (not meta.get('complete') or not meta.get('host_onnx_passed')
            or sha(graph) != meta['graph_sha256'] or sha(oracle) != meta['oracle_sha256']):
        raise ValueError('Source graph/oracle incomplete or checksum changed: '+name)
    model = onnx.load(str(graph), load_external_data=False)
    incoming, raw_outgoing = _unary(model)
    layout = meta.get('onnx_output_layout', 'NCHW')
    if layout not in ('NCHW', 'NHWC') or (name != 'corr_refine_net' and layout != 'NCHW'):
        raise ValueError('Residual and head sources require NCHW; no implicit layout conversion')
    outgoing = [raw_outgoing[i] for i in (0, 3, 1, 2)] if layout == 'NHWC' else raw_outgoing
    if (meta.get('dtype') != 'float32' or meta.get('logical_layout') != 'NCHW'
            or meta.get('returns_list') is not False or incoming != meta['input_shape']
            or outgoing != meta['output_shape']):
        raise ValueError('Source ONNX boundary differs from logical metadata: '+name)
    session = _session(model)
    with np.load(oracle, allow_pickle=False) as data:
        value, reference = data['input'], data['expected']
        if value.dtype != np.float32 or reference.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Invalid unchanged source oracle')
        if list(value.shape) != incoming or list(reference.shape) != outgoing:
            raise ValueError('Source oracle shape differs from ONNX boundary: '+name)
        actual = _run(session, value, layout)
        check = numerical_comparison(actual, reference, rtol=2e-3, atol=2e-4)
    if not check['passed']:
        raise ValueError('Source ONNX no longer matches original oracle: '+name)
    provenance = dict(manifest_sha256=sha(manifest_path), graph_sha256=sha(graph),
        oracle_sha256=sha(oracle), context_sha256=manifest['context_sha256'],
        source_oracle_check=check)
    return manifest, meta, model, session, provenance


def export(corr_export, linear_export, fixtures, out):
    """Prepare a local candidate from hash-verified exports and real ARM inputs."""
    import onnx
    from gaussian_generation.npu_branch.diagnostic_runner import array_sha
    sources = [_source(corr_export, 'corr_refine_net'),
               _source(linear_export, 'regressor_residual'),
               _source(linear_export, 'depth_head')]
    base = sources[0][0]
    if any(m['weight_sha256'] != base['weight_sha256'] or m['input_shape'] != base['input_shape']
           or m['source_commit'] != base['source_commit'] for m, *_ in sources):
        raise ValueError('Source checkpoint, upstream revision or encoder input differs')
    capture_path = Path(fixtures)/'capture_manifest.json'
    capture = json.loads(capture_path.read_text(encoding='utf-8'))
    if (capture.get('schema') != 'mvsplat_real_context_stage_fixtures_v1'
            or not capture.get('complete') or capture.get('npu_executed') is not False
            or capture.get('reference_device') != 'board ARM CPU'
            or capture['weight_sha256'] != base['weight_sha256']
            or capture['contexts']['A']['context_sha256'] == capture['contexts']['B']['context_sha256']):
        raise ValueError('Expected unchanged, distinct real ARM CPU view-pair captures')
    layout = sources[0][1].get('onnx_output_layout', 'NCHW')
    model, meta = compose(*(s[2] for s in sources), corr_output_layout=layout)
    session = _session(model)
    samples, references, fixture_provenance = {}, {}, {}
    for label in ('A', 'B'):
        context = capture['contexts'][label]
        item = context['modules']['corr_refine_net']
        if not item['complete'] or context['context_views'] != 2 or Path(item['fixture']).name != item['fixture']:
            raise ValueError('Invalid original full two-view capture metadata')
        path = Path(fixtures)/label/item['fixture']
        if sha(path) != item['fixture_sha256']:
            raise ValueError('Changed real ARM source fixture')
        with np.load(path, allow_pickle=False) as data:
            value, arm_corr = data['input'].copy(), data['expected'].copy()
        if (value.dtype != np.float32 or arm_corr.dtype != np.float32
                or not np.isfinite(value).all() or not np.isfinite(arm_corr).all()
                or array_sha(value) != item['input_sha256'] or array_sha(arm_corr) != item['expected_sha256']
                or list(value.shape) != meta['input_shape'] or list(arm_corr.shape) != sources[0][1]['output_shape']):
            raise ValueError('Original capture input/output identity or boundary mismatch')
        corr = _run(sources[0][3], value, layout)
        check = numerical_comparison(corr, arm_corr, rtol=2e-3, atol=2e-4)
        if not check['passed']:
            raise ValueError('Full correlation source ONNX differs from ARM hook capture')
        residual = _run(sources[1][3], value, 'NCHW')
        reference = _run(sources[2][3], np.ascontiguousarray(corr+residual), 'NCHW')
        samples[label], references[label] = value, reference
        fixture_provenance[label] = dict(context_sha256=context['context_sha256'],
            arm_fixture_sha256=sha(path), input_sha256=array_sha(value),
            arm_corr_check=check, expected_sha256=array_sha(reference))
    if array_sha(samples['A']) == array_sha(samples['B']) or array_sha(references['A']) == array_sha(references['B']):
        raise ValueError('Real A/B inputs and expected logits must differ')
    out = new_directory(Path(out))
    folder = new_directory(out/NAME)
    graph = folder/'model.onnx'
    onnx.save(model, str(graph))
    history = []
    repeated = {}
    for label in ('A', 'B', 'B', 'A'):
        actual = _run(session, samples[label], 'NHWC')
        check = numerical_comparison(actual, references[label], rtol=2e-3, atol=2e-4)
        exact_repeat = label not in repeated or np.array_equal(actual, repeated[label])
        history.append(dict(input=label, **check, repeat_exact=exact_repeat))
        if not check['passed'] or not exact_repeat:
            save(out/'host_failure.json', dict(diagnostic_only=True, checks=history))
            raise ValueError('Continuous ONNX graph differs or is not repeatable')
        repeated[label] = actual.copy()
    for label, file in (('A', 'oracle.npz'), ('B', 'oracle_alternate.npz')):
        np.savez(folder/file, input=samples[label], expected=references[label])
    meta.update(complete=True, host_onnx_passed=True, returns_list=False, path=NAME,
        full_module=True, diagnostic_only=True, production_enabled=False,
        requires_new_board_qualification=True, npu_executed=False,
        oracle_provenance='source_onnx_derived_chain_on_original_real_ARM_inputs',
        independent_arm_depth_reference=False,
        composed_modules=['corr_refine_net', 'regressor_residual', 'depth_head_lowres'],
        cpu_remainder='softmax, disparity expectation, warping, upsampling, Gaussian adapter',
        source_provenance={name: s[4] for name, s in zip(
            ('corr_refine_net', 'regressor_residual', 'depth_head'), sources)},
        fixtures=fixture_provenance, host_sequence=history,
        parameters=sum(meta['source_parameters'].values()),
        graph_sha256=sha(graph), oracle_sha256=sha(folder/'oracle.npz'),
        alternate_oracle_sha256=sha(folder/'oracle_alternate.npz'),
        tolerance=dict(rtol=2e-3, atol=2e-4), ops=dict(Counter(n.op_type for n in model.graph.node)))
    manifest = dict(schema='mvsplat_partitions_v1', diagnostic_only=True,
        npu_executed=False, production_enabled=False, acceptance_receipt=False,
        weight_sha256=base['weight_sha256'], input_shape=base['input_shape'],
        source_commit=base['source_commit'], context_sha256=capture['contexts']['A']['context_sha256'],
        source_contexts_preserved=True, capture_manifest_sha256=sha(capture_path),
        exporter_sha256=sha(Path(__file__)), partitions={NAME: meta})
    save(out/'manifest.json', manifest)
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corr-export', type=Path, required=True)
    p.add_argument('--linear-export', type=Path, required=True)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    result = export(a.corr_export, a.linear_export, a.fixtures, a.out)
    meta = result['partitions'][NAME]
    print(json.dumps(dict(complete=meta['complete'], npu_executed=False,
                         logical_io_bytes=meta['logical_io_bytes']), indent=2))


if __name__ == '__main__':
    main()
