"""Opt-in full-resolution projection/refinement/two-head ONNX composition.

The unary input is [upsampled features, RGB, coarse disparity, confidence].
Upsampling remains outside; its original 128-channel result fans out inside
this graph to projection and Gaussian prediction. No trained value, two-view
attention or normalization is changed. Output is [Gaussian raw84, disparity
raw2] with an explicit NHWC wire conversion. Preparation is not board approval.

Fixture schema ``mvsplat_refine_heads_inputs_v1`` binds packed input arrays and
records their stated provenance. References are independently executed source
ONNX chains, never promoted to real ARM/full-video reconstruction oracles.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import re

import numpy as np

from common import FULL_WEIGHT_SHA256, SOURCE_COMMIT, new_directory, save, sha
from gaussian_generation.npu_branch.corr_depth_export import (
    _branch, _run, _session, _unary,
)
from gaussian_generation.npu_branch.unet_export import numerical_comparison


NAME = 'refine_heads_pipeline'
FIXTURE_SCHEMA = 'mvsplat_refine_heads_inputs_v1'
SOURCE_NAMES = ('proj_feature', 'refine_unet', 'to_gaussians', 'to_disparity')


def _logical_refine(model, layout):
    """Remove only a verified terminal layout transform; retain all Conv nodes."""
    import onnx
    result = copy.deepcopy(model)
    incoming, outgoing = _unary(result)
    if layout == 'NCHW':
        onnx.checker.check_model(result)
        return result, incoming, outgoing
    if layout != 'NHWC' or not result.graph.node:
        raise ValueError('Unsupported refinement source output layout')
    last = result.graph.node[-1]
    perm = next((list(a.ints) for a in last.attribute if a.name == 'perm'), None)
    if (last.op_type != 'Transpose' or last.domain not in ('', 'ai.onnx') or
            perm != [0, 2, 3, 1] or len(last.input) != 1 or
            list(last.output) != [result.graph.output[0].name]):
        raise ValueError('NHWC refinement requires the exact terminal NCHW-to-NHWC Transpose')
    logical = [outgoing[i] for i in (0, 3, 1, 2)]
    result.graph.output[0].CopyFrom(onnx.helper.make_tensor_value_info(
        last.input[0], onnx.TensorProto.FLOAT, logical))
    del result.graph.node[-1]
    onnx.checker.check_model(result)
    return result, incoming, logical


def compose(projector, refinement, gaussian, disparity, *, refine_output_layout='NHWC',
            rgb_channels=3, coarse_channels=1, confidence_channels=1,
            head_tile_core=32, head_halo=2):
    """Compose exact source operators; small static shapes are allowed for tests.

    In production, ``export`` separately enforces 2x133x128x128 -> 2x86x128x128.
    Tiling arguments describe only the existing comparison contract; the new
    full-view refinement graph is never tiled or split into separate views.
    """
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    if (any(type(value) is not int or value < 1 for value in
            (rgb_channels, coarse_channels, confidence_channels, head_tile_core)) or
            type(head_halo) is not int or head_halo < 0):
        raise ValueError('Expected positive channel/tile counts and nonnegative halo')
    up_shape, projection_shape = _unary(projector)
    refinement, refinement_shape, refined_shape = _logical_refine(refinement, refine_output_layout)
    gaussian_in, gaussian_out = _unary(gaussian)
    disparity_in, disparity_out = _unary(disparity)
    batch, up_channels, height, width = up_shape
    projected_channels = projection_shape[1]
    refine_channels = refined_shape[1]
    def shape(channels):
        return [batch, channels, height, width]
    if (batch != 2 or projection_shape != shape(projected_channels) or
            refinement_shape != shape(rgb_channels + projected_channels + coarse_channels + confidence_channels) or
            refined_shape != shape(refine_channels) or
            gaussian_in != shape(refine_channels + rgb_channels + up_channels) or
            disparity_in != refined_shape or gaussian_out != shape(gaussian_out[1]) or
            disparity_out != shape(disparity_out[1])):
        raise ValueError('Projection/refinement/heads two-view, channels or spatial boundary mismatch')
    models = (projector, refinement, gaussian, disparity)
    versions = [{item.domain: item.version for item in model.opset_import} for model in models]
    if any(item != versions[0] for item in versions[1:]):
        raise ValueError('Source ONNX operator-set versions differ')
    incoming = shape(up_channels + rgb_channels + coarse_channels + confidence_channels)
    outgoing = shape(gaussian_out[1] + disparity_out[1])
    nodes, constants = [], []
    first = 0
    for label, channels in (('up', up_channels), ('rgb', rgb_channels),
                            ('coarse', coarse_channels), ('confidence', confidence_channels)):
        prefix = '__mgs_input_' + label
        for suffix, value in (('starts', first), ('ends', first+channels), ('axes', 1), ('steps', 1)):
            constants.append(numpy_helper.from_array(np.asarray([value], np.int64), prefix+'_'+suffix))
        nodes.append(helper.make_node('Slice', ['features', prefix+'_starts', prefix+'_ends',
            prefix+'_axes', prefix+'_steps'], [label], name=prefix+'_slice'))
        first += channels
    projection = _branch(projector, 'projection/', 'up')
    refine = _branch(refinement, 'refine/', 'refine_features')
    gauss = _branch(gaussian, 'gaussians/', 'gaussian_features')
    disp = _branch(disparity, 'disparity/', refine[2])
    nodes += projection[0]
    nodes.append(helper.make_node('Concat', ['rgb', projection[2], 'coarse', 'confidence'],
                                  ['refine_features'], axis=1, name='original_refine_input'))
    nodes += refine[0]
    nodes.append(helper.make_node('Concat', [refine[2], 'rgb', 'up'], ['gaussian_features'],
                                  axis=1, name='original_gaussian_input_with_reused_up'))
    nodes += gauss[0] + disp[0]
    nodes.append(helper.make_node('Concat', [gauss[2], disp[2]], ['packed_raw_nchw'],
                                  axis=1, name='packed_gaussian_disparity_raw'))
    wire_shape = [outgoing[i] for i in (0, 2, 3, 1)]
    nodes.append(helper.make_node('Transpose', ['packed_raw_nchw'], ['result_nhwc'],
        perm=[0, 2, 3, 1], name='explicit_raw_heads_nchw_to_nhwc'))
    graph = helper.make_graph(nodes, 'mvsplat_projection_refinement_heads_pipeline',
        [helper.make_tensor_value_info('features', TensorProto.FLOAT, incoming)],
        [helper.make_tensor_value_info('result_nhwc', TensorProto.FLOAT, wire_shape)],
        constants + [value for branch in (projection, refine, gauss, disp) for value in branch[1]])
    model = helper.make_model(graph, opset_imports=list(projector.opset_import),
                              producer_name='mvsplat_refine_heads_candidate')
    model.ir_version = max(item.ir_version for item in models)
    onnx.checker.check_model(model)
    byte_count = lambda dimensions: int(np.prod(dimensions))*4
    full_heads = sum(byte_count(item) for item in
                    (refinement_shape, refined_shape, gaussian_in, gaussian_out, disparity_in, disparity_out))
    tiles = batch*((height+head_tile_core-1)//head_tile_core)*((width+head_tile_core-1)//head_tile_core)
    tile_size = head_tile_core+2*head_halo
    tiled_heads = sum(byte_count([1, item[1], tile_size, tile_size]) for item in
                      (gaussian_in, gaussian_out, disparity_in, disparity_out))*tiles
    existing_tiled = byte_count(refinement_shape)+byte_count(refined_shape)+tiled_heads
    continuous = byte_count(incoming)+byte_count(outgoing)
    return model, dict(input_shape=incoming, output_shape=outgoing,
        onnx_output_shape=wire_shape, logical_layout='NCHW', onnx_output_layout='NHWC', dtype='float32',
        packed_input=[dict(name=name, channels=channels) for name, channels in
            (('upsampled_features', up_channels), ('rgb', rgb_channels),
             ('coarse_disparity', coarse_channels), ('pdf_max', confidence_channels))],
        packed_output=[dict(name='raw_gaussians', channels=gaussian_out[1]),
                       dict(name='delta_disparity_raw_density', channels=disparity_out[1])],
        source_initializers={label: sum(int(np.prod(value.dims)) for value in original.graph.initializer)
                             for label, original in zip(SOURCE_NAMES, models)},
        source_layouts=dict(refine_output=refine_output_layout),
        upsampler_included=False, upsampled_feature_reused=True,
        full_view_batch_preserved=True, spatial_tiling_used=False,
        trained_parameters_changed=False, source_operators_changed=False,
        logical_io_bytes=dict(current_refine_and_tiled_heads=existing_tiled,
            refine_and_full_heads=full_heads,
            projection_refine_and_full_heads=full_heads+byte_count(up_shape)+byte_count(projection_shape),
            continuous=continuous, saved_vs_current_tiled=existing_tiled-continuous),
        logical_requests=dict(current_refine_and_tiled_heads=1+2*tiles, continuous=1),
        comparison_tiling=dict(core_size=head_tile_core, halo=head_halo, tiles_per_head=tiles),
        logical_io_scope='static FP32 boundary bytes once per logical request; not DDR traffic or SDK passes')


def _source(folder, name):
    """Audit a hash-bound source export and its original standalone ORT oracle."""
    import onnx
    if name not in SOURCE_NAMES:
        raise ValueError('Unknown projection/refinement/head source')
    folder = Path(folder)
    manifest_path = folder/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema') != 'mvsplat_partitions_v1' or manifest.get('npu_executed') is not False:
        raise ValueError('Expected unchanged source-only export manifest')
    meta = manifest['partitions'][name]
    graph, oracle = folder/name/'model.onnx', folder/name/'oracle.npz'
    if (not meta.get('complete') or not meta.get('host_onnx_passed') or
            sha(graph) != meta['graph_sha256'] or sha(oracle) != meta['oracle_sha256']):
        raise ValueError('Source graph/oracle incomplete or checksum changed: '+name)
    model = onnx.load(str(graph), load_external_data=False)
    incoming, wire_outgoing = _unary(model)
    layout = meta.get('onnx_output_layout', 'NCHW')
    if layout not in ('NCHW', 'NHWC') or (name != 'refine_unet' and layout != 'NCHW'):
        raise ValueError('Projection and prediction-head source outputs require NCHW')
    if name == 'refine_unet':
        _, _, outgoing = _logical_refine(model, layout)
    else:
        outgoing = wire_outgoing
    if (meta.get('dtype') != 'float32' or meta.get('logical_layout') != 'NCHW' or
            meta.get('returns_list') is not False or incoming != meta['input_shape'] or
            outgoing != meta['output_shape']):
        raise ValueError('Source ONNX boundary differs from logical metadata: '+name)
    session = _session(model)
    with np.load(oracle, allow_pickle=False) as data:
        value, expected = data['input'], data['expected']
        if (value.dtype != np.float32 or expected.dtype != np.float32 or
                not np.isfinite(value).all() or not np.isfinite(expected).all() or
                list(value.shape) != incoming or list(expected.shape) != outgoing):
            raise ValueError('Invalid unchanged source oracle boundary: '+name)
        check = numerical_comparison(_run(session, value, layout), expected, rtol=2e-3, atol=2e-4)
    if not check['passed']:
        raise ValueError('Source ONNX no longer matches original oracle: '+name)
    provenance = dict(manifest_sha256=sha(manifest_path), graph_sha256=sha(graph),
        oracle_sha256=sha(oracle), context_sha256=manifest['context_sha256'],
        source_oracle_check=check)
    return manifest, meta, model, session, provenance


def _source_reference(sources, value, *, up_channels, rgb_channels=3, coarse_channels=1):
    """Execute the separate original graphs, never the new composed graph."""
    up = np.ascontiguousarray(value[:, :up_channels])
    rgb = value[:, up_channels:up_channels+rgb_channels]
    coarse = value[:, up_channels+rgb_channels:up_channels+rgb_channels+coarse_channels]
    confidence = value[:, up_channels+rgb_channels+coarse_channels:]
    projection = _run(sources[0][3], up, 'NCHW')
    refine_input = np.ascontiguousarray(np.concatenate((rgb, projection, coarse, confidence), axis=1))
    refined = _run(sources[1][3], refine_input, sources[1][1].get('onnx_output_layout', 'NCHW'))
    gauss_input = np.ascontiguousarray(np.concatenate((refined, rgb, up), axis=1))
    gaussians = _run(sources[2][3], gauss_input, 'NCHW')
    disparity = _run(sources[3][3], refined, 'NCHW')
    return np.ascontiguousarray(np.concatenate((gaussians, disparity), axis=1))


def _fixtures(folder, shape, weight_sha256):
    from gaussian_generation.npu_branch.diagnostic_runner import array_sha
    folder = Path(folder)
    capture_path = folder/'capture_manifest.json'
    capture = json.loads(capture_path.read_text(encoding='utf-8'))
    if (capture.get('schema') != FIXTURE_SCHEMA or capture.get('complete') is not True or
            capture.get('npu_executed') is not False or capture.get('weight_sha256') != weight_sha256 or
            set(capture.get('contexts', {})) != {'A', 'B'} or
            not isinstance(capture.get('input_provenance'), str) or not capture['input_provenance'].strip() or
            not isinstance(capture.get('context_binding_scope'), str) or not capture['context_binding_scope'].strip() or
            capture.get('independent_arm_reference') or capture.get('reference_device') == 'board ARM CPU'):
        raise ValueError('Expected explicit packed-input diagnostic provenance, not an ARM/full-chain oracle')
    samples, supplied, provenance = {}, {}, {}
    for label in ('A', 'B'):
        context = capture['contexts'][label]
        fingerprint, fixture = context.get('context_sha256'), context.get('fixture')
        if (context.get('context_views') != 2 or not isinstance(fingerprint, str) or
                not re.fullmatch('[0-9a-f]{64}', fingerprint) or not isinstance(fixture, str) or
                Path(fixture).name != fixture or not fixture.endswith('.npz')):
            raise ValueError('Invalid two-view packed fixture identity or path')
        path = folder/fixture
        if sha(path) != context['fixture_sha256']:
            raise ValueError('Changed packed input fixture file')
        with np.load(path, allow_pickle=False) as data:
            if 'input' not in data.files or set(data.files)-{'input', 'expected'}:
                raise ValueError('Packed fixture only supports input and optional source-derived expected')
            value = data['input'].copy()
            expected = data['expected'].copy() if 'expected' in data.files else None
        if (value.dtype != np.float32 or list(value.shape) != shape or not np.isfinite(value).all() or
                array_sha(value) != context['input_sha256']):
            raise ValueError('Packed input shape, FP32/finite or array identity mismatch')
        if expected is not None:
            if (capture.get('reference_provenance') != 'source_onnx_derived' or
                    expected.dtype != np.float32 or not np.isfinite(expected).all() or
                    array_sha(expected) != context.get('expected_sha256')):
                raise ValueError('Optional expected array must be explicitly source-ONNX-derived and hash-bound')
            supplied[label] = expected
        elif 'expected_sha256' in context:
            raise ValueError('Expected-array digest has no corresponding fixture array')
        samples[label] = value
        provenance[label] = dict(context_sha256=fingerprint, context_views=2,
            fixture_sha256=sha(path), input_sha256=array_sha(value))
    if (capture['contexts']['A']['context_sha256'] == capture['contexts']['B']['context_sha256'] or
            array_sha(samples['A']) == array_sha(samples['B'])):
        raise ValueError('A/B input fingerprints and packed arrays must differ')
    return capture, capture_path, samples, supplied, provenance


def export(linear_export, refine_export, fixtures, out):
    """Create a fixed-128 candidate; source-derived oracles stay diagnostic-only."""
    import onnx
    from gaussian_generation.npu_branch.diagnostic_runner import array_sha
    sources = [_source(linear_export, 'proj_feature'), _source(refine_export, 'refine_unet'),
               _source(linear_export, 'to_gaussians'), _source(linear_export, 'to_disparity')]
    if any(manifest.get('weight_sha256') != FULL_WEIGHT_SHA256 or
            manifest.get('source_commit') != SOURCE_COMMIT or
            manifest.get('input_shape') != [1, 2, 3, 128, 128] for manifest, *_ in sources):
        raise ValueError('Source checkpoint, upstream revision or fixed encoder input differs')
    model, meta = compose(*(source[2] for source in sources),
        refine_output_layout=sources[1][1].get('onnx_output_layout', 'NCHW'))
    if meta['input_shape'] != [2, 133, 128, 128] or meta['output_shape'] != [2, 86, 128, 128]:
        raise ValueError('Full-source candidate requires the original fixed 128-resolution channel contract')
    if sources[0][1]['input_shape'] != [2, 128, 128, 128] or sources[1][1]['output_shape'] != [2, 32, 128, 128]:
        raise ValueError('Full-source projection/refinement channels differ')
    if sources[2][1]['output_shape'] != [2, 84, 128, 128] or sources[3][1]['output_shape'] != [2, 2, 128, 128]:
        raise ValueError('Full-source raw Gaussian/disparity channels differ')
    capture, capture_path, samples, supplied, provenance = _fixtures(fixtures, meta['input_shape'], FULL_WEIGHT_SHA256)
    references = {label: _source_reference(sources, value, up_channels=128)
                  for label, value in samples.items()}
    for label, expected in supplied.items():
        check = numerical_comparison(references[label], expected, rtol=2e-3, atol=2e-4)
        if not check['passed']:
            raise ValueError('Supplied source-derived expected differs from the original ONNX chain')
        provenance[label]['supplied_expected_check'] = check
    if array_sha(references['A']) == array_sha(references['B']):
        raise ValueError('Packed A/B interventions did not change predicted raw outputs')
    session = _session(model)
    history, repeated = [], {}
    for label in ('A', 'B', 'B', 'A'):
        actual = _run(session, samples[label], 'NHWC')
        check = numerical_comparison(actual, references[label], rtol=2e-3, atol=2e-4)
        repeat = label not in repeated or np.array_equal(actual, repeated[label])
        history.append(dict(input=label, **check, repeat_exact=repeat))
        if not check['passed'] or not repeat:
            raise ValueError('Composed full-resolution graph differs or is not repeatable')
        repeated[label] = actual.copy()
    out = new_directory(Path(out))
    folder = new_directory(out/NAME)
    graph = folder/'model.onnx'
    onnx.save(model, str(graph))
    for label, filename in (('A', 'oracle.npz'), ('B', 'oracle_alternate.npz')):
        np.savez(folder/filename, input=samples[label], expected=references[label])
        provenance[label]['expected_sha256'] = array_sha(references[label])
    meta.update(complete=True, host_onnx_passed=True, returns_list=False, path=NAME,
        full_module=True, diagnostic_only=True, production_enabled=False, acceptance_receipt=False,
        requires_new_board_qualification=True, npu_executed=False,
        oracle_provenance='source_onnx_derived_chain_on_explicit_packed_diagnostic_inputs',
        independent_arm_pipeline_reference=False,
        input_provenance=capture['input_provenance'], context_binding_scope=capture['context_binding_scope'],
        composed_modules=list(SOURCE_NAMES),
        cpu_remainder='upsampler, coarse depth softmax/expectation, density/depth postprocess, world Gaussian adapter',
        source_provenance={name: source[4] for name, source in zip(SOURCE_NAMES, sources)},
        fixtures=provenance, host_sequence=history,
        parameters=sum(source[1].get('parameters', meta['source_initializers'][name])
                       for name, source in zip(SOURCE_NAMES, sources)),
        graph_sha256=sha(graph), oracle_sha256=sha(folder/'oracle.npz'),
        alternate_oracle_sha256=sha(folder/'oracle_alternate.npz'),
        tolerance=dict(rtol=2e-3, atol=2e-4), ops=dict(Counter(node.op_type for node in model.graph.node)))
    manifest = dict(schema='mvsplat_partitions_v1', diagnostic_only=True,
        npu_executed=False, production_enabled=False, acceptance_receipt=False,
        weight_sha256=FULL_WEIGHT_SHA256, input_shape=[1, 2, 3, 128, 128], source_commit=SOURCE_COMMIT,
        context_sha256=capture['contexts']['A']['context_sha256'],
        input_provenance=capture['input_provenance'], context_binding_scope=capture['context_binding_scope'],
        reference_provenance='source_onnx_derived', independent_arm_pipeline_reference=False,
        capture_manifest_sha256=sha(capture_path), exporter_sha256=sha(Path(__file__)),
        partitions={NAME: meta})
    save(out/'manifest.json', manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--linear-export', type=Path, required=True,
                        help='Original seven-graph export containing projection and both full heads')
    parser.add_argument('--refine-export', type=Path, required=True,
                        help='Full refine_unet source export, with its audited logical/wire layout')
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    result = export(args.linear_export, args.refine_export, args.fixtures, args.out)
    print(json.dumps(dict(complete=True, npu_executed=False, diagnostic_only=True,
        logical_io_bytes=result['partitions'][NAME]['logical_io_bytes']), indent=2))


if __name__ == '__main__':
    main()
