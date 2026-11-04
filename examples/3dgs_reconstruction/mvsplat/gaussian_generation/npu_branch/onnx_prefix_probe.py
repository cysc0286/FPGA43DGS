"""Extract small diagnostic prefixes from an existing full U-Net ONNX graph.

These are source-ONNX-derived probes, not independent ARM intermediate
captures and never production partitions. Original nodes and initializers are
preserved byte-for-byte; only unused suffix nodes are removed and an explicit
NHWC output transpose is appended for the existing compiler host ABI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


PREFIXES = (
    ('first_conv', 'Conv', '/inner/inner.0/Conv'),
    ('first_groupnorm_affine', 'Add', '/inner/inner.1/Add'),
    ('first_groupnorm_gelu', 'Mul', '/inner/inner.2/Mul_1'),
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def extract_prefix(model, output_path, *, op_type, node_name, output_shape=None):
    """Write the exact original graph prefix ending at one named node."""
    import onnx
    from onnx import helper

    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {}
    for value in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output):
        tensor_shape = value.type.tensor_type.shape
        if value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT:
            dims = [d.dim_value if d.dim_value else None for d in tensor_shape.dim]
            shapes[value.name] = dims

    matches = [i for i, node in enumerate(inferred.graph.node)
               if node.op_type == op_type and node.name == node_name]
    if len(matches) != 1:
        raise ValueError(f'Expected one {op_type} node named {node_name!r}; found {len(matches)}')
    end = matches[0]
    endpoint = inferred.graph.node[end].output[0]
    shape = shapes.get(endpoint)
    if output_shape is not None:
        if len(output_shape) != 4 or any(not isinstance(d, int) or d <= 0 for d in output_shape):
            raise ValueError('Observed endpoint shape must have four positive dimensions')
        if shape and (len(shape) != len(output_shape)
                      or any(s is not None and s != actual for s, actual in zip(shape, output_shape))):
            raise ValueError('Observed endpoint shape disagrees with inferred dimensions')
        shape = list(output_shape)
    if not shape or len(shape) != 4 or any(d is None for d in shape):
        raise ValueError(f'Endpoint {endpoint!r} lacks a static rank-four shape: {shape}')
    n, c, h, w = shape
    input_shape = shapes.get(inferred.graph.input[0].name)
    if not input_shape or len(input_shape) != 4 or any(d is None for d in input_shape):
        raise ValueError('Source input must have a static rank-four shape')
    if n != input_shape[0]:
        raise ValueError('Prefix must preserve the complete source view batch')
    if any(dim <= 0 for dim in (n, c, h, w)):
        raise ValueError('Only positive static FP32 NCHW prefixes are supported')

    nodes = list(inferred.graph.node[:end + 1])
    used = {name for node in nodes for name in node.input}
    initializers = [value for value in inferred.graph.initializer if value.name in used]
    available = {value.name for value in inferred.graph.input} | {value.name for value in initializers}
    for node in nodes:
        missing = set(node.input) - available
        if missing:
            raise ValueError(f'Prefix has unresolved dependency at {node.name}: {sorted(missing)}')
        available.update(node.output)

    output_name = 'result_nhwc'
    transpose = helper.make_node('Transpose', [endpoint], [output_name],
                                 name='diagnostic_explicit_nchw_to_nhwc', perm=[0, 2, 3, 1])
    inputs = [value for value in inferred.graph.input
              if value.name not in {initializer.name for initializer in inferred.graph.initializer}]
    if len(inputs) != 1:
        raise ValueError('Expected one public input tensor')
    graph = helper.make_graph(nodes + [transpose], 'mvsplat_unet_diagnostic_prefix',
        inputs, [helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, [n, h, w, c])],
        initializer=initializers)
    result = helper.make_model(graph, opset_imports=list(inferred.opset_import),
                               producer_name=inferred.producer_name,
                               producer_version=inferred.producer_version)
    result.ir_version = inferred.ir_version
    onnx.checker.check_model(result)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(result, str(output_path))
    return dict(endpoint_node=node_name, endpoint_op=op_type, endpoint_tensor=endpoint,
        input_name=inputs[0].name, input_shape=input_shape,
        input_layout='NCHW', logical_output_shape=[n, c, h, w], logical_output_layout='NCHW',
        raw_output_name=output_name, raw_output_shape=[n, h, w, c], raw_output_layout='NHWC',
        source_node_count=len(inferred.graph.node), prefix_node_count=len(nodes),
        source_initializer_count=len(inferred.graph.initializer),
        retained_initializer_count=len(initializers),
        retained_initializer_parameters=sum(math.prod(value.dims) for value in initializers),
        original_parameters=sum(math.prod(value.dims) for value in initializers),
        parameters=sum(math.prod(value.dims) for value in initializers),
        source_initializer_bytes=sum(len(x.raw_data) for x in inferred.graph.initializer),
        retained_initializer_bytes=sum(len(x.raw_data) for x in initializers),
        source_initializers_preserved=True, graph_sha256=sha(output_path))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-export', type=Path, required=True)
    p.add_argument('--partition', choices=('corr_refine_net', 'refine_unet'), required=True)
    p.add_argument('--fixtures', type=Path, required=True,
                   help='Unchanged original real ARM full-stage capture_manifest.json directory')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    import onnx
    import numpy as np
    import onnxruntime as ort
    from .diagnostic_runner import array_sha, compare
    from collections import Counter
    if a.out.exists():
        p.error('Refusing to overwrite diagnostic artifacts')
    source_manifest_path = a.source_export / 'manifest.json'
    source_manifest = json.loads(source_manifest_path.read_text(encoding='utf-8'))
    source_meta = source_manifest['partitions'][a.partition]
    if (not source_meta.get('complete') or not source_meta.get('full_module')
            or source_manifest.get('npu_executed') is not False):
        raise ValueError('Expected a complete source-only full-module export')
    source_model = a.source_export / a.partition / 'model.onnx'
    if sha(source_model) != source_meta['graph_sha256']:
        raise ValueError('Source full ONNX SHA256 mismatch')
    model = onnx.load(str(source_model), load_external_data=False)
    capture_path = a.fixtures / 'capture_manifest.json'
    capture = json.loads(capture_path.read_text(encoding='utf-8'))
    if (capture.get('schema') != 'mvsplat_real_context_stage_fixtures_v1'
            or not capture.get('complete') or capture.get('npu_executed') is not False
            or capture.get('reference_device') != 'board ARM CPU'
            or source_manifest['weight_sha256'] != capture['weight_sha256']
            or capture['contexts']['A']['context_sha256'] == capture['contexts']['B']['context_sha256']):
        raise ValueError('Need unchanged distinct real ARM CPU full-stage captures')
    samples, references = {}, {}
    for label in ('A', 'B'):
        context = capture['contexts'][label]
        stage = context['modules'][a.partition]
        if not stage['complete'] or context['context_views'] != 2 or Path(stage['fixture']).name != stage['fixture']:
            raise ValueError('Invalid original two-view capture metadata')
        path = a.fixtures / label / stage['fixture']
        if sha(path) != stage['fixture_sha256']:
            raise ValueError('Original ARM fixture file changed')
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {'input', 'expected'}:
                raise ValueError('Original ARM fixture fields changed')
            value, expected = data['input'].copy(), data['expected'].copy()
        if (value.dtype != np.float32 or expected.dtype != np.float32
                or not np.isfinite(value).all() or not np.isfinite(expected).all()
                or not np.count_nonzero(value) or array_sha(value) != stage['input_sha256']
                or array_sha(expected) != stage['expected_sha256']
                or list(value.shape) != source_meta['input_shape']
                or list(expected.shape) != source_meta['output_shape']):
            raise ValueError('Original ARM FP32 tensor identity changed')
        samples[label] = value, expected
        references[label] = dict(context_sha256=context['context_sha256'],
            fixture=str(path.resolve()), fixture_sha256=sha(path),
            input_sha256=array_sha(value), original_arm_full_expected_sha256=array_sha(expected))
    if np.array_equal(samples['A'][0], samples['B'][0]) or compare(samples['A'][1], samples['B'][1])['passed']:
        raise ValueError('Need distinguishable real A/B inputs and original ARM outputs')
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    augmented = onnx.ModelProto()
    augmented.CopyFrom(model)
    endpoints = []
    for label, op_type, node_name in PREFIXES:
        nodes = [node for node in model.graph.node if node.name == node_name and node.op_type == op_type]
        if len(nodes) != 1:
            raise ValueError('Source topology differs from the three explicit stem boundaries')
        endpoint = nodes[0].output[0]
        endpoints.append(endpoint)
        augmented.graph.output.append(onnx.helper.make_tensor_value_info(endpoint, onnx.TensorProto.FLOAT, None))
    observer = ort.InferenceSession(augmented.SerializeToString(), sess_options=options,
                                    providers=['CPUExecutionProvider'])
    observed, full_checks = {}, {}
    for label, (value, expected) in samples.items():
        outputs = observer.run(None, {observer.get_inputs()[0].name: value})
        raw = outputs[0]
        if source_meta['onnx_output_layout'] == 'NHWC':
            raw = np.ascontiguousarray(raw.transpose(0, 3, 1, 2))
        full_checks[label] = compare(raw, expected)
        if not full_checks[label]['passed']:
            raise ValueError('Original full ONNX does not match original ARM reference for ' + label)
        observed[label] = outputs[1:]
    del observer
    a.out.mkdir(parents=True)
    records = {}
    for index, (label, op_type, node_name) in enumerate(PREFIXES):
        # ICraft constrains jr_path + net_name length. Keep diagnostic keys short
        # while recording the complete original module/node identity below.
        suffix = ('conv', 'gn', 'gelu')[index]
        key = a.partition.split('_')[0] + '_p_' + suffix
        folder = a.out / key
        path = folder / 'model.onnx'
        shape = list(observed['A'][index].shape)
        if list(observed['B'][index].shape) != shape:
            raise ValueError('A/B intermediate shapes differ')
        record = extract_prefix(model, path, op_type=op_type, node_name=node_name, output_shape=shape)
        checks = {}
        session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        for sample_label, (value, _) in samples.items():
            expected = np.ascontiguousarray(observed[sample_label][index])
            if expected.dtype != np.float32 or not np.isfinite(expected).all():
                raise ValueError('Source ONNX intermediate is nonfinite or not FP32')
            actual = np.ascontiguousarray(session.run(None, {record['input_name']: value})[0].transpose(0, 3, 1, 2))
            checks[sample_label] = compare(actual, expected)
            if not checks[sample_label]['passed']:
                raise ValueError('Extracted prefix differs from original full ONNX intermediate')
            oracle = folder / ('oracle.npz' if sample_label == 'A' else 'oracle_b.npz')
            np.savez_compressed(oracle, input=value, expected=expected)
            record['oracle_sha256' if sample_label == 'A' else 'oracle_b_sha256'] = sha(oracle)
            references[sample_label].setdefault('derived_intermediates', {})[key] = dict(
                sha256=array_sha(expected), source_output_tensor=endpoints[index],
                provenance='original full source ONNX FP32 output; not original ARM intermediate hook')
        del session
        record.update(complete=True, host_onnx_passed=True, host_onnx_checks=checks,
            module=a.partition, kind='onnx_prefix', prefix_label=label,
            input_shape=source_meta['input_shape'], output_shape=shape,
            logical_layout='NCHW', onnx_output_layout='NHWC', dtype='float32',
            onnx_output_shape=[shape[0], shape[2], shape[3], shape[1]],
            returns_list=False, full_view_batch_preserved=True,
            diagnostic_only=True, production_qualified=False, npu_executed=False,
            intermediate_oracle_provenance='derived full source ONNX; not original ARM intermediate hook',
            ops=dict(Counter(n.op_type for n in onnx.load(str(path)).graph.node)),
            path=path.relative_to(a.out).as_posix(), model_bytes=path.stat().st_size,
            status='source_onnx_prefix_host_verified_not_compiled')
        records[key] = record
    manifest = dict(schema='mvsplat_partitions_v1', complete=True,
        diagnostic_only=True, production_qualified=False, npu_executed=False,
        intermediate_oracle_provenance='derived from original full source ONNX on CPU; not independent ARM intermediate capture',
        original_arm_full_stage_fixtures='unchanged and checked against complete source graph; not prefix intermediate oracle',
        original_arm_fixture_manifest_sha256=sha(capture_path), original_arm_references=references,
        original_arm_full_graph_checks=full_checks,
        partition=a.partition, source_export=str(a.source_export.resolve()),
        source_manifest_sha256=sha(source_manifest_path), source_model_sha256=sha(source_model),
        input_shape=source_manifest['input_shape'],
        weight_sha256=source_manifest['weight_sha256'], partitions=records)
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(complete=True, partition=a.partition,
        prefixes={name: dict(input_shape=r['input_shape'], output_shape=r['raw_output_shape'],
            nodes=r['prefix_node_count'], initializers=r['retained_initializer_count'],
            bytes=r['model_bytes'], sha256=r['graph_sha256']) for name, r in records.items()}),
        allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
