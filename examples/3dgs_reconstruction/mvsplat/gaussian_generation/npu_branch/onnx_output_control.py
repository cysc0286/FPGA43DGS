"""Controlled output-tail alternatives for a host-verified NHWC UNet ONNX.

Never relabel an ICraft tensor. Remove only the verified final NCHW->NHWC
Transpose. Optionally execute an actual identity 1x1 Conv to give the compiler
a conventional convolution boundary. Every candidate needs a fresh ARM ORT
check against the unchanged real NCHW oracle before it is marked complete.
"""
import argparse
from collections import Counter
import copy
import gc
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from common import new_directory, save, sha
from gaussian_generation.npu_branch.unet_export import numerical_comparison


def change_output_tail(model, logical_shape, *, identity_conv=False, output_layout='NCHW'):
    """Return a distinct ONNX model with audited, narrowly scoped tail changes."""
    import onnx
    from onnx import helper, numpy_helper
    if output_layout not in ('NCHW','NHWC') or len(logical_shape) != 4:
        raise ValueError('Expected four-dimensional logical NCHW output')
    if any(not isinstance(v,int) or v <= 0 for v in logical_shape):
        raise ValueError('Static positive output dimensions are required')
    if len(model.graph.output) != 1 or not model.graph.node:
        raise ValueError('Expected one ONNX output and nonempty graph')
    tail = model.graph.node[-1]
    old_output = model.graph.output[0]
    permutation = next((list(a.ints) for a in tail.attribute if a.name == 'perm'), None)
    if (tail.op_type != 'Transpose' or tail.domain not in ('','ai.onnx')
            or permutation != [0,2,3,1] or len(tail.input) != 1
            or list(tail.output) != [old_output.name]):
        raise ValueError('Expected the exact terminal NCHW-to-NHWC Transpose')
    expected_nhwc = [logical_shape[i] for i in (0,2,3,1)]
    dtype = old_output.type.tensor_type
    old_shape = [d.dim_value for d in dtype.shape.dim]
    if dtype.elem_type != onnx.TensorProto.FLOAT or old_shape != expected_nhwc:
        raise ValueError('Original explicit NHWC output shape/dtype differs from manifest')
    # ONNX tensors are in framework order. No compiled dtype or layout tag is
    # edited here; the real compiler and existing ABI audit decide placement.
    result = copy.deepcopy(model)
    source = result.graph.node[-1].input[0]
    del result.graph.node[-1]
    del result.graph.output[:]
    added_ops = []
    prefix = '__mgs_output_control_'
    used = {v for node in result.graph.node for v in list(node.input)+list(node.output)}
    used.update(t.name for t in result.graph.initializer)
    if any(name.startswith(prefix) for name in used):
        raise ValueError('Output-control namespace already present')
    if identity_conv:
        channels = logical_shape[1]
        weight = np.eye(channels,dtype=np.float32).reshape(channels,channels,1,1)
        bias = np.zeros(channels,dtype=np.float32)
        result.graph.initializer.extend([
            numpy_helper.from_array(weight,prefix+'weight'),
            numpy_helper.from_array(bias,prefix+'bias')])
        destination = prefix+'nchw'
        result.graph.node.append(helper.make_node('Conv',
            [source,prefix+'weight',prefix+'bias'],[destination],
            name=prefix+'identity_conv',kernel_shape=[1,1],strides=[1,1],pads=[0,0,0,0]))
        source = destination
        added_ops.append('Conv(identity_1x1)')
    output_shape = list(logical_shape)
    if output_layout == 'NHWC':
        destination = prefix+'nhwc'
        result.graph.node.append(helper.make_node('Transpose',[source],[destination],
            name=prefix+'nhwc_transpose',perm=[0,2,3,1]))
        source = destination
        output_shape = expected_nhwc
        added_ops.append('Transpose(0,2,3,1)')
    result.graph.output.append(helper.make_tensor_value_info(source,onnx.TensorProto.FLOAT,output_shape))
    onnx.checker.check_model(result)
    elements = int(np.prod(logical_shape))
    return result, dict(removed_op='terminal Transpose(0,2,3,1)',added_ops=added_ops,
        identity_conv=identity_conv,identity_conv_multiplications=elements*logical_shape[1] if identity_conv else 0,
        logical_output_shape=list(logical_shape),onnx_output_shape=output_shape,
        onnx_output_layout=output_layout,compiled_layout_modified=False)


def prepare(graphs, partition, out, *, identity_conv=False, output_layout='NCHW'):
    import onnx
    source_manifest = Path(graphs)/'manifest.json'
    manifest = json.loads(source_manifest.read_text())
    source = manifest['partitions'][partition]
    if (not source.get('complete') or not source.get('host_onnx_passed')
            or source.get('onnx_output_layout') != 'NHWC'):
        raise ValueError('Source must be a complete host-verified explicit NHWC export')
    graph = Path(graphs)/partition/'model.onnx'
    oracle = Path(graphs)/partition/'oracle.npz'
    if sha(graph) != source['graph_sha256'] or sha(oracle) != source['oracle_sha256']:
        raise ValueError('Source graph/oracle hash mismatch')
    model = onnx.load(str(graph))
    candidate, change = change_output_tail(model,source['output_shape'],
        identity_conv=identity_conv,output_layout=output_layout)
    del model
    out = new_directory(Path(out))
    folder = new_directory(out/partition)
    onnx.save(candidate,str(folder/'model.onnx'))
    ops = dict(Counter((node.domain+':' if node.domain else '')+node.op_type for node in candidate.graph.node))
    del candidate
    shutil.copyfile(oracle,folder/'oracle.npz')
    item = copy.deepcopy(source)
    item.update(complete=False,npu_executed=False,host_onnx_passed=False,host_onnx_check=None,
        source_graph_sha256=source['graph_sha256'],source_manifest_sha256=sha(source_manifest),
        graph_sha256=sha(folder/'model.onnx'),oracle_sha256=sha(folder/'oracle.npz'),
        onnx_output_layout=output_layout,onnx_output_shape=change['onnx_output_shape'],
        onnx_reference_worker_compatible=output_layout == 'NCHW',
        output_tail_control=change,ops=ops,status='prepared_pending_arm_host_verification')
    if identity_conv and isinstance(item.get('parameters'),int):
        channels = source['output_shape'][1]
        item['parameters'] += channels*channels+channels
    manifest['partitions'] = {partition:item}
    manifest.update(diagnostic_only=True,npu_executed=False,
        output_control_exporter_sha256=sha(Path(__file__)))
    save(out/'manifest.json',manifest)
    save(folder/'export_result.json',item)
    gc.collect()
    return manifest


def verify(graphs, partition):
    """Use ONNX Runtime only: no Torch import or original model allocation."""
    import onnxruntime as ort
    graphs = Path(graphs)
    manifest = json.loads((graphs/'manifest.json').read_text())
    item = manifest['partitions'][partition]
    folder = graphs/partition
    if sha(folder/'model.onnx') != item['graph_sha256'] or sha(folder/'oracle.npz') != item['oracle_sha256']:
        raise ValueError('Candidate graph/oracle hash mismatch')
    item.update(complete=False,host_onnx_passed=False)
    save(graphs/'manifest.json',manifest)
    try:
        with np.load(folder/'oracle.npz',allow_pickle=False) as oracle:
            x, expected = oracle['input'].copy(), oracle['expected'].copy()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        started = time.perf_counter()
        session = ort.InferenceSession(str(folder/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        item['host_session_load_seconds'] = time.perf_counter()-started
        started = time.perf_counter()
        wire = session.run(None,{'features':x})[0]
        item['host_onnx_seconds'] = time.perf_counter()-started
        del session
        if list(wire.shape) != item['onnx_output_shape']:
            raise ValueError('Actual ONNX output shape differs from candidate contract')
        actual = wire.transpose(0,3,1,2) if item['onnx_output_layout'] == 'NHWC' else wire
        check = numerical_comparison(actual,expected,rtol=2e-3,atol=2e-4)
        item.update(complete=check['passed'],host_onnx_passed=check['passed'],host_onnx_check=check,
            max_abs_error=check.get('max_abs_error'),mean_abs_error=check.get('mean_abs_error'),
            status='exported_host_verified_not_compiled' if check['passed'] else 'host_onnx_failed',
            verification_scope='ONNX CPU oracle only; not ICraft/NPU execution',
            torch_loaded='torch' in sys.modules,torch_required=False)
        if not check['passed']:
            raise ValueError('Controlled-tail ONNX differs from unchanged real oracle')
    except Exception as exc:
        item.update(complete=False,status='host_verification_failed',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        gc.collect()
        save(folder/'export_result.json',item)
        save(graphs/'manifest.json',manifest)
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','verify','both'))
    p.add_argument('--graphs',type=Path,required=True)
    p.add_argument('--partition',required=True)
    p.add_argument('--out',type=Path)
    p.add_argument('--identity-conv-tail',action='store_true')
    p.add_argument('--output-layout',choices=('NCHW','NHWC'),default='NCHW')
    a = p.parse_args(argv)
    if a.action != 'verify':
        if a.out is None:
            p.error('prepare/both requires --out')
        prepare(a.graphs,a.partition,a.out,identity_conv=a.identity_conv_tail,output_layout=a.output_layout)
    if a.action in ('verify','both'):
        r = verify(a.graphs if a.action == 'verify' else a.out,a.partition)
        print(json.dumps(dict(partition=a.partition,**r['partitions'][a.partition])),flush=True)
    else:
        print(json.dumps(dict(prepared=str(a.out),complete=False,needs_host_verification=True)),flush=True)


if __name__=='__main__':
    main()
