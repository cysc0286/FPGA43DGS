"""Fuse the two unchanged ONNX heads using captured CPU fixtures.

No model reload or full forward is performed here. The output bundle remains a
candidate: compilation, fresh-input checks and board image qualification must
be performed separately. Every saved oracle is the complete two-view image.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from common import FULL_WEIGHT_SHA256, new_directory, save, sha
from gaussian_generation.npu_branch.head_probe import _head, _session
from gaussian_generation.npu_branch.spatial import execute_spatial, validate_tiling


def fused_model(gaussian, disparity, *, batch=1, size=36, refine_channels=32):
    """Compose graph branches without editing trained parameters or padding."""
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    gauss_nodes, _, inputs, _, outputs = _head(gaussian)
    disp_nodes, _, disp_inputs, _, disp_outputs = _head(disparity)
    if disp_inputs != refine_channels or refine_channels > inputs:
        raise ValueError('Disparity input is not the Gaussian input prefix')
    if ([v.SerializeToString() for v in gaussian.opset_import] !=
            [v.SerializeToString() for v in disparity.opset_import]):
        raise ValueError('Head ONNX operator sets differ')
    for nodes in (gauss_nodes, disp_nodes):
        for node in (nodes[0], nodes[-1]):
            attrs = {a.name:helper.get_attribute_value(a) for a in node.attribute}
            if any(attrs.get(key) != value for key,value in (
                    ('kernel_shape',[3,3]), ('strides',[1,1]),
                    ('dilations',[1,1]), ('pads',[1,1,1,1]))):
                raise ValueError('Two-pixel halo requires unchanged stride-one 3x3 heads')
    if batch < 1 or size < 5:
        raise ValueError('Invalid fused execution shape')

    def branch(source, prefix, alias):
        model = copy.deepcopy(source)
        input_name = model.graph.input[0].name
        def name(value):
            return alias if value == input_name else prefix+value if value else ''
        for node in model.graph.node:
            node.name = prefix+node.name
            for index,value in enumerate(node.input): node.input[index] = name(value)
            for index,value in enumerate(node.output): node.output[index] = name(value)
        for init in model.graph.initializer:
            init.name = name(init.name)
        return list(model.graph.node), list(model.graph.initializer), name(model.graph.output[0].name)

    gn, gi, gout = branch(gaussian, 'gaussian/', 'features')
    dn, di, dout = branch(disparity, 'disparity/', 'disparity/features')
    values = [numpy_helper.from_array(np.asarray(value,np.int64), 'prefix_'+key)
              for key,value in (('starts',[0]),('ends',[refine_channels]),('axes',[1]),('steps',[1]))]
    nodes = [helper.make_node('Slice', ['features','prefix_starts','prefix_ends',
             'prefix_axes','prefix_steps'], ['disparity/features'], name='refine_prefix')]
    nodes += gn+dn+[helper.make_node('Concat',[gout,dout],['result'],axis=1,name='fused_result')]
    graph=helper.make_graph(nodes,'mvsplat_fused_local_heads',
        [helper.make_tensor_value_info('features',TensorProto.FLOAT,[batch,inputs,size,size])],
        [helper.make_tensor_value_info('result',TensorProto.FLOAT,[batch,outputs+disp_outputs,size,size])],
        gi+di+values)
    model=copy.deepcopy(gaussian)
    model.graph.CopyFrom(graph)
    onnx.checker.check_model(model)
    return model


def export(graphs, fixtures, out, *, tile_size=36, core_size=32, context_sha256=None):
    import onnx
    graphs, fixtures = Path(graphs), Path(fixtures)
    source = json.loads((graphs/'manifest.json').read_text())
    if source['weight_sha256'] != FULL_WEIGHT_SHA256:
        raise ValueError('Unexpected source checkpoint')
    full = {}
    for name in ('to_gaussians','to_disparity'):
        original=source['partitions'][name]
        if not original['complete'] or sha(graphs/name/'model.onnx') != original['graph_sha256']:
            raise ValueError('Unverified source head graph '+name)
        with np.load(fixtures/(name+'.npz'),allow_pickle=False) as data:
            full[name] = (data['input'].copy(),data['expected'].copy())
    x, gaussian = full['to_gaussians']
    refine, disparity = full['to_disparity']
    if (x.ndim != 4 or gaussian.ndim != 4 or refine.ndim != 4 or disparity.ndim != 4 or
            x.dtype != np.float32 or gaussian.dtype != np.float32 or
            refine.dtype != np.float32 or disparity.dtype != np.float32 or
            x.shape[1] != 163 or gaussian.shape[1] != 84 or refine.shape[1] != 32 or
            disparity.shape[1] != 2 or not np.array_equal(x[:,:32],refine) or
            any(not np.isfinite(v).all() for v in (x,gaussian,refine,disparity))):
        raise ValueError('Fixture prefix, shape, dtype or finite contract differs')
    expected=np.concatenate((gaussian,disparity),axis=1)
    if any(v.shape[:1]+v.shape[2:] != x.shape[:1]+x.shape[2:] for v in (gaussian,refine,disparity)):
        raise ValueError('Fixture batch/spatial dimensions differ')
    if tile_size > min(x.shape[-2:]) or core_size < 1 or core_size > tile_size-4:
        raise ValueError('Invalid core and two-pixel halo')
    model=fused_model(onnx.load(str(graphs/'to_gaussians/model.onnx')),
                      onnx.load(str(graphs/'to_disparity/model.onnx')),
                      batch=1,size=tile_size)
    out=new_directory(Path(out))
    folder=new_directory(out/'fused_heads')
    onnx.save(model,str(folder/'model.onnx'))
    meta=dict(complete=False,path='depth_predictor.to_gaussians+depth_predictor.to_disparity',
        input_shape=[1,163,tile_size,tile_size],output_shape=[1,86,tile_size,tile_size],
        returns_list=False,logical_layout='NCHW',dtype='float32',npu_executed=False,
        spatial_tiling=dict(schema='mvsplat_local_head_tiles_v1',halo=2,core_size=core_size,
            input_shape=list(x.shape),output_shape=list(expected.shape)),
        fused_heads=dict(schema='mvsplat_fused_heads_v1',refine_channels=32,
            gaussian_channels=84,disparity_channels=2,channel_order=['gaussians','disparity']),
        source_graphs={n:source['partitions'][n]['graph_sha256'] for n in full},
        fixtures={n:sha(fixtures/(n+'.npz')) for n in full},
        graph_sha256=sha(folder/'model.onnx'),
        tolerance=dict(rtol=2e-3,atol=2e-4),
        logical_duplicate_input_bytes_saved=int(refine.shape[0]*
            len(range(0,x.shape[2],core_size))*len(range(0,x.shape[3],core_size))*
            32*tile_size*tile_size*4),
        production_enabled=False,requires_new_board_qualification=True)
    if context_sha256 is not None and (len(context_sha256) != 64 or
            any(c not in '0123456789abcdef' for c in context_sha256)):
        raise ValueError('Invalid supplied fixture context checksum')
    manifest=dict(schema='mvsplat_partitions_v1',weight_sha256=source['weight_sha256'],
        input_shape=source['input_shape'],context_sha256=context_sha256,
        source_context_sha256=source.get('context_sha256'),
        source_manifest_sha256=sha(graphs/'manifest.json'),npu_executed=False,
        partitions={'fused_heads':meta})
    try:
        validate_tiling(meta)
        session=_session(model)
        start=time.perf_counter()
        actual=execute_spatial(x,meta,lambda tile:session.run(None,{'features':tile})[0])
        elapsed=time.perf_counter()-start
        delta=actual-expected
        passed=bool(np.allclose(actual,expected,rtol=2e-3,atol=2e-4))
        np.savez(folder/'oracle.npz',input=x,expected=expected)
        meta.update(complete=passed,host_onnx_passed=passed,
            oracle_sha256=sha(folder/'oracle.npz'),host_onnx_seconds=elapsed,
            max_abs_error=float(np.abs(delta).max()),mean_abs_error=float(np.abs(delta).mean()),
            rmse=float(np.sqrt(np.mean(delta**2))),
            host_check_scope='both views, complete full image, all seams and original zero-padded boundaries')
        del session,actual,delta
        if not passed:raise ValueError('Fused full-image ONNX output differs from original heads')
    except BaseException as exc:
        meta['error']=type(exc).__name__+': '+str(exc)
        raise
    finally:
        save(out/'manifest.json',manifest)
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graphs',type=Path,required=True)
    p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tile-size',type=int,default=36)
    p.add_argument('--core-size',type=int,default=32)
    p.add_argument('--context-sha256',help='Actual captured context checksum, not the older source-graph oracle context')
    a=p.parse_args(argv)
    print(json.dumps(export(a.graphs,a.fixtures,a.out,tile_size=a.tile_size,
                           core_size=a.core_size,context_sha256=a.context_sha256),indent=2))


if __name__=='__main__':
    main()
