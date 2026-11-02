"""Assemble audited local-head tiles; qualification remains a separate board step.

Only execution batch/H/W may change. All trained weights, operations and layer
padding must match the original graph byte for byte after this shape rewrite.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np

from common import new_directory, save, sha
from gaussian_generation.npu_branch.artifacts import relative_file, validate_bundle
from gaussian_generation.npu_branch.compile_graphs import audit
from gaussian_generation.npu_branch.spatial import execute_spatial, validate_tiling


def assemble(graphs, compiled, head_graphs, head_compiled, names, out, core=32):
    import onnx
    from gaussian_generation.npu_branch.head_probe import _head, _session

    graphs, compiled = Path(graphs), Path(compiled)
    gm, _ = validate_bundle(graphs, names, 'onnx_reference')
    cm, _ = validate_bundle(compiled, names, 'npu')
    if cm['source_manifest_sha256'] != sha(graphs/'manifest.json'):
        raise ValueError('Base graph and compiled bundles differ')
    if len(head_graphs) != len(head_compiled):
        raise ValueError('Each tiled export needs its own compiled bundle')
    candidates = {}
    for graph_root, compiled_root in zip(head_graphs, head_compiled):
        graph_root, compiled_root = Path(graph_root), Path(compiled_root)
        hg = json.loads((graph_root/'manifest.json').read_text())
        hc = json.loads((compiled_root/'manifest.json').read_text())
        name = hg.get('source_partition')
        if (hg.get('schema') != 'mvsplat_head_probe_v1' or
                name not in ('to_gaussians', 'to_disparity') or name not in names or
                name in candidates or hg['weight_sha256'] != gm['weight_sha256'] or
                hc['source_manifest_sha256'] != sha(graph_root/'manifest.json')):
            raise ValueError('Unbound, duplicate or unselected head tile')
        original = gm['partitions'][name]
        if (hg['source_graph_sha256'] != original['graph_sha256'] or
                hg['source_oracle_sha256'] != original['oracle_sha256']):
            raise ValueError('Head tile has a different original graph/oracle')
        choices = [k for k, v in hg['partitions'].items() if v['input_shape'][0] == 1
                   and k in hc['partitions'] and hc['partitions'][k]['complete']]
        if len(choices) != 1:
            raise ValueError('Expected exactly one compiled single-view tile per head')
        case = choices[0]
        exported, built = hg['partitions'][case], copy.deepcopy(hc['partitions'][case])
        model_path = relative_file(graph_root, case+'/model.onnx')
        if sha(model_path) != exported['graph_sha256'] or built['export'] != exported:
            raise ValueError('Compiled head does not bind the supplied ONNX export')
        source = onnx.load(str(relative_file(graphs, name+'/model.onnx')))
        nodes, *_ = _head(source)
        for node in (nodes[0], nodes[-1]):
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            if any(attrs.get(k) != v for k, v in (('kernel_shape',[3,3]), ('strides',[1,1]),
                    ('dilations',[1,1]), ('pads',[1,1,1,1]))):
                raise ValueError('The two-pixel halo does not cover this head')
        expected = copy.deepcopy(source)
        for boundary, shape in zip((*expected.graph.input, *expected.graph.output),
                                   (exported['input_shape'], exported['output_shape'])):
            for axis, dimension in enumerate(shape):
                boundary.type.tensor_type.shape.dim[axis].dim_value = dimension
        del expected.graph.value_info[:]
        tiled = onnx.load(str(model_path))
        if expected.SerializeToString() != tiled.SerializeToString():
            raise ValueError('Tile changed more than execution batch/H/W')
        meta = copy.deepcopy(original)
        meta.update(input_shape=exported['input_shape'], output_shape=exported['output_shape'],
            graph_sha256=sha(model_path), source_full_graph_sha256=original['graph_sha256'],
            spatial_tiling=dict(schema='mvsplat_local_head_tiles_v1', halo=2, core_size=core,
                                input_shape=original['input_shape'], output_shape=original['output_shape']))
        validate_tiling(meta)
        oracle_path = relative_file(graphs, name+'/oracle.npz')
        if sha(oracle_path) != original['oracle_sha256']:
            raise ValueError('Original full-image oracle changed')
        session = _session(tiled)
        with np.load(oracle_path, allow_pickle=False) as sample:
            actual = execute_spatial(sample['input'], meta,
                lambda x: session.run(None, {'features': x})[0])
            delta = actual-sample['expected']
            if not np.allclose(actual, sample['expected'], rtol=2e-3, atol=2e-4):
                raise ValueError('Full-image tile assembly changed the source oracle')
            meta['spatial_host_check'] = dict(passed=True, max_abs=float(np.abs(delta).max()),
                rmse=float(np.sqrt(np.mean(delta**2))), scope='full image, both views, all seams and boundaries')
        candidates[name] = (graph_root, compiled_root, case, meta, built)

    out = new_directory(Path(out))
    gm, cm = copy.deepcopy(gm), copy.deepcopy(cm)
    gm['partitions'] = {n:gm['partitions'][n] for n in names}
    cm['partitions'] = {n:cm['partitions'][n] for n in names}
    for name in names:
        gf, cf = out/'graphs'/name, out/'compiled'/name
        gf.mkdir(parents=True)
        cf.mkdir(parents=True)
        shutil.copy2(relative_file(graphs, name+'/oracle.npz'), gf/'oracle.npz')
        if name in candidates:
            graph_root, compiled_root, case, meta, built = candidates[name]
            shutil.copy2(relative_file(graph_root, case+'/model.onnx'), gf/'model.onnx')
            for kind in ('graph','raw'):
                src = relative_file(compiled_root, built[kind])
                target = cf/src.name
                shutil.copy2(src, target)
                built[kind] = target.relative_to(out/'compiled').as_posix()
            gm['partitions'][name] = meta
            built['export'] = meta
            built['audit'] = audit(out/'compiled'/built['graph'], out/'compiled'/built['raw'], meta)
            built['status'] = 'spatial_candidate_requires_board_qualification'
            cm['partitions'][name] = built
        else:
            shutil.copy2(relative_file(graphs, name+'/model.onnx'), gf/'model.onnx')
            for kind in ('graph','raw'):
                src = relative_file(compiled, cm['partitions'][name][kind])
                target = cf/src.name
                shutil.copy2(src, target)
                cm['partitions'][name][kind] = target.relative_to(out/'compiled').as_posix()
    save(out/'graphs/manifest.json', gm)
    cm['source_manifest_sha256'] = sha(out/'graphs/manifest.json')
    save(out/'compiled/manifest.json', cm)
    validate_bundle(out/'graphs', names, 'onnx_reference')
    validate_bundle(out/'compiled', names, 'npu')
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graphs', type=Path, required=True)
    p.add_argument('--compiled', type=Path, required=True)
    p.add_argument('--head-graphs', nargs='+', type=Path, required=True)
    p.add_argument('--head-compiled', nargs='+', type=Path, required=True)
    p.add_argument('--partitions', nargs='+', required=True)
    p.add_argument('--core-size', type=int, default=32)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    print(assemble(a.graphs, a.compiled, a.head_graphs, a.head_compiled,
                   a.partitions, a.out, a.core_size))


if __name__ == '__main__':
    main()
