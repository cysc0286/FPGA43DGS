"""Compose independently audited partitions without changing their networks.

Only selected graph/oracle/compiled assets are copied. Every source bundle is
hash-bound and re-audited, and the composed bundle requires fresh native image
qualification. This never promotes a failed compilation or an old receipt.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil

from common import new_directory, save, sha
from gaussian_generation.npu_branch.artifacts import relative_file, validate_bundle
from gaussian_generation.npu_branch.compile_graphs import audit
from gaussian_generation.npu_branch.catalog import validate_selection


def _oracle_context(manifest, name):
    """A composed root context is a qualification target, not an oracle origin."""
    if 'oracle_source_context_sha256s' in manifest:
        origins = manifest['oracle_source_context_sha256s']
        if not isinstance(origins, dict) or name not in origins:
            raise ValueError('Missing inherited oracle source context: '+name)
        value = origins[name]
    else:
        if manifest.get('oracle_contexts_mixed'):
            raise ValueError('Mixed source manifest lacks per-partition oracle origins')
        value = manifest.get('context_sha256')
    if value is not None and (not isinstance(value, str) or len(value) != 64 or
                             any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('Invalid original oracle context SHA256: '+name)
    return value


def assemble(sources, names, output, context_sha256=None, *, allow_mixed_oracle_contexts=False):
    sources, names = list(sources), validate_selection(names)
    if not sources:
        raise ValueError('Need source bundles and unique selected partitions')
    owners = {}
    manifests = []
    for graphs, compiled in sources:
        graphs, compiled = Path(graphs), Path(compiled)
        gm=json.loads((graphs/'manifest.json').read_text())
        cm=json.loads((compiled/'manifest.json').read_text())
        if cm['source_manifest_sha256'] != sha(graphs/'manifest.json'):
            raise ValueError('Compiled bundle does not bind its exported manifest')
        if (cm['weight_sha256'] != gm['weight_sha256'] or
                cm['input_shape'] != gm['input_shape']):
            raise ValueError('Compiled root checkpoint/input shape differs from exported manifest')
        manifests.append((graphs,compiled,gm,cm))
        for name in names:
            if name in gm['partitions'] and name in cm['partitions']:
                if name in owners:
                    raise ValueError('Ambiguous partition source: '+name)
                validate_bundle(graphs,[name],'onnx_reference')
                validate_bundle(compiled,[name],'npu')
                if cm['partitions'][name]['export'] != gm['partitions'][name]:
                    raise ValueError('Compiled export metadata differs: '+name)
                oracle = relative_file(graphs, name+'/oracle.npz')
                if sha(oracle) != gm['partitions'][name]['oracle_sha256']:
                    raise ValueError('Changed source oracle: '+name)
                if 'alternate_oracle_sha256' in gm['partitions'][name]:
                    alternate = relative_file(graphs, name+'/oracle_alternate.npz')
                    if sha(alternate) != gm['partitions'][name]['alternate_oracle_sha256']:
                        raise ValueError('Changed alternate oracle: '+name)
                owners[name]=(graphs,compiled,gm,cm)
    if set(owners)!=set(names) or len(set(names))!=len(names):
        raise ValueError('Missing or duplicate selected partitions')
    base=manifests[0][2]
    if any(gm['weight_sha256']!=base['weight_sha256'] or gm['input_shape']!=base['input_shape']
           for _,_,gm,_ in manifests):
        raise ValueError('Composed networks differ in checkpoint or root input shape')
    oracle_origins = {name:_oracle_context(owners[name][2],name) for name in names}
    contexts = {value for value in oracle_origins.values() if value is not None}
    if len(contexts) > 1 and not allow_mixed_oracle_contexts:
        raise ValueError('Source oracle contexts differ; an override cannot relabel their provenance')
    if allow_mixed_oracle_contexts and context_sha256 is None:
        raise ValueError('Mixed-oracle assembly requires an explicit qualification target context')
    if contexts and context_sha256 is not None and context_sha256 not in contexts and not allow_mixed_oracle_contexts:
        raise ValueError('Target context differs from declared source oracle context')
    selected_context = context_sha256 if context_sha256 is not None else next(iter(contexts), None)
    if (not isinstance(selected_context,str) or len(selected_context) != 64 or
            any(c not in '0123456789abcdef' for c in selected_context)):
        raise ValueError('An explicit SHA256 qualification context is required')
    output=new_directory(Path(output))
    gm=dict(schema='mvsplat_partitions_v1',weight_sha256=base['weight_sha256'],
            input_shape=base['input_shape'],context_sha256=selected_context,
            qualification_context_sha256=selected_context,
            oracle_contexts_mixed=len(contexts)>1,
            context_binding_scope=('context_sha256 and qualification_context_sha256 identify the target '
                'for new whole-model qualification, not the origin of every partition oracle; '
                'oracle_source_context_sha256s and source_provenance preserve the original sources'),
            oracle_source_context_sha256s=oracle_origins,
            partitions={},npu_executed=False,requires_new_board_qualification=True,
            source_manifest_sha256s=[sha(g/'manifest.json') for g,_,_,_ in manifests],
            source_provenance=[dict(graph_manifest_sha256=sha(g/'manifest.json'),
                compiled_manifest_sha256=sha(c/'manifest.json'),
                context_sha256=source.get('context_sha256'),
                source_files=source.get('source_files'),
                selected_partitions=[name for name in names if owners[name][0] == g],
                oracle_source_context_sha256s={name:oracle_origins[name] for name in names
                                              if owners[name][0] == g},
                inherited_source_provenance=copy.deepcopy(source.get('source_provenance')))
                for g,c,source,_ in manifests])
    cm=dict(schema='mvsplat_compiled_v1',weight_sha256=base['weight_sha256'],
            input_shape=base['input_shape'],partitions={},npu_executed=False,
            requires_new_board_qualification=True)
    for name in names:
        graph_root, compiled_root, source_graphs, source_compiled=owners[name]
        meta=copy.deepcopy(source_graphs['partitions'][name])
        built=copy.deepcopy(source_compiled['partitions'][name])
        gf,cf=output/'graphs'/name,output/'compiled'/name
        gf.mkdir(parents=True);cf.mkdir(parents=True)
        for filename in ('model.onnx','oracle.npz'):
            shutil.copy2(relative_file(graph_root,name+'/'+filename),gf/filename)
            key = 'graph_sha256' if filename == 'model.onnx' else 'oracle_sha256'
            if sha(gf/filename) != meta[key]:
                raise ValueError('Copied source asset differs: '+name+'/'+filename)
        if 'alternate_oracle_sha256' in meta:
            alternate=relative_file(graph_root,name+'/oracle_alternate.npz')
            if sha(alternate)!=meta['alternate_oracle_sha256']:
                raise ValueError('Changed alternate oracle')
            shutil.copy2(alternate,gf/alternate.name)
            if sha(gf/alternate.name) != meta['alternate_oracle_sha256']:
                raise ValueError('Copied alternate oracle differs')
        for kind in ('graph','raw'):
            src=relative_file(compiled_root,built[kind])
            dest=cf/src.name
            shutil.copy2(src,dest)
            built[kind]=dest.relative_to(output/'compiled').as_posix()
        copied_audit=audit(output/'compiled'/built['graph'],output/'compiled'/built['raw'],meta)
        if copied_audit != built['audit']:
            raise ValueError('Copied compiled assets differ from their audited source: '+name)
        built['audit']=copied_audit
        gm['partitions'][name]=meta
        cm['partitions'][name]=built
    save(output/'graphs/manifest.json',gm)
    cm['source_manifest_sha256']=sha(output/'graphs/manifest.json')
    save(output/'compiled/manifest.json',cm)
    validate_bundle(output/'graphs',names,'onnx_reference')
    validate_bundle(output/'compiled',names,'npu')
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',nargs=2,action='append',required=True,metavar=('GRAPHS','COMPILED'))
    p.add_argument('--partitions',nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--context-sha256')
    p.add_argument('--allow-mixed-oracle-contexts',action='store_true',
                   help='Require an explicit qualification target and preserve every oracle source context')
    a=p.parse_args(argv)
    print(assemble(a.source,a.partitions,a.out,a.context_sha256,
                   allow_mixed_oracle_contexts=a.allow_mixed_oracle_contexts))


if __name__=='__main__':
    main()
