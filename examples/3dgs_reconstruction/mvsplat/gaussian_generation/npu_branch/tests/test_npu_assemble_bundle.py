"""Toy, genuinely audited bundle composition; no model/compiler/NPU execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from common import FULL_WEIGHT_SHA256, sha
from gaussian_generation.npu_branch.assemble_bundle import assemble
from gaussian_generation.npu_branch.compile_graphs import audit


CONTEXT = 'a'*64


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value,sort_keys=True),encoding='utf-8')


def source(root, names, context=CONTEXT, source_files=None):
    root=Path(root)
    graphs,compiled=root/'graphs',root/'compiled'
    graphs.mkdir(parents=True)
    compiled.mkdir()
    gm=dict(schema='mvsplat_partitions_v1',weight_sha256=FULL_WEIGHT_SHA256,
            input_shape=[1,2,3,128,128],partitions={})
    if context is not None:gm['context_sha256']=context
    if source_files is not None:gm['source_files']=source_files
    cm=dict(schema='mvsplat_compiled_v1',weight_sha256=FULL_WEIGHT_SHA256,
            input_shape=[1,2,3,128,128],partitions={})
    for name in names:
        g,c=graphs/name,compiled/name
        g.mkdir();c.mkdir()
        (g/'model.onnx').write_bytes(b'contract-fixture-'+name.encode())
        sample=np.arange(12,dtype=np.float32).reshape(1,2,2,3)
        np.savez(g/'oracle.npz',input=sample,expected=sample)
        np.savez(g/'oracle_alternate.npz',input=-sample,expected=-sample)
        meta=dict(complete=True,host_onnx_passed=True,input_shape=[1,2,2,3],
                  output_shape=[1,2,2,3],returns_list=False,dtype='float32',logical_layout='NCHW',
                  graph_sha256=sha(g/'model.onnx'),oracle_sha256=sha(g/'oracle.npz'),
                  alternate_oracle_sha256=sha(g/'oracle_alternate.npz'))
        gm['partitions'][name]=meta
        dtype=dict(element_dtype='@fp(32)',layout='@layout(NCHW)',shape=[1,2,2,3])
        net=dict(params_bytes=4,icraft_version='toy-test',ops=[
            dict(_type_key='icraft::xir::Input',outputs=[dict(dtype=dtype)]),
            dict(_type_key='icraft::xir::HardOp',compile_target='@zhuget(330)',name='Toy.Conv2dNode'),
            dict(_type_key='icraft::xir::Output',inputs=[dict(dtype=dtype)])])
        write(c/'network.json',net)
        (c/'parameters.raw').write_bytes(b'\0'*4)
        cm['partitions'][name]=dict(complete=True,status='compiled_not_executed',export=copy.deepcopy(meta),
            graph=name+'/network.json',raw=name+'/parameters.raw',
            audit=audit(c/'network.json',c/'parameters.raw',meta))
    write(graphs/'manifest.json',gm)
    cm['source_manifest_sha256']=sha(graphs/'manifest.json')
    write(compiled/'manifest.json',cm)
    return graphs,compiled


def mutate_manifest(pair, mutate):
    g,c=pair
    gm,cm=read(g/'manifest.json'),read(c/'manifest.json')
    mutate(gm,cm)
    write(g/'manifest.json',gm)
    cm['source_manifest_sha256']=sha(g/'manifest.json')
    write(c/'manifest.json',cm)


class AssembleBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_selected_partitions_only_and_all_source_identity_retained(self):
        source_files={'src/encoder.py':'b'*64,'runtime/adapter.py':'c'*64}
        first=source(self.root/'one',['backbone_cnn','upsampler'],source_files=source_files)
        second=source(self.root/'two',['depth_head'],source_files=source_files)
        original={str(p):p.read_bytes() for pair in (first,second) for d in pair for p in d.rglob('*') if p.is_file()}
        out=assemble([first,second],['backbone_cnn','depth_head'],self.root/'out')
        gm,cm=read(out/'graphs/manifest.json'),read(out/'compiled/manifest.json')
        self.assertEqual(set(gm['partitions']),{'backbone_cnn','depth_head'})
        self.assertFalse((out/'graphs/upsampler').exists())
        self.assertEqual(cm['source_manifest_sha256'],sha(out/'graphs/manifest.json'))
        self.assertEqual(gm['context_sha256'],CONTEXT)
        self.assertEqual(gm['qualification_context_sha256'],CONTEXT)
        self.assertFalse(gm['oracle_contexts_mixed'])
        self.assertTrue(gm['requires_new_board_qualification'])
        self.assertTrue(cm['requires_new_board_qualification'])
        self.assertFalse(gm['npu_executed'])
        self.assertEqual(gm['source_manifest_sha256s'],[sha(p[0]/'manifest.json') for p in (first,second)])
        for record,pair,name in zip(gm['source_provenance'],(first,second),('backbone_cnn','depth_head')):
            self.assertEqual(record['graph_manifest_sha256'],sha(pair[0]/'manifest.json'))
            self.assertEqual(record['compiled_manifest_sha256'],sha(pair[1]/'manifest.json'))
            self.assertEqual(record['context_sha256'],CONTEXT)
            self.assertEqual(record['source_files'],source_files)
            self.assertEqual(record['selected_partitions'],[name])
        for name,meta in gm['partitions'].items():
            self.assertEqual(sha(out/'graphs'/name/'oracle.npz'),meta['oracle_sha256'])
            self.assertEqual(sha(out/'graphs'/name/'oracle_alternate.npz'),meta['alternate_oracle_sha256'])
            self.assertEqual(cm['partitions'][name]['export'],meta)
            self.assertEqual(cm['partitions'][name]['audit'],audit(
                out/'compiled'/cm['partitions'][name]['graph'],out/'compiled'/cm['partitions'][name]['raw'],meta))
        for path,data in original.items():self.assertEqual(Path(path).read_bytes(),data)

    def test_compile_manifest_must_bind_exact_source_manifest(self):
        pair=source(self.root/'one',['backbone_cnn'])
        gm=read(pair[0]/'manifest.json');gm['new_uncompiled_note']='changed';write(pair[0]/'manifest.json',gm)
        with self.assertRaisesRegex(ValueError,'bind'):
            assemble([pair],['backbone_cnn'],self.root/'out')
        self.assertFalse((self.root/'out').exists())

    def test_failed_compilation_never_promoted(self):
        pair=source(self.root/'one',['backbone_cnn'])
        mutate_manifest(pair,lambda g,c:c['partitions']['backbone_cnn'].update(complete=False))
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            assemble([pair],['backbone_cnn'],self.root/'out')

    def test_compiled_export_metadata_cannot_differ(self):
        pair=source(self.root/'one',['backbone_cnn'])
        mutate_manifest(pair,lambda g,c:c['partitions']['backbone_cnn']['export'].update(extra='unbound'))
        with self.assertRaisesRegex(ValueError,'metadata differs'):
            assemble([pair],['backbone_cnn'],self.root/'out')

    def test_changed_compiled_graph_or_raw_rejected(self):
        for kind in ('graph','raw'):
            with self.subTest(kind=kind):
                pair=source(self.root/kind,['backbone_cnn'])
                folder=pair[1]/'backbone_cnn'
                if kind=='raw':(folder/'parameters.raw').write_bytes(b'12345')
                else:
                    graph=read(folder/'network.json')
                    graph['ops'][1]['compile_target']='@host'
                    write(folder/'network.json',graph)
                with self.assertRaises(ValueError):
                    assemble([pair],['backbone_cnn'],self.root/(kind+'_out'))

    def test_copy_cannot_silently_replace_audited_parameters(self):
        import shutil
        pair=source(self.root/'one',['backbone_cnn'])
        original_copy=shutil.copy2
        def corrupt_after_copy(src,dest):
            result=original_copy(src,dest)
            if Path(dest).suffix=='.raw':Path(dest).write_bytes(b'1234')
            return result
        with patch('gaussian_generation.npu_branch.assemble_bundle.shutil.copy2',side_effect=corrupt_after_copy):
            with self.assertRaisesRegex(ValueError,'audited source'):
                assemble([pair],['backbone_cnn'],self.root/'out')
        self.assertFalse((self.root/'out/compiled/manifest.json').exists())

    def test_changed_onnx_or_primary_oracle_or_alternate_rejected_before_output(self):
        for filename,pattern in (('model.onnx','ONNX'),('oracle.npz','source oracle'),
                                 ('oracle_alternate.npz','alternate oracle')):
            with self.subTest(filename=filename):
                pair=source(self.root/filename,['backbone_cnn'])
                (pair[0]/'backbone_cnn'/filename).write_bytes(b'mutated')
                dest=self.root/(filename+'_out')
                with self.assertRaisesRegex(ValueError,pattern):
                    assemble([pair],['backbone_cnn'],dest)
                self.assertFalse(dest.exists())

    def test_root_compiled_shape_must_match_export(self):
        pair=source(self.root/'one',['backbone_cnn'])
        mutate_manifest(pair,lambda g,c:c.update(input_shape=[1,2,3,256,256]))
        with self.assertRaisesRegex(ValueError,'root checkpoint/input shape'):
            assemble([pair],['backbone_cnn'],self.root/'out')

    def test_compiled_checkpoint_identity_cannot_change(self):
        pair=source(self.root/'one',['backbone_cnn'])
        mutate_manifest(pair,lambda g,c:c.update(weight_sha256='d'*64))
        with self.assertRaisesRegex(ValueError,'checkpoint/input shape'):
            assemble([pair],['backbone_cnn'],self.root/'out')

    def test_sources_cannot_use_different_root_shapes(self):
        one=source(self.root/'one',['backbone_cnn'])
        two=source(self.root/'two',['depth_head'])
        def change(g,c):g['input_shape']=c['input_shape']=[1,2,3,256,256]
        mutate_manifest(two,change)
        with self.assertRaisesRegex(ValueError,'root input shape'):
            assemble([one,two],['backbone_cnn','depth_head'],self.root/'out')

    def test_source_context_disagreement_cannot_be_hidden_by_override(self):
        one=source(self.root/'one',['backbone_cnn'])
        two=source(self.root/'two',['depth_head'],context='d'*64)
        with self.assertRaisesRegex(ValueError,'contexts differ'):
            assemble([one,two],['backbone_cnn','depth_head'],self.root/'out',CONTEXT)

    def test_mixed_oracle_contexts_need_flag_and_target_and_preserve_each_origin(self):
        one=source(self.root/'one',['backbone_cnn'])
        other_context='d'*64
        two=source(self.root/'two',['depth_head'],context=other_context)
        original=[read(pair[0]/'manifest.json') for pair in (one,two)]
        out=assemble([one,two],['backbone_cnn','depth_head'],self.root/'out',other_context,
                     allow_mixed_oracle_contexts=True)
        gm,cm=read(out/'graphs/manifest.json'),read(out/'compiled/manifest.json')
        self.assertEqual(gm['qualification_context_sha256'],other_context)
        self.assertEqual(gm['context_sha256'],other_context)
        self.assertTrue(gm['oracle_contexts_mixed'])
        self.assertIn('not the origin of every partition oracle',gm['context_binding_scope'])
        self.assertEqual(gm['oracle_source_context_sha256s'],
                         {'backbone_cnn':CONTEXT,'depth_head':other_context})
        self.assertEqual([row['context_sha256'] for row in gm['source_provenance']],
                         [CONTEXT,other_context])
        for pair,previous,name in zip((one,two),original,('backbone_cnn','depth_head')):
            self.assertEqual(read(pair[0]/'manifest.json'),previous)
            self.assertEqual(gm['partitions'][name],previous['partitions'][name])
            self.assertEqual(cm['partitions'][name]['export'],previous['partitions'][name])
            self.assertEqual(sha(out/'graphs'/name/'oracle.npz'),previous['partitions'][name]['oracle_sha256'])

    def test_mixed_oracle_flag_alone_cannot_invent_a_qualification_target(self):
        one=source(self.root/'one',['backbone_cnn'])
        two=source(self.root/'two',['depth_head'],context='d'*64)
        with self.assertRaisesRegex(ValueError,'explicit qualification target'):
            assemble([one,two],['backbone_cnn','depth_head'],self.root/'out',
                     allow_mixed_oracle_contexts=True)
        self.assertFalse((self.root/'out').exists())

    def test_explicit_mixed_mode_can_qualify_new_target_without_relabeling_oracles(self):
        one=source(self.root/'one',['backbone_cnn'])
        target='e'*64
        out=assemble([one],['backbone_cnn'],self.root/'out',target,allow_mixed_oracle_contexts=True)
        gm=read(out/'graphs/manifest.json')
        self.assertEqual(gm['qualification_context_sha256'],target)
        self.assertEqual(gm['oracle_source_context_sha256s']['backbone_cnn'],CONTEXT)
        self.assertEqual(gm['source_provenance'][0]['context_sha256'],CONTEXT)
        self.assertFalse(gm['oracle_contexts_mixed'])

    def test_override_cannot_relabel_declared_oracle_context(self):
        pair=source(self.root/'one',['backbone_cnn'])
        with self.assertRaisesRegex(ValueError,'Target context differs'):
            assemble([pair],['backbone_cnn'],self.root/'out','d'*64)

    def test_nested_mixed_origins_survive_equal_qualification_roots(self):
        one=source(self.root/'one',['backbone_cnn'])
        target='d'*64
        two=source(self.root/'two',['depth_head'],context=target)
        nested=assemble([one,two],['backbone_cnn','depth_head'],self.root/'nested',target,
                        allow_mixed_oracle_contexts=True)
        third=source(self.root/'third',['upsampler'],context=target)
        origins={'backbone_cnn':CONTEXT,'depth_head':target,'upsampler':target}
        out=assemble([(nested/'graphs',nested/'compiled'),third],list(origins),self.root/'out',target,
                     allow_mixed_oracle_contexts=True)
        manifest=read(out/'graphs/manifest.json')
        self.assertTrue(manifest['oracle_contexts_mixed'])
        self.assertEqual(manifest['oracle_source_context_sha256s'],origins)
        self.assertEqual(manifest['context_sha256'],target)
        nested_record=manifest['source_provenance'][0]
        self.assertEqual(nested_record['context_sha256'],target)
        self.assertEqual(nested_record['oracle_source_context_sha256s'],
                         {'backbone_cnn':CONTEXT,'depth_head':target})
        self.assertEqual(nested_record['inherited_source_provenance'],
                         read(nested/'graphs/manifest.json')['source_provenance'])

    def test_nested_mixed_source_requires_flag_even_if_root_target_is_single(self):
        one=source(self.root/'one',['backbone_cnn'])
        two=source(self.root/'two',['depth_head'],context='d'*64)
        nested=assemble([one,two],['backbone_cnn','depth_head'],self.root/'nested','d'*64,
                        allow_mixed_oracle_contexts=True)
        with self.assertRaisesRegex(ValueError,'contexts differ'):
            assemble([(nested/'graphs',nested/'compiled')],['backbone_cnn','depth_head'],self.root/'out')
        self.assertFalse((self.root/'out').exists())

    def test_nested_subset_uses_selected_original_oracle_context(self):
        one=source(self.root/'one',['backbone_cnn'])
        two=source(self.root/'two',['depth_head'],context='d'*64)
        nested=assemble([one,two],['backbone_cnn','depth_head'],self.root/'nested','d'*64,
                        allow_mixed_oracle_contexts=True)
        out=assemble([(nested/'graphs',nested/'compiled')],['backbone_cnn'],self.root/'out')
        manifest=read(out/'graphs/manifest.json')
        self.assertEqual(manifest['context_sha256'],CONTEXT)
        self.assertEqual(manifest['oracle_source_context_sha256s'],{'backbone_cnn':CONTEXT})
        self.assertFalse(manifest['oracle_contexts_mixed'])

    def test_incomplete_or_invalid_inherited_origins_fail_before_output(self):
        changes=[
            lambda g:g.update(oracle_source_context_sha256s=[]),
            lambda g:g.update(oracle_source_context_sha256s={}),
            lambda g:g.update(oracle_source_context_sha256s={'backbone_cnn':'not-a-sha'}),
            lambda g:g.update(oracle_contexts_mixed=True),
        ]
        for index,change in enumerate(changes):
            with self.subTest(index=index):
                pair=source(self.root/str(index),['backbone_cnn'])
                mutate_manifest(pair,lambda g,c:change(g))
                out=self.root/('out'+str(index))
                with self.assertRaisesRegex(ValueError,'oracle.*context|oracle origins'):
                    assemble([pair],['backbone_cnn'],out,CONTEXT,allow_mixed_oracle_contexts=True)
                self.assertFalse(out.exists())

    def test_legacy_missing_context_is_preserved_and_explicitly_bound(self):
        pair=source(self.root/'one',['backbone_cnn'],context=None)
        out=assemble([pair],['backbone_cnn'],self.root/'out',CONTEXT)
        manifest=read(out/'graphs/manifest.json')
        self.assertEqual(manifest['context_sha256'],CONTEXT)
        self.assertIsNone(manifest['source_provenance'][0]['context_sha256'])
        self.assertIsNone(manifest['source_provenance'][0]['source_files'])

    def test_context_hash_required_if_legacy_source_did_not_declare_it(self):
        pair=source(self.root/'one',['backbone_cnn'],context=None)
        for index,value in enumerate((None,'short','z'*64)):
            with self.assertRaisesRegex(ValueError,'SHA256'):
                assemble([pair],['backbone_cnn'],self.root/('out'+str(index)),value)

    def test_missing_ambiguous_duplicate_and_mixed_head_selection_rejected(self):
        one=source(self.root/'one',['backbone_cnn'])
        two=source(self.root/'two',['backbone_cnn'])
        cases=[([one],['depth_head'],'Missing'),([one,two],['backbone_cnn'],'Ambiguous'),
               ([one],['backbone_cnn','backbone_cnn'],'unique'),([],['backbone_cnn'],'source'),
               ([one],['fused_heads','to_gaussians'],'Fused')]
        for index,(sources,names,pattern) in enumerate(cases):
            with self.subTest(names=names):
                with self.assertRaisesRegex(ValueError,pattern):
                    assemble(sources,names,self.root/('out'+str(index)))


if __name__=='__main__':
    unittest.main()
