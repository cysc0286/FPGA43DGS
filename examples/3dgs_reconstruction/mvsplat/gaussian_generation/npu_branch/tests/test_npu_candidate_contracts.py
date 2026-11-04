"""Small migration-contract regression tests; no model or NPU execution."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from common import FULL_WEIGHT_SHA256, sha
from gaussian_generation.npu_branch.artifacts import from_wire, to_wire, validate_bundle
from gaussian_generation.npu_branch.catalog import PARTITIONS
from gaussian_generation.npu_branch.config import BackendConfig
from gaussian_generation.npu_branch.ffn import FFN_PARTITIONS, adapter_spec
from gaussian_generation.npu_branch.worker import runtime_identity, sdk_passes
from gaussian_generation.npu_branch.runtime import PartitionRuntime
from gaussian_generation.runtime import execution_source_hashes


def ffn_meta(name='transformer_ffn_0'):
    incoming, outgoing = [1,6,5,7], [1,3,5,7]
    return dict(complete=True, input_shape=incoming, output_shape=outgoing,
                host_onnx_passed=True, returns_list=False, dtype='float32', logical_layout='NCHW',
                onnx_output_layout='NHWC', onnx_output_shape=[1,5,7,3],
                ffn_adapter=adapter_spec(name,[2,6,5,7],[2,3,5,7],incoming,outgoing))


def reference_bundle(root, items):
    root = Path(root)
    manifest=dict(schema='mvsplat_partitions_v1', weight_sha256=FULL_WEIGHT_SHA256,
                  input_shape=[1,2,3,128,128],partitions={})
    for name,meta in items.items():
        folder=root/name
        folder.mkdir()
        (folder/'model.onnx').write_bytes(b'not-executed-contract-fixture')
        meta=copy.deepcopy(meta)
        meta['graph_sha256']=sha(folder/'model.onnx')
        manifest['partitions'][name]=meta
    (root/'manifest.json').write_text(json.dumps(manifest))


class NpuCandidateContracts(unittest.TestCase):
    def test_exact_six_ffn_catalog_paths(self):
        self.assertEqual(len(FFN_PARTITIONS),6)
        for i in range(6):
            name='transformer_ffn_'+str(i)
            self.assertEqual(FFN_PARTITIONS[name],
                'backbone.transformer.layers.'+str(i)+'.cross_attn_ffn.mlp')
            self.assertEqual(PARTITIONS[name],FFN_PARTITIONS[name])
        self.assertNotIn('transformer_ffn_6',PARTITIONS)

    def test_reference_nhwc_native_output_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            reference_bundle(directory,{'transformer_ffn_0':ffn_meta()})
            _,parts=validate_bundle(directory,['transformer_ffn_0'],'onnx_reference')
            p=parts['transformer_ffn_0']
            self.assertEqual(p['input'],dict(layout='NCHW',shape=[1,6,5,7]))
            self.assertEqual(p['output'],dict(layout='NHWC',shape=[1,5,7,3]))
            logical=np.arange(105,dtype=np.float32).reshape(1,3,5,7)
            # A native ONNX NHWC result is copied unchanged into the wire.
            native=np.ascontiguousarray(logical.transpose(0,2,3,1))
            restored=from_wire(native,p['output'])
            np.testing.assert_array_equal(restored,logical)
            np.testing.assert_array_equal(to_wire(logical,p['output']),native)
            native[:]=-1
            np.testing.assert_array_equal(restored,logical)

    def test_reference_output_swapped_axes_rejected(self):
        m=ffn_meta()
        m['onnx_output_shape']=[1,7,5,3]
        with tempfile.TemporaryDirectory() as directory:
            reference_bundle(directory,{'transformer_ffn_0':m})
            with self.assertRaisesRegex(ValueError,'axes'):
                validate_bundle(directory,['transformer_ffn_0'],'onnx_reference')

    def test_reference_unreviewed_output_layout_rejected(self):
        m=ffn_meta()
        m['onnx_output_layout']='***C'
        with tempfile.TemporaryDirectory() as directory:
            reference_bundle(directory,{'transformer_ffn_0':m})
            with self.assertRaisesRegex(ValueError,'layout'):
                validate_bundle(directory,['transformer_ffn_0'],'onnx_reference')

    def test_reference_wrong_ffn_layer_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            reference_bundle(directory,{'transformer_ffn_0':ffn_meta('transformer_ffn_1')})
            with self.assertRaisesRegex(ValueError,'different layer'):
                validate_bundle(directory,['transformer_ffn_0'],'onnx_reference')

    def test_reference_wrong_ffn_path_rejected(self):
        m=ffn_meta()
        m['ffn_adapter']['path']=FFN_PARTITIONS['transformer_ffn_1']
        with tempfile.TemporaryDirectory() as directory:
            reference_bundle(directory,{'transformer_ffn_0':m})
            with self.assertRaisesRegex(ValueError,'adapter'):
                validate_bundle(directory,['transformer_ffn_0'],'onnx_reference')

    def test_unknown_and_duplicate_ffn_selection_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            reference_bundle(directory,{'transformer_ffn_0':ffn_meta()})
            for names in (['transformer_ffn_0','transformer_ffn_0'],['transformer_ffn_6']):
                with self.assertRaisesRegex(ValueError,'unique, known'):
                    validate_bundle(directory,names,'onnx_reference')

    def test_public_config_rejects_fused_and_either_original_head(self):
        for head in ('to_gaussians','to_disparity'):
            with self.assertRaisesRegex(ValueError,'mutually exclusive'):
                BackendConfig(backend='npu',partition_bundle=Path('compiled'),
                    npu_library=Path('bridge.so'),partitions=('fused_heads',head))
        # FFNs and the fused pair represent different computation and may coexist.
        good=BackendConfig(backend='npu',partition_bundle=Path('compiled'),
            npu_library=Path('bridge.so'),partitions=('backbone_cnn','transformer_ffn_0','fused_heads'))
        self.assertEqual(good.partitions[-1],'fused_heads')

    def test_candidate_sdk_double_pass_is_charged_without_env_opt_in(self):
        with patch.dict(os.environ,{},clear=True):
            for name in FFN_PARTITIONS:
                self.assertEqual(sdk_passes(name,ffn_meta(name)),2)
            for name in ('to_gaussians','to_disparity','fused_heads'):
                self.assertEqual(sdk_passes(name,{'spatial_tiling':{'halo':2}}),2)
            self.assertEqual(sdk_passes('depth_head',{}),1)
            self.assertEqual(sdk_passes('backbone_cnn',{}),1)
        with patch.dict(os.environ,{'MGS_NPU_SETTLE_BACKBONE':'1'},clear=True):
            self.assertEqual(sdk_passes('backbone_cnn',{}),2)

    def test_full_refinement_sdk_double_pass_keeps_depth_head_single_pass(self):
        # Full U-Nets need their own input-switch protection. They have no
        # tiling or FFN metadata, so accounting must bind their exact names.
        for environment in ({}, {'MGS_NPU_SETTLE_BACKBONE':'1'}):
            with self.subTest(environment=environment),patch.dict(os.environ,environment,clear=True):
                for name in ('corr_refine_net','refine_unet'):
                    self.assertEqual(sdk_passes(name,{}),2)
                self.assertEqual(sdk_passes('depth_head',{}),1)
                self.assertEqual(sdk_passes('corr_refine_net_prefix',{}),1)

    def test_candidate_npu_rejects_explicitly_disabled_fresh_inputs_before_allocation(self):
        for name, candidate in (('transformer_ffn_0', {'ffn_adapter':{'schema':'candidate'}}),
                                ('to_gaussians', {'spatial_tiling':{'halo':2}})):
            with patch('gaussian_generation.npu_branch.runtime.validate_bundle',
                       return_value=({}, {name:dict(meta=candidate)})):
                with self.assertRaisesRegex(ValueError,'fresh SDK input'):
                    PartitionRuntime('npu','unused',[name],'unused',
                                     environment={'MGS_NPU_FRESH_INPUT':'0'})

    def test_sdk_bundle_validation_does_not_import_torch(self):
        # This is a separate interpreter because the test runner legitimately
        # imports Torch for tiny equivalence tests. The SDK worker must not.
        with tempfile.TemporaryDirectory() as directory:
            reference_bundle(directory,{'transformer_ffn_0':ffn_meta()})
            code=("import sys; from gaussian_generation.npu_branch.artifacts import validate_bundle; "
                  "validate_bundle(sys.argv[1],['transformer_ffn_0'],'onnx_reference'); "
                  "assert 'torch' not in sys.modules, 'SDK bundle validation imported Torch'")
            result=subprocess.run([sys.executable,'-c',code,directory],
                cwd=Path(__file__).resolve().parents[3],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_runtime_identity_binds_all_candidate_switches(self):
        environment=dict(MGS_NPU_DDR_GUARD_MIB='64',MGS_NPU_FRESH_INPUT='1',
            MGS_NPU_SETTLE_BACKBONE='1',MGS_NPU_REAPPLY='1',MGS_NPU_FORCE_WAIT='1',
            MGS_NPU_DIRECT_SFB='0')
        with patch.dict(os.environ,environment,clear=True),patch.object(Path,'read_text',return_value=''):
            identity=runtime_identity(['fused_heads'],list(FFN_PARTITIONS))
        self.assertEqual(identity['ddr_guard_mib'],64)
        self.assertEqual(identity['ffn_settle_passes'],2)
        self.assertEqual(identity['ffn_partitions'],list(FFN_PARTITIONS))
        self.assertEqual(identity['spatial_head_settle_passes'],2)
        self.assertEqual(identity['spatial_heads'],['fused_heads'])
        for field in ('fresh_input_tensor','session_reapply','force_output_wait'):
            self.assertIs(identity[field],True)
        self.assertEqual(identity['backbone_settle_passes'],2)
        self.assertNotIn('direct_sfb_output',identity)

    def test_runtime_identity_binds_exact_refinement_partition_selection(self):
        environment={'MGS_NPU_DDR_GUARD_MIB':'64','MGS_NPU_FRESH_INPUT':'1'}
        selections=(['corr_refine_net','refine_unet'],['corr_refine_net'],['refine_unet'],[])
        for names in selections:
            with self.subTest(names=names),patch.dict(os.environ,environment,clear=True),patch.object(Path,'read_text',return_value=''):
                identity=runtime_identity(['fused_heads'],['transformer_ffn_0'],names)
            self.assertEqual(identity['ffn_partitions'],['transformer_ffn_0'])
            self.assertEqual(identity['spatial_heads'],['fused_heads'])
            if names:
                self.assertEqual(identity['refinement_partitions'],names)
                self.assertEqual(identity['refinement_settle_passes'],2)
            else:
                self.assertNotIn('refinement_partitions',identity)
                self.assertNotIn('refinement_settle_passes',identity)

    def test_disabled_runtime_switch_not_reported_enabled(self):
        with patch.dict(os.environ,{'MGS_NPU_DDR_GUARD_MIB':'64'},clear=True),patch.object(Path,'read_text',return_value=''):
            identity=runtime_identity()
        for field in ('fresh_input_tensor','session_reapply','force_output_wait','direct_sfb_output',
                      'ffn_settle_passes','backbone_settle_passes','spatial_head_settle_passes'):
            self.assertNotIn(field,identity)

    def test_execution_identity_includes_new_adapters_and_boundary_logic(self):
        source_root=Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            vendor=Path(directory)
            (vendor/'src').mkdir()
            (vendor/'src/sentinel.py').write_bytes(b'original_vendor_code = 1\n')
            identities=execution_source_hashes(vendor)
        self.assertEqual(identities['src/sentinel.py'],
            __import__('hashlib').sha256(b'original_vendor_code = 1\n').hexdigest())
        for name in ('ffn.py','ffn_contract.py','fused_heads.py','catalog.py','artifacts.py','runtime.py','worker.py','spatial.py'):
            relative='gaussian_generation/npu_branch/'+name
            self.assertEqual(identities['runtime/'+relative],sha(source_root/relative))
        self.assertEqual(identities['runtime/gaussian_generation/runtime.py'],sha(source_root/'gaussian_generation/runtime.py'))


if __name__ == '__main__':
    unittest.main()
