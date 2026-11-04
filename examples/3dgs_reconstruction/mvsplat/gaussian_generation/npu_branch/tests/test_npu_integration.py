"""Formal entry wiring and acceptance contracts; mocks never attest NPU speed."""
from collections import deque
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from common import FULL_WEIGHT_SHA256, sha
from gaussian_generation.npu_branch.config import BackendConfig, GenerationSession
from gaussian_generation.npu_branch.acceptance import (
    SCHEMA, QUALITY_LIMITS, check_receipt, validate_worker)
from gaussian_generation.npu_branch.readiness import probe_oracles, validate_oracles
from gaussian_generation.npu_branch.runtime import PartitionRuntime


def receipt():
    check=dict(partition='to_gaussians',repeat_exact=True,output_sha256='output',oracle_sha256='oracle')
    return dict(schema=SCHEMA,complete=True,diagnostic_only=False,npu_executed=True,fpga_executed=True,
                gaussian_repeat_exact=True,weight_sha256=FULL_WEIGHT_SHA256,partitions=['to_gaussians'],
                quality_limits=QUALITY_LIMITS,oracle_checks=[check],
                runtime_identity=dict(sdk_libraries={'libicraft.so':'sdk'}),
                source_files={'src/model.py':'model_source_hash'},
                provenance=dict(bundle_sha256='bundle',oracle_manifest_sha256='graphs',bridge_sha256='bridge'),
                views={str(i):dict(npu_vs_cpu=dict(psnr_db=50.,ssim=.999),truth_psnr_drop_db=.01,
                                   truth_ssim_drop=.001) for i in range(3)})


class IntegrationTests(unittest.TestCase):
    def test_device_lifecycle_renderer_first_and_npu_closes_first(self):
        from contextlib import contextmanager
        from initialize.session import WarmSession
        events = []
        renderer = SimpleNamespace(runtime=SimpleNamespace(configuration={}))
        model = SimpleNamespace(load_seconds=0.)
        @contextmanager
        def resource(name, value):
            events.append('open_' + name)
            try:
                yield value
            finally:
                events.append('close_' + name)
        args = SimpleNamespace(backend='npu', threads=4, prepare_threads=1, live_renderer='binary')
        generation = SimpleNamespace(model=model, partitions=None, readiness={})
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(WarmSession, '_open_renderer', side_effect=lambda *a: resource('renderer', renderer)), \
             patch('gaussian_generation.npu_branch.config.GenerationSession', side_effect=lambda *a, **k: resource('npu', generation)):
            with WarmSession(args, Path(directory)):
                self.assertEqual(events, ['open_renderer', 'open_npu'])
            self.assertEqual(events, ['open_renderer', 'open_npu', 'close_npu', 'close_renderer'])
            self.assertEqual(renderer.runtime.pl_ddr_limit_bytes, 64*1024*1024)

    def test_renderer_rejects_dma_workspace_crossing_npu_boundary_before_submit(self):
        from rendering.runtime import LiveRenderer
        runtime = LiveRenderer.__new__(LiveRenderer)
        runtime.scene = dict(selected=300000)
        runtime.configuration = dict(batch=2)
        runtime.pl_ddr_limit_bytes = 64*1024*1024
        runtime._send = MagicMock()
        with self.assertRaisesRegex(ValueError, 'reserved PL DDR'):
            runtime.render_camera(b'no camera needed to exercise boundary')
        runtime._send.assert_not_called()

    def test_native_qualification_rgb_io_without_pillow(self):
        import cv2
        import sys
        from gaussian_generation.npu_branch.qualify import image_metrics, read_rgb, write_rgb
        rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', {'PIL': None}):
            sys.modules.pop('evaluate', None)
            same = np.zeros((16, 16, 3), np.float64)
            self.assertEqual(image_metrics(same, same)['psnr_db'], 120.)
            self.assertEqual(image_metrics(same, same)['ssim'], 1.)
            path = Path(directory) / 'colors.png'
            write_rgb(path, rgb)
            np.testing.assert_array_equal(cv2.imread(str(path)), rgb[:, :, ::-1])
            np.testing.assert_array_equal(read_rgb(path), rgb)
            invalid = Path(directory) / 'invalid.png'
            invalid.write_text('not an image')
            with self.assertRaisesRegex(ValueError, 'Cannot read held-out image'):
                read_rgb(invalid)

    def test_explicit_single_session_and_complete_forwarding(self):
        args=SimpleNamespace(backend='npu',partition_bundle='bundle',partitions=['to_gaussians'],
                             npu_library='bridge',oracle_graphs='graphs',worker_python='python',
                             npu_acceptance='acceptance.json',buffer_policy='per_partition',buffer_limit_mib=64)
        c=BackendConfig.from_args(args);cli=c.cli()
        for name in ('partition-bundle','npu-library','oracle-graphs','worker-python','npu-acceptance'):
            self.assertIn('--'+name,cli)
        self.assertEqual(cli[-2:],['--partitions','to_gaussians'])
        self.assertIn('64',cli)
        self.assertEqual(BackendConfig.from_args(SimpleNamespace()).cli(),['--backend','cpu'])

    def test_unsafe_or_silently_ignored_configuration_rejected(self):
        for values in (
            dict(partitions=('to_gaussians',)),
            dict(backend='npu',partition_bundle=Path('b'),partitions=('to_gaussians',)),
            dict(backend='npu',partition_bundle=Path('b'),npu_library=Path('lib'),partitions=('depth_head','depth_head')),
            dict(backend='onnx_reference',partition_bundle=Path('b'),partitions=('depth_head',),npu_acceptance=Path('r')),
            dict(buffer_limit_mib=0),dict(backend='invalid')):
            with self.subTest(values=values),self.assertRaises(ValueError):BackendConfig(**values)

    def test_runtime_cleanup_when_model_load_fails_and_on_success(self):
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as directory:
                args=SimpleNamespace(backend='npu',partition_bundle='bundle',partitions=['to_gaussians'],
                                     npu_library='bridge',weights='weights',vendor='vendor')
                with patch('gaussian_generation.npu_branch.runtime.PartitionRuntime') as worker_type, \
                     patch('gaussian_generation.npu_branch.readiness.validate_oracles',return_value={'complete':True}), \
                     patch('gaussian_generation.runtime.ModelRuntime') as model:
                    worker=worker_type.return_value.__enter__.return_value
                    if fail:model.side_effect=RuntimeError('model failed')
                    if fail:
                        with self.assertRaisesRegex(RuntimeError,'model failed'):
                            GenerationSession(args,Path(directory),threads=2)
                    else:
                        with GenerationSession(args,Path(directory),threads=2) as session:
                            model.return_value.attach_partitions.assert_called_once_with(worker)
                            self.assertEqual(session.readiness,{'complete':True})
                    worker_type.return_value.__exit__.assert_called_once()

    def test_invalid_readiness_prevents_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(backend='npu',partition_bundle='bundle',partitions=['to_gaussians'],
                                 npu_library='bridge',weights='weights',vendor='vendor')
            with patch('gaussian_generation.npu_branch.runtime.PartitionRuntime') as worker, \
                 patch('gaussian_generation.npu_branch.readiness.validate_oracles',side_effect=ValueError('bad oracle')), \
                 patch('gaussian_generation.runtime.ModelRuntime') as model:
                with self.assertRaisesRegex(ValueError,'bad oracle'):GenerationSession(args,directory,threads=2)
                model.assert_not_called();worker.return_value.__exit__.assert_called_once()
                self.assertFalse(json.loads((Path(directory)/'generation_failure.json').read_text())['public_ready'])

    def test_diagnostic_or_weak_quality_receipt_never_becomes_formal(self):
        self.assertIs(check_receipt(receipt())['complete'],True)
        for field,value in (('diagnostic_only',True),('fpga_executed',False),('npu_executed',False),
                            ('gaussian_repeat_exact',False),('runtime_identity',{}),('partitions',['a','b'])):
            r=receipt();r[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):check_receipt(r)
        for field,value in (('psnr_db',39.),('ssim',.98)):
            r=receipt();r['views']['0']['npu_vs_cpu'][field]=value
            with self.assertRaises(ValueError):check_receipt(r)
        for field,value in (('truth_psnr_drop_db',.21),('truth_ssim_drop',.006)):
            r=receipt();r['views']['0'][field]=value
            with self.assertRaises(ValueError):check_receipt(r)
        r=receipt();del r['views']['2']
        with self.assertRaises(ValueError):check_receipt(r)

    def test_qualified_receipt_binds_runtime_graph_bridge_and_oracle(self):
        r=receipt();worker=SimpleNamespace(parts={'to_gaussians':{}},readiness={'runtime_identity':r['runtime_identity']},provenance=r['provenance'])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'receipt.json';path.write_text(json.dumps(r))
            self.assertEqual(validate_worker(path,worker,r['oracle_checks'])['policy'],'native_fpga_image')
            for key in ('bundle_sha256','bridge_sha256','oracle_manifest_sha256'):
                old=worker.provenance[key];worker.provenance[key]='changed'
                with self.assertRaises(ValueError):validate_worker(path,worker,r['oracle_checks'])
                worker.provenance[key]=old
            changed=copy.deepcopy(r['oracle_checks']);changed[0]['output_sha256']='new output'
            with self.assertRaises(ValueError):validate_worker(path,worker,changed)
            with self.assertRaises(ValueError):validate_worker(path,worker,[])

    def test_multiple_sessions_require_complete_exact_set_receipt(self):
        names=['backbone_cnn','depth_head','to_disparity']
        c=BackendConfig(backend='npu',partition_bundle=Path('bundle'),npu_library=Path('bridge'),partitions=tuple(names))
        self.assertEqual(c.cli()[-3:],names)
        r=receipt();r['partitions']=names
        r['oracle_checks']=[dict(r['oracle_checks'][0],partition=n) for n in names]
        check_receipt(r)
        for checks in (r['oracle_checks'][:-1], r['oracle_checks']+[r['oracle_checks'][0]]):
            invalid=copy.deepcopy(r);invalid['oracle_checks']=checks
            with self.assertRaisesRegex(ValueError,'every selected partition'):check_receipt(invalid)

    def test_finite_stable_bias_is_diagnostic_until_qualified(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'to_gaussians').mkdir();(root/'manifest.json').write_text('{}')
            x=np.ones((1,1,2,2),np.float32);oracle=root/'to_gaussians/oracle.npz';np.savez(oracle,input=x,expected=x)
            meta={'partitions':{'to_gaussians':{'oracle_sha256':sha(oracle)}}}
            worker=SimpleNamespace(parts={'to_gaussians':{}},execute=lambda n,v:v+.1,
                                   provenance={'oracle_manifest_sha256':sha(root/'manifest.json')})
            with patch('gaussian_generation.npu_branch.readiness.validate_bundle',return_value=(meta,{})):
                probe=probe_oracles(worker,root,root/'probe.json')
                self.assertTrue(probe['complete']);self.assertFalse(probe['strict_passed']);self.assertTrue(probe['diagnostic_only'])
                with self.assertRaisesRegex(ValueError,'numerical readiness'):
                    validate_oracles(worker,root,root/'gate.json')
                with patch('gaussian_generation.npu_branch.acceptance.validate_worker',return_value={'policy':'native_fpga_image'}) as gate:
                    passed=validate_oracles(worker,root,root/'gate.json',acceptance='receipt')
                    self.assertTrue(passed['complete']);self.assertFalse(passed['strict_passed']);gate.assert_called_once()

    def test_long_stream_report_uses_sequence_and_bounds_history(self):
        runtime=PartitionRuntime.__new__(PartitionRuntime)
        runtime.calls=deque((dict(sequence=i,npu_executed=True,total_seconds=.1,input_bytes=4,output_bytes=4)
                             for i in range(1000)),maxlen=256)
        runtime.backend='npu';runtime.parts={'to_gaussians':{'meta':{}}};runtime.provenance={}
        runtime.sequence=1000
        self.assertEqual(len(runtime.report(999)['calls']),1)
        self.assertEqual(runtime.report()['history_first_sequence'],744)
        self.assertEqual(len(runtime.calls),256)
        with self.assertRaisesRegex(ValueError,'evicted'):runtime.report(0)

    def test_deployment_manifest_contains_viewer_assets_and_npu_entry(self):
        from initialize.board_sync import SOURCE_FILES
        for name in ('web/viewer.html','web/viewer.css','web/viewer.js',
                     'gaussian_generation/npu_branch/config.py','gaussian_generation/npu_branch/qualify.py'):
            self.assertIn(name,SOURCE_FILES)


if __name__=='__main__':unittest.main()
