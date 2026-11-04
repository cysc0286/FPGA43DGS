"""Explicit generated-source/constructor contracts; no board or video run."""
import builtins
import inspect
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from common import FULL_WEIGHT_SHA256, sha
from gaussian_generation import memory as memory_runtime
from gaussian_generation import upstream as encoder_runtime
from gaussian_generation import runtime as generation_runtime
from gaussian_generation.npu_branch import corr_depth, refine_heads
from gaussian_generation.npu_branch.tests.test_npu_refine_heads import TOY_SOURCE, forward_inputs, load


TARGET = Path('src/model/encoder/costvolume/depth_predictor_multiview.py')
HEADER = '    def forward(self, value, images, near, far):\n'
SOURCE = TOY_SOURCE.replace(HEADER, HEADER + '        if False:\n'
    '            for _ in ():\n' + memory_runtime.OLD + '\n')


def vendor_fixture(root, *, source=SOURCE, base=False, override=None):
    vendor = root/'vendor'
    target = vendor/TARGET
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding='utf-8')
    (vendor/'src/untouched.py').write_text('sentinel = 1\n', encoding='utf-8')
    basic = vendor/'config/model/encoder/costvolume.yaml'
    experiment = vendor/'config/experiment/re10k.yaml'
    basic.parent.mkdir(parents=True)
    experiment.parent.mkdir(parents=True)
    basic.write_text('wo_depth_refine: ' + str(base).lower() + '\n', encoding='utf-8')
    experiment.write_text('model:\n  encoder: ' + ('{}\n' if override is None else
        '\n    wo_depth_refine: ' + str(override).lower() + '\n'), encoding='utf-8')
    return vendor


class GeneratedSourceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_default_bytes_and_no_new_files_are_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vendor = vendor_fixture(root)
            original = (vendor/TARGET).read_bytes()
            with patch.object(refine_heads, 'patch_source', side_effect=AssertionError('not selected')):
                destination = memory_runtime.prepare_chunked_source(vendor, root/'runtime_source', 16)
            replacement = '''                raw_correlation_in = heterogs_chunked_correlation(
                    feat01, feat10, intr_curr, pose_curr, disp_candi_curr, 16)'''
            expected = root/'old_source.py'
            expected.write_text(SOURCE.replace(memory_runtime.OLD, replacement) + memory_runtime.HELPER,
                                encoding='utf-8')
            self.assertEqual((destination/TARGET).read_bytes(), expected.read_bytes())
            self.assertEqual((vendor/TARGET).read_bytes(), original)
            self.assertNotIn(refine_heads.METHOD, (destination/TARGET).read_text(encoding='utf-8'))
            self.assertFalse((root/'refine_heads_pipeline.json').exists())
            self.assertFalse((root/'refine_heads_pipeline.patch').exists())
            self.assertNotIn('refine_heads_pipeline', json.loads((root/'memory_schedule.json').read_text()))

    def test_refine_only_and_both_seams_bind_tools_final_copy_and_unchanged_vendor(self):
        for both in (False, True):
            with self.subTest(both=both), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                vendor = vendor_fixture(root)
                before = {path.relative_to(vendor): path.read_bytes()
                          for path in vendor.rglob('*') if path.is_file()}
                options = dict(refine_heads_pipeline=True)
                if both:
                    options['corr_depth_pipeline'] = True
                destination = memory_runtime.prepare_chunked_source(vendor, root/'runtime_source', 16, **options)
                self.assertEqual(before, {path.relative_to(vendor): path.read_bytes()
                    for path in vendor.rglob('*') if path.is_file()})
                source = (destination/TARGET).read_text(encoding='utf-8')
                self.assertIn('def _refine_to_heads(', source)
                self.assertEqual('def _corr_to_depth_logits(' in source, both)
                schedule = json.loads((root/'memory_schedule.json').read_text())
                for name, tool in [('refine_heads_pipeline', refine_heads)] + (
                        [('corr_depth_pipeline', corr_depth)] if both else []):
                    record = json.loads((root/(name+'.json')).read_text())
                    self.assertEqual(record['generated_file_sha256'], sha(destination/TARGET))
                    self.assertEqual(record['original_file_sha256'], sha(vendor/TARGET))
                    self.assertEqual(record['tool_sha256'], sha(Path(tool.__file__)))
                    self.assertFalse(record['qualified'])
                    self.assertFalse(record['npu_executed'])
                    self.assertEqual(schedule[name]['metadata'], name+'.json')
                    self.assertIn('def ' + tool.METHOD + '(', (root/(name+'.patch')).read_text())
                self.assertEqual(schedule['modified_sha256'], sha(destination/TARGET))

    def test_real_small_torch_state_and_both_cpu_paths_remain_exact(self):
        torch.manual_seed(120)
        arguments = forward_inputs()
        for both in (False, True):
            with self.subTest(both=both), tempfile.TemporaryDirectory() as directory, torch.inference_mode():
                root = Path(directory)
                vendor = vendor_fixture(root)
                destination = memory_runtime.prepare_chunked_source(vendor, root/'runtime_source', 16,
                    refine_heads_pipeline=True, corr_depth_pipeline=both)
                original_type = load(SOURCE)
                candidate_type = load((destination/TARGET).read_text(encoding='utf-8'))
                for ablation in (False, True):
                    original = original_type(ablation).eval()
                    candidate = candidate_type(ablation).eval()
                    candidate.load_state_dict(original.state_dict(), strict=True)
                    self.assertEqual(original.state_dict().keys(), candidate.state_dict().keys())
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(
                        original(*arguments), candidate(*arguments))))

    def test_existing_or_changed_source_is_rejected_before_writing(self):
        sources = [refine_heads.patch_source(SOURCE)[0],
            SOURCE.replace('proj_feature, fullres_disps, pdf_max', 'proj_feature, pdf_max, fullres_disps'),
            SOURCE.replace(memory_runtime.OLD, '                changed_upstream()')]
        for source in sources:
            with self.subTest(source=source[-70:]), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                vendor = vendor_fixture(root, source=source)
                before = (vendor/TARGET).read_bytes()
                with self.assertRaises(ValueError):
                    memory_runtime.prepare_chunked_source(vendor, root/'runtime_source', 16,
                                                          refine_heads_pipeline=True)
                self.assertFalse((root/'runtime_source').exists())
                self.assertFalse((root/'refine_heads_pipeline.json').exists())
                self.assertEqual((vendor/TARGET).read_bytes(), before)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vendor = vendor_fixture(root)
            destination = root/'runtime_source'
            destination.mkdir()
            sentinel = destination/'keep'
            sentinel.write_bytes(b'untouched')
            with self.assertRaises(FileExistsError):
                memory_runtime.prepare_chunked_source(vendor, destination, 16, refine_heads_pipeline=True)
            self.assertEqual(sentinel.read_bytes(), b'untouched')

    def test_invalid_flags_and_chunk_fail_before_source_access(self):
        with patch.object(Path, 'read_text', side_effect=AssertionError('source opened')):
            for flag in (1, 0, None, 'false'):
                with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'boolean'):
                    memory_runtime.prepare_chunked_source('missing', 'missing', 16, refine_heads_pipeline=flag)
            for chunk in (True, False, 0, -1, 129, 1.5, None, '16'):
                with self.subTest(chunk=chunk), self.assertRaisesRegex(ValueError, '[Dd]epth chunk'):
                    memory_runtime.prepare_chunked_source('missing', 'missing', chunk, refine_heads_pipeline=True)

    def test_new_seam_source_hash_changes_execution_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vendor = vendor_fixture(root)
            driver = root/'fake/gaussian_generation/runtime.py'
            driver.parent.mkdir(parents=True)
            driver.write_text('driver=1\n')
            (driver.parent/'memory.py').write_text('schedule=1\n')
            (driver.parent.parent/'common.py').write_text('common=1\n')
            tool = driver.parent/'npu_branch/refine_heads.py'
            tool.parent.mkdir()
            tool.write_text('seam=1\n')
            with patch.object(generation_runtime, '__file__', str(driver)):
                before = generation_runtime.execution_source_hashes(vendor)
                tool.write_text('seam=2\n')
                after = generation_runtime.execution_source_hashes(vendor)
            key = 'runtime/gaussian_generation/npu_branch/refine_heads.py'
            self.assertNotEqual(before[key], after[key])
            self.assertEqual(before[TARGET.as_posix()], after[TARGET.as_posix()])


class ConstructorTests(unittest.TestCase):
    def test_flags_and_nonzero_integer_chunk_checked_before_import_or_weight_access(self):
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name in ('torch', 'numpy', 'omegaconf'):
                raise AssertionError('numerical dependency imported')
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded_import), \
                patch.object(generation_runtime, 'sha', side_effect=AssertionError('weight accessed')):
            for chunk in (0, -1, 129, True, None, 1.5, '16'):
                with self.subTest(chunk=chunk), self.assertRaisesRegex(ValueError, 'nonzero depth_chunk'):
                    generation_runtime.ModelRuntime('missing', 'missing', 1, chunk, 'missing',
                                                    refine_heads_pipeline=True)
            for flag in ('true', 0, 1, None):
                with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'boolean'):
                    generation_runtime.ModelRuntime('missing', 'missing', 1, 16, 'missing',
                                                    refine_heads_pipeline=flag)

    def test_keyword_only_tail_default_is_false(self):
        signature = inspect.signature(generation_runtime.ModelRuntime.__init__)
        option = signature.parameters['refine_heads_pipeline']
        self.assertIs(option.default, False)
        self.assertEqual(option.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(list(signature.parameters)[-2:], ['corr_depth_pipeline', 'refine_heads_pipeline'])

    def test_default_refine_and_combined_constructors_keep_state_outputs_and_old_calls(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        reference = load(SOURCE)().eval()
        checkpoint = {'state_dict': {'encoder.depth_predictor.' + key: value
                                    for key, value in reference.state_dict().items()}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vendor = vendor_fixture(root)
            original = (vendor/TARGET).read_bytes()
            weights = root/'weights.ckpt'
            weights.write_bytes(b'fixture; torch.load is injected')
            def load_encoder(active):
                predictor_type = load((Path(active)/TARGET).read_text(encoding='utf-8'))
                class Encoder(torch.nn.Module):
                    def __init__(self, cfg):
                        super().__init__()
                        self.depth_predictor = predictor_type(cfg.wo_depth_refine)
                return Encoder, lambda cfg: None
            def digest(path):
                return FULL_WEIGHT_SHA256 if Path(path) == weights else sha(path)
            with patch.dict(os.environ), patch.object(generation_runtime, 'sha', side_effect=digest), \
                    patch.object(torch, 'load', return_value=checkpoint), \
                    patch.object(torch, 'set_num_interop_threads'), \
                    patch.object(encoder_runtime, 'load_encoder', side_effect=load_encoder), \
                    patch.object(generation_runtime.importlib.metadata, 'version', return_value='test-only'), \
                    patch.object(memory_runtime, 'prepare_chunked_source', wraps=memory_runtime.prepare_chunked_source) as prepare:
                cases = [({}, {}), ({'refine_heads_pipeline': True}, {'refine_heads_pipeline': True}),
                         ({'corr_depth_pipeline': True, 'refine_heads_pipeline': True},
                          {'corr_depth_pipeline': True, 'refine_heads_pipeline': True})]
                runtimes = []
                for index, (options, expected_kwargs) in enumerate(cases):
                    prepare.reset_mock()
                    runtime = generation_runtime.ModelRuntime(weights, vendor, 1, 16, root/str(index), **options)
                    prepare.assert_called_once_with(vendor, root/str(index)/'runtime_source', 16, **expected_kwargs)
                    self.assertEqual(runtime.refine_heads_pipeline, options.get('refine_heads_pipeline', False))
                    self.assertEqual('refine_heads_pipeline' in runtime.metadata, bool(options))
                    self.assertEqual(hasattr(runtime.model.depth_predictor, refine_heads.METHOD), bool(options))
                    runtimes.append(runtime)
            arguments = forward_inputs()
            with torch.inference_mode():
                baseline = runtimes[0].model.depth_predictor(*arguments)
                for candidate in runtimes[1:]:
                    self.assertEqual(candidate.model.state_dict().keys(), runtimes[0].model.state_dict().keys())
                    self.assertTrue(all(torch.equal(value, candidate.model.state_dict()[key])
                        for key, value in runtimes[0].model.state_dict().items()))
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(
                        baseline, candidate.model.depth_predictor(*arguments))))
                    self.assertNotEqual(candidate.source_hashes[TARGET.as_posix()],
                                        runtimes[0].source_hashes[TARGET.as_posix()])
                    self.assertIn('runtime/gaussian_generation/npu_branch/refine_heads.py', candidate.source_hashes)
            self.assertEqual((vendor/TARGET).read_bytes(), original)

    def test_ablation_yaml_rejected_before_model_or_checkpoint_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vendor = vendor_fixture(root, override=True)
            with patch.object(generation_runtime, 'sha', return_value=FULL_WEIGHT_SHA256), \
                    patch.object(encoder_runtime, 'load_encoder', return_value=(None, None)), \
                    patch.object(torch, 'load', side_effect=AssertionError('checkpoint loaded')), \
                    patch.object(torch, 'set_num_interop_threads'):
                with self.assertRaisesRegex(ValueError, 'before checkpoint loading'):
                    generation_runtime.ModelRuntime(root/'weights', vendor, 1, 16, root/'runtime',
                                                    refine_heads_pipeline=True)

    def test_wrong_generated_predictor_or_actual_ablation_rejected_before_checkpoint(self):
        for actual_ablation in (False, True):
            with self.subTest(actual_ablation=actual_ablation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                vendor = vendor_fixture(root)
                original_type = load(SOURCE)
                class WrongEncoder(torch.nn.Module):
                    def __init__(self, cfg):
                        super().__init__()
                        self.depth_predictor = original_type(actual_ablation)
                with patch.object(generation_runtime, 'sha', return_value=FULL_WEIGHT_SHA256), \
                        patch.object(encoder_runtime, 'load_encoder', return_value=(WrongEncoder, lambda cfg: None)), \
                        patch.object(torch, 'load', side_effect=AssertionError('checkpoint loaded')), \
                        patch.object(torch, 'set_num_interop_threads'):
                    with self.assertRaisesRegex(ValueError, 'non-ablation generated method seam'):
                        generation_runtime.ModelRuntime(root/'weights', vendor, 1, 16, root/'runtime',
                                                        refine_heads_pipeline=True)


if __name__ == '__main__':
    unittest.main()
