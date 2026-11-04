"""Opt-in generated-source integration; no real checkpoint or board execution."""
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
from gaussian_generation.npu_branch import corr_depth
from gaussian_generation.npu_branch.tests.test_npu_corr_depth import TOY_SOURCE, load


TARGET = Path('src/model/encoder/costvolume/depth_predictor_multiview.py')
# The dead block supplies the exact upstream cost-volume scheduling seam while
# the live small predictor evaluates the unchanged three-step logit path.
SOURCE = TOY_SOURCE.replace('    def forward(self, raw_correlation_in):\n',
    '    def forward(self, raw_correlation_in):\n'
    '        if False:\n'
    '            for _ in ():\n' + memory_runtime.OLD + '\n')


def vendor_fixture(root, source=SOURCE):
    vendor = root/'vendor'
    target = vendor/TARGET
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding='utf-8')
    (vendor/'src/untouched.py').write_text('sentinel = 1\n', encoding='utf-8')
    return vendor


def generated(root, vendor, *, enabled):
    destination = root/'runtime_source'
    memory_runtime.prepare_chunked_source(vendor, destination, 16, corr_depth_pipeline=enabled)
    return destination


class GeneratedSourceTests(unittest.TestCase):
    def test_default_text_and_bytes_equal_the_previous_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vendor = vendor_fixture(root)
            original = (vendor/TARGET).read_bytes()
            with patch.object(corr_depth, 'patch_source', side_effect=AssertionError('not opt-in')):
                destination = memory_runtime.prepare_chunked_source(vendor, root/'default', 16)
            replacement = '''                raw_correlation_in = heterogs_chunked_correlation(
                    feat01, feat10, intr_curr, pose_curr, disp_candi_curr, 16)'''
            expected = SOURCE.replace(memory_runtime.OLD, replacement) + memory_runtime.HELPER
            # Reproduce the prior Path.write_text newline behavior on this OS.
            expected_file = root/'previous.py'
            expected_file.write_text(expected, encoding='utf-8')
            self.assertEqual((destination/TARGET).read_bytes(), expected_file.read_bytes())
            self.assertEqual((vendor/TARGET).read_bytes(), original)
            self.assertNotIn(corr_depth.METHOD, (destination/TARGET).read_text())
            self.assertFalse((root/'corr_depth_pipeline.json').exists())
            self.assertFalse((root/'corr_depth_pipeline.patch').exists())
            record = json.loads((root/'memory_schedule.json').read_text())
            self.assertNotIn('corr_depth_pipeline', record)

    def test_opt_in_writes_only_copy_and_binds_transform_tool_and_final_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vendor = vendor_fixture(root)
            original = {file.relative_to(vendor): file.read_bytes() for file in vendor.rglob('*') if file.is_file()}
            destination = generated(root, vendor, enabled=True)
            source = (destination/TARGET).read_text(encoding='utf-8')
            self.assertIn('def _corr_to_depth_logits(', source)
            self.assertIn('heterogs_chunked_correlation(', source)
            self.assertEqual({file.relative_to(vendor): file.read_bytes() for file in vendor.rglob('*')
                              if file.is_file()}, original)
            self.assertEqual((destination/'src/untouched.py').read_bytes(), original[Path('src/untouched.py')])
            record = json.loads((root/'corr_depth_pipeline.json').read_text())
            self.assertEqual(record['generated_file_sha256'], sha(destination/TARGET))
            self.assertEqual(record['original_file_sha256'], sha(vendor/TARGET))
            self.assertEqual(record['tool_sha256'], sha(Path(corr_depth.__file__)))
            self.assertFalse(record['qualified'])
            self.assertFalse(record['npu_executed'])
            self.assertEqual(record['transform']['matched_blocks'], 1)
            schedule = json.loads((root/'memory_schedule.json').read_text())
            self.assertEqual(schedule['modified_sha256'], record['generated_file_sha256'])
            self.assertEqual(schedule['corr_depth_pipeline']['metadata'], 'corr_depth_pipeline.json')
            self.assertIn('def _corr_to_depth_logits(', (root/'corr_depth_pipeline.patch').read_text())
            identities = generation_runtime.execution_source_hashes(destination)
            self.assertEqual(identities[TARGET.as_posix()], sha(destination/TARGET))
            self.assertEqual(identities['runtime/gaussian_generation/npu_branch/corr_depth.py'],
                             sha(Path(corr_depth.__file__)))

    def test_generated_copy_keeps_real_small_torch_state_and_both_cpu_methods(self):
        with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
            root = Path(directory)
            vendor = vendor_fixture(root)
            destination = generated(root, vendor, enabled=True)
            original_type = load(SOURCE)
            changed_type = load((destination/TARGET).read_text(encoding='utf-8'))
            value = torch.randn(2, 4, 7, 9)
            for ablation in (False, True):
                original, changed = original_type(ablation).eval(), changed_type(ablation).eval()
                changed.load_state_dict(original.state_dict(), strict=True)
                self.assertEqual(list(original.state_dict()), list(changed.state_dict()))
                self.assertTrue(callable(getattr(changed, corr_depth.METHOD)))
                self.assertTrue(torch.equal(original(value), changed(value)))

    def test_collision_and_changed_seam_fail_before_destination_creation(self):
        broken_sources = [SOURCE.replace('raw_correlation + self.regressor_residual',
                                          'raw_correlation - self.regressor_residual'),
                          corr_depth.patch_source(SOURCE)[0],
                          SOURCE.replace(memory_runtime.OLD, '                changed_upstream()')]
        for source in broken_sources:
            with self.subTest(source=source[-50:]), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                vendor = vendor_fixture(root, source)
                original = (vendor/TARGET).read_bytes()
                with self.assertRaises(ValueError):
                    generated(root, vendor, enabled=True)
                self.assertFalse((root/'runtime_source').exists())
                self.assertFalse((root/'corr_depth_pipeline.json').exists())
                self.assertEqual((vendor/TARGET).read_bytes(), original)

    def test_existing_destination_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vendor = vendor_fixture(root)
            destination = root/'runtime_source'
            destination.mkdir()
            sentinel = destination/'keep.txt'
            sentinel.write_bytes(b'keep prior copy')
            original = (vendor/TARGET).read_bytes()
            with self.assertRaises(FileExistsError):
                generated(root, vendor, enabled=True)
            self.assertEqual(sentinel.read_bytes(), b'keep prior copy')
            self.assertEqual((vendor/TARGET).read_bytes(), original)
            self.assertFalse((root/'corr_depth_pipeline.json').exists())

    def test_explicit_boolean_required_before_source_access(self):
        with patch.object(Path, 'read_text', side_effect=AssertionError('should not access source')):
            for value in ('false', 0, 1, None):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'boolean'):
                    memory_runtime.prepare_chunked_source('absent', 'absent-copy', 16,
                                                         corr_depth_pipeline=value)
            for chunk in (True, 1.5, None, 0):
                with self.subTest(chunk=chunk), self.assertRaisesRegex(ValueError, '[Dd]epth chunk'):
                    memory_runtime.prepare_chunked_source('absent', 'absent-copy', chunk,
                                                         corr_depth_pipeline=True)

    def test_identity_changes_when_seam_tool_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vendor = vendor_fixture(root)
            driver = root/'fake_mvsplat/gaussian_generation/runtime.py'
            driver.parent.mkdir(parents=True)
            driver.write_text('driver = 1\n')
            (driver.parent/'memory.py').write_text('schedule = 1\n')
            (driver.parent.parent/'common.py').write_text('common = 1\n')
            tool = driver.parent/'npu_branch/corr_depth.py'
            tool.parent.mkdir()
            tool.write_text('seam = 1\n')
            with patch.object(generation_runtime, '__file__', str(driver)):
                before = generation_runtime.execution_source_hashes(vendor)
                tool.write_text('seam = 2\n')
                after = generation_runtime.execution_source_hashes(vendor)
            key = 'runtime/gaussian_generation/npu_branch/corr_depth.py'
            self.assertNotEqual(before[key], after[key])
            self.assertEqual(before[TARGET.as_posix()], after[TARGET.as_posix()])


class ConstructorTests(unittest.TestCase):
    def test_candidate_zero_or_invalid_chunk_rejected_before_import_or_checkpoint_access(self):
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name in ('torch', 'numpy', 'omegaconf'):
                raise AssertionError('constructor imported numerical runtime before rejection')
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded_import), \
                patch.object(generation_runtime, 'sha', side_effect=AssertionError('checkpoint accessed')):
            for chunk in (0, -1, 129, True, None):
                with self.subTest(chunk=chunk), self.assertRaisesRegex(ValueError, 'nonzero depth_chunk'):
                    generation_runtime.ModelRuntime('missing', 'missing', 1, chunk, 'missing',
                                                     corr_depth_pipeline=True)
            for flag in ('true', 1, None):
                with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'boolean'):
                    generation_runtime.ModelRuntime('missing', 'missing', 1, 16, 'missing',
                                                     corr_depth_pipeline=flag)

    def test_new_constructor_option_is_keyword_only_with_unchanged_default(self):
        signature = inspect.signature(generation_runtime.ModelRuntime.__init__)
        option = signature.parameters['corr_depth_pipeline']
        self.assertIs(option.default, False)
        self.assertEqual(option.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(list(signature.parameters)[:6],
                         ['self', 'weights', 'vendor', 'threads', 'depth_chunk', 'runtime_dir'])

    def test_default_and_opt_in_constructor_only_load_their_own_copy_with_same_state(self):
        from omegaconf import OmegaConf
        torch.set_num_threads(1)
        original_type = load(SOURCE)
        reference = original_type().eval()
        checkpoint = {'state_dict': {'encoder.depth_predictor.' + name: value
                                    for name, value in reference.state_dict().items()}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vendor = vendor_fixture(root)
            original = (vendor/TARGET).read_bytes()
            weights = root/'weights.ckpt'
            weights.write_bytes(b'test-only, never loaded')
            def load_encoder(active):
                predictor_type = load((Path(active)/TARGET).read_text(encoding='utf-8'))
                class TinyEncoder(torch.nn.Module):
                    def __init__(self, cfg):
                        super().__init__()
                        self.depth_predictor = predictor_type()
                return TinyEncoder, lambda cfg: None
            def config(path):
                return OmegaConf.create({'model': {'encoder': {}}} if Path(path).name == 're10k.yaml' else {})
            def digest(path):
                return FULL_WEIGHT_SHA256 if Path(path) == weights else sha(path)
            prepare = memory_runtime.prepare_chunked_source
            with patch.dict(os.environ), patch.object(generation_runtime, 'sha', side_effect=digest), \
                    patch.object(torch, 'load', return_value=checkpoint), \
                    patch.object(torch, 'set_num_interop_threads'), \
                    patch.object(encoder_runtime, 'load_encoder', side_effect=load_encoder), \
                    patch.object(OmegaConf, 'load', side_effect=config), \
                    patch.object(generation_runtime.importlib.metadata, 'version', return_value='test-only'), \
                    patch.object(memory_runtime, 'prepare_chunked_source', wraps=prepare) as prepare_spy:
                baseline = generation_runtime.ModelRuntime(weights, vendor, 1, 16, root/'baseline')
                prepare_spy.assert_called_once_with(vendor, root/'baseline/runtime_source', 16)
                prepare_spy.reset_mock()
                candidate = generation_runtime.ModelRuntime(weights, vendor, 1, 16, root/'candidate',
                                                              corr_depth_pipeline=True)
                prepare_spy.assert_called_once_with(vendor, root/'candidate/runtime_source', 16,
                                                    corr_depth_pipeline=True)
            self.assertFalse(baseline.corr_depth_pipeline)
            self.assertNotIn('corr_depth_pipeline', baseline.metadata)
            self.assertFalse(hasattr(baseline.model.depth_predictor, corr_depth.METHOD))
            self.assertTrue(candidate.corr_depth_pipeline)
            self.assertTrue(candidate.metadata['corr_depth_pipeline'])
            self.assertTrue(hasattr(candidate.model.depth_predictor, corr_depth.METHOD))
            self.assertEqual(baseline.model.state_dict().keys(), candidate.model.state_dict().keys())
            for name, value in baseline.model.state_dict().items():
                self.assertTrue(torch.equal(value, candidate.model.state_dict()[name]))
            value = torch.randn(2, 4, 7, 9)
            with torch.inference_mode():
                self.assertTrue(torch.equal(baseline.model.depth_predictor(value),
                                            candidate.model.depth_predictor(value)))
            key = TARGET.as_posix()
            self.assertNotEqual(baseline.source_hashes[key], candidate.source_hashes[key])
            self.assertIn('runtime/gaussian_generation/npu_branch/corr_depth.py', candidate.source_hashes)
            self.assertEqual((vendor/TARGET).read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
