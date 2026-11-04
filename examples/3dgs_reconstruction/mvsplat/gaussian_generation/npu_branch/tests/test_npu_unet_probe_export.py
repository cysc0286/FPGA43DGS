"""Tiny original-topology U-Net probes; no pretrained model or NPU execution."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from common import save, sha
from gaussian_generation.npu_branch.unet_probe_export import (
    boundary_plan, capture_boundaries, checked_array, checkpoint_scope,
    explicit_output, export_candidate, load_captured_samples, node_module,
    prefix_module, select_boundaries,
)


class AttentionBlock(torch.nn.Module):
    def forward(self, value):
        return self._forward(value)

    def _forward(self, value):
        # Preserve cross-view dependence in the toy, too.
        return value+value.flip(0)*.125


class TinyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.float32
        self.cross_attn_condition = False
        self.num_classes = None
        self.predict_codebook_ids = False
        self.input_blocks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(2, 2, 1)),
            torch.nn.Sequential(torch.nn.Conv2d(2, 2, 3, stride=2, padding=1)),
        ])
        self.middle_block = torch.nn.Sequential(
            torch.nn.GroupNorm(1, 2), AttentionBlock(), torch.nn.SiLU())
        self.output_blocks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(4, 2, 1),
                                torch.nn.Upsample(scale_factor=2, mode='nearest')),
            torch.nn.Sequential(torch.nn.Conv2d(4, 2, 1)),
        ])
        self.out = torch.nn.Sequential(torch.nn.Conv2d(2, 3, 1), torch.nn.SiLU())

    def forward(self, value):
        h, skips = value.type(self.dtype), []
        for block in self.input_blocks:
            h = block(h)
            skips.append(h)
        for block in self.middle_block:
            h = block(h)
        for block in self.output_blocks:
            h = block(torch.cat((h, skips.pop()), dim=1))
        return self.out(h.type(value.dtype))


class BoundaryProbeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.module = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 1),
            torch.nn.GroupNorm(1, 2), torch.nn.GELU(), TinyUNet(),
            torch.nn.Conv2d(3, 2, 1)).eval()
        self.x = np.arange(96, dtype=np.float32).reshape(2, 3, 4, 4)/96
        with torch.inference_mode():
            self.y = self.module(torch.from_numpy(self.x)).numpy()

    def capture(self, root, label='A', value=None):
        value = self.x if value is None else value
        with torch.inference_mode():
            expected = self.module(torch.from_numpy(value)).numpy()
        return capture_boundaries(self.module, value, expected, root, label=label)

    def test_plan_matches_all_actual_unet_paths(self):
        index, plan = boundary_plan(self.module)
        self.assertEqual(index, 3)
        self.assertEqual(plan[0].name, 'stem_00')
        self.assertEqual(plan[-1].name, 'full')
        self.assertIn('middle_01', {node.name for node in plan})
        for node in plan:
            self.assertIsInstance(node_module(self.module, node), torch.nn.Module)

    def test_selection_rejects_unknown_globs(self):
        _, plan = boundary_plan(self.module)
        self.assertEqual([node.name for node in select_boundaries(plan, ['stem_*'])],
                         ['stem_00', 'stem_01', 'stem_02'])
        with self.assertRaisesRegex(ValueError, 'Unknown boundary'):
            select_boundaries(plan, ['up_99'])

    def test_optional_layers_exclude_rank_three_attention_internals(self):
        self.module[3].middle_block[1].norm = torch.nn.GroupNorm(1, 2)
        _, plan = boundary_plan(self.module, include_layers=True)
        self.assertTrue(any(node.group == 'layer' for node in plan))
        self.assertFalse(any(node.path.endswith('middle_block.1.norm') for node in plan))
        self.assertTrue(all(not node.prefix for node in plan if node.group == 'layer'))

    def test_every_block_and_prefix_match_original_captured_skip_path(self):
        with tempfile.TemporaryDirectory() as folder:
            trace = self.capture(folder)
            _, plan = boundary_plan(self.module)
            for node in plan:
                with np.load(Path(folder)/trace['boundaries'][node.name]['oracle']) as f:
                    x, expected = f['input'], f['expected']
                with torch.inference_mode():
                    actual_block = node_module(self.module, node)(torch.from_numpy(x)).numpy()
                    actual_prefix = prefix_module(self.module, node.name)(torch.from_numpy(self.x)).numpy()
                np.testing.assert_array_equal(actual_block, expected, err_msg=node.name)
                np.testing.assert_array_equal(actual_prefix, expected, err_msg=node.name)

    def test_up_block_receives_original_concatenated_skip_without_cutting_views(self):
        with tempfile.TemporaryDirectory() as folder:
            trace = self.capture(folder)
            entry = trace['boundaries']['up_00']
            self.assertEqual(entry['input_shape'], [2, 4, 2, 2])
            self.assertEqual(entry['output_shape'], [2, 2, 4, 4])

    def test_capture_restores_hooks_on_reference_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'does not match'):
                capture_boundaries(self.module, self.x, self.y+1, folder, label='A')
        self.assertFalse(any(child._forward_hooks for child in self.module.modules()))

    def test_capture_refuses_overwrite_and_training(self):
        with tempfile.TemporaryDirectory() as folder:
            self.capture(folder)
            with self.assertRaises(FileExistsError):
                self.capture(folder)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'eval mode'):
                capture_boundaries(self.module.train(), self.x, self.y, folder, label='A')

    def test_rejects_conditional_topology(self):
        self.module[3].cross_attn_condition = True
        with self.assertRaisesRegex(ValueError, 'multi-input'):
            boundary_plan(self.module)

    def test_identity_sink_preserves_non_square_channel_order(self):
        value = torch.arange(24, dtype=torch.float32).reshape(2, 2, 2, 3)
        wrapped = explicit_output(torch.nn.Identity(), 2, 'NHWC', True)
        actual = wrapped(value)
        self.assertEqual(tuple(actual.shape), (2, 2, 3, 2))
        self.assertTrue(torch.equal(actual.permute(0, 3, 1, 2), value))

    def test_plain_conv_checkpoint_scope_is_legal(self):
        with checkpoint_scope(self.module[0]) as patched:
            self.assertEqual(patched, [])

    def test_invalid_boundary_tensor_rejected(self):
        with self.assertRaisesRegex(ValueError, 'rank four'):
            checked_array(torch.ones(2, 3), 'bad')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            checked_array(torch.full((1, 1, 1, 1), torch.inf), 'bad')

    def test_capture_loader_hash_checks_block_and_prefix_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stage = root/'refine_unet'
            stage.mkdir()
            trace = self.capture(stage)
            source = stage/'source_a.npz'
            np.savez(source, input=self.x, expected=self.y)
            trace.update(source_fixture=source.name, source_fixture_sha256=sha(source))
            save(root/'capture_manifest.json', {'modules':{'refine_unet':{'captures':{'A':trace}}}})
            node = next(n for n in boundary_plan(self.module)[1] if n.name == 'up_00')
            _, block = load_captured_samples(root, 'refine_unet', node, 'block')
            _, prefix = load_captured_samples(root, 'refine_unet', node, 'prefix')
            self.assertEqual(block['A'][0].shape, (2, 4, 2, 2))
            np.testing.assert_array_equal(prefix['A'][0], self.x)
            source.write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'source fixture hash'):
                load_captured_samples(root, 'refine_unet', node, 'prefix')

    def test_actual_tiny_onnx_block_and_prefix_both_validate_a_b(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            a = self.capture(root, 'A')
            b = self.capture(root, 'B', np.zeros_like(self.x))
            node = next(n for n in boundary_plan(self.module)[1] if n.name == 'up_00')
            samples = {}
            for label, trace in [('A', a), ('B', b)]:
                with np.load(root/trace['boundaries'][node.name]['oracle']) as data:
                    samples[label] = (data['input'].copy(), data['expected'].copy())
            block_folder = root/'export_block'
            block_folder.mkdir()
            item = export_candidate(node_module(self.module, node), self.module, samples,
                block_folder, identity_conv_tail=True)
            self.assertTrue(item['complete'])
            self.assertEqual(item['identity_conv_multiply_count'], 128)
            self.assertTrue(item['host_onnx_checks']['B']['passed'])
            self.assertFalse(item['input_switch_reference']['input_exactly_equal'])
            prefix_samples = {'A':(self.x, samples['A'][1]),
                              'B':(np.zeros_like(self.x), samples['B'][1])}
            prefix_folder = root/'export_prefix'
            prefix_folder.mkdir()
            item = export_candidate(prefix_module(self.module, node.name), self.module,
                                    prefix_samples, prefix_folder)
            self.assertTrue(item['complete'])
            self.assertNotIn('forward', self.module[3].middle_block[1].__dict__)


if __name__ == '__main__':
    unittest.main()
