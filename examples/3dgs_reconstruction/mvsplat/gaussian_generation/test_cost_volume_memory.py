"""Check cost-volume inference scratch without changing geometry or hypotheses."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from gaussian_generation import memory as memory_runtime


def warp(features, intrinsics, pose, depth, warp_padding_mode='zeros'):
    """The upstream bilinear pose/depth warp, isolated from model imports."""
    batch, candidates, height, width = depth.shape
    with torch.no_grad():
        yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        grid = torch.stack((xx, yy, torch.ones_like(xx))).to(depth).unsqueeze(0).repeat(batch, 1, 1, 1)
        points = torch.inverse(intrinsics).bmm(grid.reshape(batch, 3, -1))
        points = pose[:, :3, :3].bmm(points).unsqueeze(2).repeat(1, 1, candidates, 1) * depth.reshape(batch, 1, candidates, -1)
        points = points + pose[:, :3, -1:].unsqueeze(-1)
        points = intrinsics.bmm(points.reshape(batch, 3, -1)).reshape(batch, 3, candidates, -1)
        pixels = points[:, :2] / points[:, -1:].clamp(min=1e-3)
        grid = torch.stack((2 * pixels[:, 0] / (width - 1) - 1,
                            2 * pixels[:, 1] / (height - 1) - 1), dim=-1)
    return F.grid_sample(features, grid.reshape(batch, candidates * height, width, 2),
                         mode='bilinear', padding_mode=warp_padding_mode, align_corners=True).reshape(
                             batch, features.shape[1], candidates, height, width)


def baseline(first, second, intrinsics, pose, disparities, chunk):
    pieces = []
    for start in range(0, disparities.shape[1], chunk):
        depth = 1.0 / disparities[:, start:start + chunk].repeat([1, 1, *second.shape[-2:]])
        warped = warp(second, intrinsics, pose, depth)
        correlation = (first.unsqueeze(2) * warped).sum(1) / first.shape[1] ** .5
        pieces.append(correlation)
        del warped, depth
    return torch.cat(pieces, dim=1)


def candidate():
    namespace = dict(torch=torch, warp_with_pose_depth_candidates=warp)
    exec(memory_runtime.HELPER, namespace)
    return namespace['heterogs_chunked_correlation']


def inputs(channels=7, height=11, width=13):
    generator = torch.Generator().manual_seed(43)
    first = torch.randn(2, channels, height, width, generator=generator)
    second = torch.randn(2, channels, height, width, generator=generator)
    intrinsics = torch.tensor([[8., 0., 6.], [0., 9., 5.], [0., 0., 1.]]).repeat(2, 1, 1)
    pose = torch.eye(4).repeat(2, 1, 1)
    pose[:, 0, 3] = torch.tensor([.13, -.17])
    pose[:, 1, 3] = torch.tensor([-.07, .09])
    disparities = torch.linspace(.05, .9, 128).reshape(1, 128, 1, 1).repeat(2, 1, 1, 1)
    return first, second, intrinsics, pose, disparities


class ScratchAllocations(TorchDispatchMode):
    """Count actual allocating dispatcher operations at the memory seam."""
    def __init__(self):
        super().__init__()
        self.product_bytes = self.concatenate_bytes = 0

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        if function == torch.ops.aten.mul.Tensor and isinstance(result, torch.Tensor) and result.ndim == 5:
            self.product_bytes += result.numel() * result.element_size()
        if function == torch.ops.aten.cat.default:
            self.concatenate_bytes += result.numel() * result.element_size()
        return result


class CostVolumeMemoryTests(unittest.TestCase):
    def test_inference_removes_full_channel_product_and_final_concatenation(self):
        values = inputs()
        with torch.inference_mode(), ScratchAllocations() as old:
            expected = baseline(*values, 16)
        with torch.inference_mode(), ScratchAllocations() as new:
            actual = candidate()(*values, 16)
        self.assertGreater(old.product_bytes, 0)
        self.assertGreater(old.concatenate_bytes, 0)
        self.assertEqual(new.product_bytes, 0, 'Inference still allocates a full channel-by-depth product')
        self.assertEqual(new.concatenate_bytes, 0, 'Inference still duplicates the final correlation volume')
        self.assertTrue(torch.equal(actual, expected))

    def test_all_hypotheses_boundaries_and_feature_inputs_remain_exact(self):
        values = inputs()
        snapshots = [v.clone() for v in values]
        with torch.inference_mode():
            for chunk in (1, 7, 16, 31, 128):
                expected = baseline(*values, chunk)
                actual = candidate()(*values, chunk)
                self.assertEqual(tuple(actual.shape), (2, 128, 11, 13))
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertEqual(actual.device, expected.device)
                self.assertTrue(torch.equal(actual, expected), 'Changed chunk numerical output: '+str(chunk))
        for actual, expected in zip(values, snapshots):
            self.assertTrue(torch.equal(actual, expected), 'Mutation escaped the disposable warp buffer')

    def test_training_keeps_original_differentiable_schedule(self):
        values = inputs()
        original = [values[0].requires_grad_(), values[1].requires_grad_(), *values[2:]]
        copied = [v.detach().clone().requires_grad_(v.requires_grad) for v in original]
        before = baseline(*original, 16)
        after = candidate()(*copied, 16)
        self.assertTrue(torch.equal(before, after))
        before.square().sum().backward()
        after.square().sum().backward()
        for old, new in zip(original[:2], copied[:2]):
            self.assertTrue(torch.equal(old.grad, new.grad))

    def test_no_grad_mode_and_mixed_type_promotion_remain_exact(self):
        values = list(inputs())
        with torch.no_grad():
            self.assertTrue(torch.equal(candidate()(*values, 16), baseline(*values, 16)))
        values[0] = values[0].double()
        with torch.inference_mode():
            before, after = baseline(*values, 16), candidate()(*values, 16)
        self.assertEqual(after.dtype, torch.float64)
        self.assertTrue(torch.equal(before, after))

    def test_source_contract_and_schedule_are_recorded(self):
        target = Path('src/model/encoder/costvolume/depth_predictor_multiview.py')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'upstream'/target
            source.parent.mkdir(parents=True)
            source.write_text('def forward():\n'+memory_runtime.OLD+'\n', encoding='utf-8')
            destination = root/'candidate'
            memory_runtime.prepare_chunked_source(root/'upstream', destination, 16)
            metadata = json.loads((root/'memory_schedule.json').read_text())
            self.assertEqual(metadata['depth_candidates'], 128)
            self.assertEqual(metadata['depth_chunk'], 16)
            self.assertEqual(metadata['inference_scratch'], 'reuse_warp_and_preallocate_correlation')
            self.assertIn('heterogs_chunked_correlation(', (destination/target).read_text())
            source.write_text('unsupported changed upstream', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Upstream'):
                memory_runtime.prepare_chunked_source(root/'upstream', root/'bad', 16)


if __name__ == '__main__':
    unittest.main()
