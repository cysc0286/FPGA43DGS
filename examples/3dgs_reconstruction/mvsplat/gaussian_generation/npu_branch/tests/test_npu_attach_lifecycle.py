"""Transactional model attachment checks without SDK or full-model inference."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from gaussian_generation.npu_branch.ffn import PointwiseFFN, adapter_spec
from gaussian_generation.npu_branch.fused_heads import torch_fused_heads
from gaussian_generation.npu_branch.runtime import PartitionRuntime
from gaussian_generation.runtime import ModelRuntime


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.refine_unet = nn.Conv2d(3,32,1)
        self.project = nn.Conv2d(3,128,1)
        self.to_gaussians = nn.Sequential(nn.Conv2d(163,8,3,padding=1),nn.GELU(),
                                          nn.Conv2d(8,84,3,padding=1))
        self.to_disparity = nn.Sequential(nn.Conv2d(32,8,3,padding=1),nn.GELU(),
                                          nn.Conv2d(8,2,3,padding=1))

    def forward(self, value):
        refine = self.refine_unet(value)
        packed = torch.cat((refine,value,self.project(value)),dim=1)
        return self.to_gaussians(packed),self.to_disparity(refine)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=nn.Module()
        self.backbone.backbone=nn.Conv2d(3,3,1)
        self.backbone.transformer=nn.Module()
        block=nn.Module()
        block.cross_attn_ffn=nn.Module()
        block.cross_attn_ffn.mlp=nn.Sequential(nn.Linear(3,6),nn.GELU(),nn.Linear(6,3))
        self.backbone.transformer.layers=nn.ModuleList([block])
        self.depth_predictor=Predictor()

    def forward(self, value):
        value=self.backbone.backbone(value)
        batch,channels,height,width=value.shape
        tokens=value.flatten(2).transpose(1,2)
        tokens=self.backbone.transformer.layers[0].cross_attn_ffn.mlp(tokens)
        value=tokens.transpose(1,2).reshape(batch,channels,height,width)
        return self.depth_predictor(value)


def runtime_for(model, fused=True):
    worker=PartitionRuntime.__new__(PartitionRuntime)
    worker.closed=worker.failed=False
    worker.process=worker.reader=worker.arena=worker.log=None
    worker.model_adapters=[]
    worker.model_replacements=[]
    worker.attached_model=None
    worker.parts={
        'transformer_ffn_0':dict(meta=dict(returns_list=False,
            input_shape=[2,3,5,7],output_shape=[2,3,5,7],
            ffn_adapter=adapter_spec('transformer_ffn_0',[2,3,5,7],[2,3,5,7],
                                     [2,3,5,7],[2,3,5,7]))),
        'backbone_cnn':dict(meta=dict(returns_list=False))}
    reference={
        'backbone_cnn':model.backbone.backbone,
        'transformer_ffn_0':PointwiseFFN(model.backbone.transformer.layers[0].cross_attn_ffn.mlp)}
    if fused:
        worker.parts['fused_heads']=dict(meta=dict(returns_list=False))
        reference['fused_heads']=torch_fused_heads(model.depth_predictor.to_gaussians,
                                                  model.depth_predictor.to_disparity)
    worker.test_calls=[]
    def execute(name, value):
        worker.test_calls.append(name)
        return reference[name](torch.from_numpy(value)).detach().numpy()
    worker.execute=execute
    return worker


class AttachLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(22)
        self.model=Model().eval()
        self.value=torch.randn(2,3,5,7)
        self.originals=(self.model.backbone.transformer.layers[0].cross_attn_ffn.mlp,
                        self.model.backbone.backbone,
                        self.model.depth_predictor.to_gaussians,
                        self.model.depth_predictor.to_disparity)

    def assert_restored(self, worker):
        actual=(self.model.backbone.transformer.layers[0].cross_attn_ffn.mlp,
                self.model.backbone.backbone,
                self.model.depth_predictor.to_gaussians,
                self.model.depth_predictor.to_disparity)
        for a,b in zip(actual,self.originals): self.assertIs(a,b)
        self.assertEqual(worker.model_adapters,[])
        self.assertEqual(worker.model_replacements,[])
        self.assertIsNone(worker.attached_model)
        self.assertEqual(len(self.model.depth_predictor._forward_pre_hooks),0)
        self.assertEqual(len(self.model.depth_predictor.refine_unet._forward_hooks),0)

    def test_close_restores_ffn_ordinary_and_fused_then_cpu_runs(self):
        worker=runtime_for(self.model)
        with torch.inference_mode():
            expected=self.model(self.value)
            worker.attach(self.model)
            actual=self.model(self.value)
            for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
            self.assertEqual(worker.test_calls,['backbone_cnn','transformer_ffn_0','fused_heads'])
            worker.close();worker.close()
            self.assert_restored(worker)
            worker.execute=lambda *_:self.fail('CPU called a closed worker')
            actual=self.model(self.value)
            for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_ordinary_replacement_failure_rolls_back_already_replaced_ffn(self):
        worker=runtime_for(self.model)
        with patch('gaussian_generation.npu_branch.runtime.replace',
                   side_effect=RuntimeError('ordinary attach failed')):
            with self.assertRaisesRegex(RuntimeError,'ordinary attach failed'):
                worker.attach(self.model)
        self.assert_restored(worker)
        with torch.inference_mode():self.model(self.value)
        worker.close()

    def test_late_fused_failure_rolls_back_ffn_and_ordinary(self):
        worker=runtime_for(self.model)
        with patch('gaussian_generation.npu_branch.fused_heads.attach_fused_heads',
                   side_effect=RuntimeError('fused attach failed')):
            with self.assertRaisesRegex(RuntimeError,'fused attach failed'):
                worker.attach(self.model)
        self.assert_restored(worker)
        worker.attach(self.model)
        with torch.inference_mode():self.model(self.value)
        worker.close()
        self.assert_restored(worker)

    def test_close_then_fresh_worker_can_reattach_same_model(self):
        for _ in range(2):
            worker=runtime_for(self.model)
            worker.attach(self.model)
            with torch.inference_mode():self.model(self.value)
            worker.close()
            self.assert_restored(worker)

    def test_second_attach_and_closed_worker_are_rejected_without_mutation(self):
        worker=runtime_for(self.model)
        worker.attach(self.model)
        with self.assertRaisesRegex(ValueError,'already attached'):
            worker.attach(self.model)
        worker.close()
        with self.assertRaisesRegex(RuntimeError,'closed'):
            worker.attach(self.model)
        self.assert_restored(worker)

    def test_conflicting_fused_and_separate_selection_rejected_before_attach(self):
        worker=runtime_for(self.model)
        worker.parts['to_gaussians']=dict(meta=dict(returns_list=False))
        with self.assertRaisesRegex(ValueError,'mutually exclusive'):
            worker.attach(self.model)
        self.assert_restored(worker)
        worker.close()

    def test_public_detach_preserves_cpu_model_and_allows_next_candidate(self):
        runtime=ModelRuntime.__new__(ModelRuntime)
        runtime.model=self.model
        runtime.partition_runtime=None
        with torch.inference_mode():
            expected=self.model(self.value)
            for _ in range(2):
                worker=runtime_for(self.model)
                runtime.attach_partitions(worker)
                runtime.model(self.value)
                runtime.detach_partitions()
                runtime.detach_partitions()  # No worker is left to close twice.
                self.assertIsNone(runtime.partition_runtime)
                self.assertTrue(worker.closed)
                self.assert_restored(worker)
                actual=runtime.model(self.value)
                for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_public_detach_does_not_clear_worker_when_close_raises(self):
        runtime=ModelRuntime.__new__(ModelRuntime)
        runtime.model=self.model
        worker=runtime_for(self.model)
        runtime.partition_runtime=None
        runtime.attach_partitions(worker)
        with patch.object(worker,'close',side_effect=RuntimeError('restoration failed')):
            with self.assertRaisesRegex(RuntimeError,'restoration failed'):
                runtime.detach_partitions()
        self.assertIs(runtime.partition_runtime,worker)
        runtime.detach_partitions()
        self.assertIsNone(runtime.partition_runtime)
        self.assert_restored(worker)

    def test_public_detach_rejects_closed_but_unrestored_worker(self):
        runtime=ModelRuntime.__new__(ModelRuntime)
        runtime.model=self.model
        worker=runtime_for(self.model)
        runtime.partition_runtime=None
        runtime.attach_partitions(worker)
        worker.closed=True  # Simulate close failing after its first assignment.
        with self.assertRaisesRegex(RuntimeError,'restoration is incomplete'):
            runtime.detach_partitions()
        self.assertIs(runtime.partition_runtime,worker)
        # Explicit recovery for this synthetic partial-close test only.
        worker.closed=False
        runtime.detach_partitions()
        self.assert_restored(worker)


if __name__=='__main__':
    unittest.main()
