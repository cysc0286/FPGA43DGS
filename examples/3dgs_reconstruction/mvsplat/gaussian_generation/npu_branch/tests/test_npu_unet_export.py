"""Small export-scope tests; never load or benchmark the full MVSplat model."""
import types
import unittest

import numpy as np
import torch

from gaussian_generation.npu_branch.unet_export import (
    bypass_attention_checkpoint, numerical_comparison, output_wrapper,
)


class AttentionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.]))
        self.wrapper_calls = 0

    def forward(self, value):
        self.wrapper_calls += 1
        return self._forward(value)

    def _forward(self, value):
        return value*self.weight + value


class ExportScopeTests(unittest.TestCase):
    def make(self):
        return torch.nn.Sequential(AttentionBlock()).eval()

    def test_same_forward_and_restores_class_method(self):
        model = self.make()
        value = torch.arange(8,dtype=torch.float32).reshape(1,2,2,2)
        expected = model(value)
        weight = model[0].weight.detach().clone()
        with bypass_attention_checkpoint(model) as patched:
            self.assertEqual(patched, ['0'])
            self.assertTrue(torch.equal(model(value), expected))
            self.assertEqual(model[0].wrapper_calls,1)
        self.assertNotIn('forward',model[0].__dict__)
        self.assertTrue(torch.equal(model(value),expected))
        self.assertEqual(model[0].wrapper_calls,2)
        self.assertTrue(torch.equal(model[0].weight,weight))

    def test_restores_after_exception(self):
        model = self.make()
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with bypass_attention_checkpoint(model):
                raise RuntimeError('deliberate')
        self.assertNotIn('forward',model[0].__dict__)

    def test_restores_existing_instance_forward(self):
        model=self.make()
        def instance(self, value):
            return self._forward(value)
        previous=types.MethodType(instance,model[0])
        model[0].forward=previous
        with bypass_attention_checkpoint(model):
            self.assertIsNot(model[0].forward,previous)
        self.assertIs(model[0].forward,previous)

    def test_nested_context_restores_enclosing_patch(self):
        model=self.make()
        with bypass_attention_checkpoint(model):
            enclosing=model[0].forward
            with bypass_attention_checkpoint(model):
                self.assertIsNot(model[0].forward,enclosing)
            self.assertIs(model[0].forward,enclosing)
        self.assertNotIn('forward',model[0].__dict__)

    def test_rejects_training_and_does_not_mutate(self):
        model=self.make().train()
        with self.assertRaisesRegex(ValueError,'eval-only'):
            with bypass_attention_checkpoint(model):
                pass
        self.assertNotIn('forward',model[0].__dict__)

    def test_rejects_prefix_without_attention(self):
        model=torch.nn.Conv2d(2,2,1).eval()
        with self.assertRaisesRegex(ValueError,'no checkpoint'):
            with bypass_attention_checkpoint(model):
                pass

    def test_explicit_nhwc_result_has_correct_axis_order(self):
        model=self.make()
        value=torch.arange(24,dtype=torch.float32).reshape(1,2,3,4)
        actual=output_wrapper(model,'NHWC')(value)
        self.assertEqual(tuple(actual.shape),(1,3,4,2))
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(torch.equal(actual.permute(0,3,1,2),model(value)))
        self.assertTrue(torch.equal(output_wrapper(model,'NCHW')(value),model(value)))

    def test_comparison_detects_nonfinite_and_shape(self):
        check=numerical_comparison(np.array([np.inf]),np.array([1.]),rtol=1e-5,atol=1e-6)
        self.assertFalse(check['passed'])
        self.assertFalse(check['finite'])
        with self.assertRaisesRegex(ValueError,'shape'):
            numerical_comparison(np.ones(2),np.ones(3),rtol=1e-5,atol=1e-6)

    def test_comparison_reports_strict_failure(self):
        value=np.array([1.,2.],dtype=np.float32)
        self.assertTrue(numerical_comparison(value,value,rtol=1e-5,atol=1e-6)['exact'])
        check=numerical_comparison(value+.01,value,rtol=1e-5,atol=1e-6)
        self.assertFalse(check['passed'])
        self.assertEqual(check['outside_tolerance_fraction'],1.)


if __name__=='__main__':
    unittest.main()
