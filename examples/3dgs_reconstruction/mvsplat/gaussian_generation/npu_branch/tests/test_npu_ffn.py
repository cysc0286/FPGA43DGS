"""Small FFN boundary tests; no checkpoint, board or full-model inference."""
import copy
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from gaussian_generation.npu_branch.ffn import (
    FFN_PARTITIONS, PointwiseFFN, TokenFFNProxy, adapter_spec, attach_ffn_partitions,
    execute_pointwise, image_to_token, locate_ffn, token_to_image, validate_adapter,
)


def mlp(bias=False):
    return nn.Sequential(nn.Linear(6, 12, bias=bias), nn.GELU(), nn.Linear(12, 3, bias=bias)).eval()


def meta(name='transformer_ffn_0', tile=(2,6,5,7)):
    output=[tile[0],3,tile[2],tile[3]]
    return dict(input_shape=list(tile), output_shape=output,
        ffn_adapter=adapter_spec(name,[2,6,5,7],[2,3,5,7],list(tile),output))


class FfnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_original_weight_and_bias_preserved(self):
        for biased in (False, True):
            original=mlp(biased)
            wrapped=PointwiseFFN(original)
            x=torch.randn(2,35,6)
            image=token_to_image(x,5,7,6)
            actual=image_to_token(wrapped(image))
            torch.testing.assert_close(actual,original(x),rtol=2e-5,atol=2e-6)
            self.assertTrue(torch.equal(wrapped.first.weight[:,:,0,0],original[0].weight))
            self.assertTrue(torch.equal(wrapped.second.weight[:,:,0,0],original[2].weight))
            self.assertEqual(wrapped.first.weight.data_ptr(),original[0].weight.data_ptr())
            self.assertIsNone(wrapped.first.bias) if not biased else self.assertTrue(
                torch.equal(wrapped.first.bias,original[0].bias))

    def test_tanh_gelu_not_silently_replaced(self):
        original=mlp()
        original[1]=nn.GELU(approximate='tanh')
        wrapped=PointwiseFFN(original)
        self.assertEqual(wrapped.activation.approximate,'tanh')
        x=torch.randn(2,35,6)
        torch.testing.assert_close(image_to_token(wrapped(token_to_image(x,5,7,6))),original(x))

    def test_noncontiguous_token_order_is_preserved(self):
        x=torch.arange(2*70*6,dtype=torch.float32).reshape(2,70,6)[:,::2]
        torch.testing.assert_close(image_to_token(token_to_image(x,5,7,6)),x,rtol=0,atol=0)

    def test_one_view_and_odd_tail_tiles_keep_every_token(self):
        original=mlp()
        wrapped=PointwiseFFN(original)
        x=np.random.default_rng(1).normal(size=(2,6,5,7)).astype(np.float32)
        calls=[]
        def execute(value):
            calls.append(value.shape)
            return wrapped(torch.from_numpy(value)).detach().numpy()
        actual=execute_pointwise(x,meta(tile=(1,6,3,4)),execute)
        expected=wrapped(torch.from_numpy(x)).detach().numpy()
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
        self.assertEqual(len(calls),8)
        self.assertEqual(set(calls),{(1,6,3,4)})

    def test_full_shape_is_one_call(self):
        x=np.zeros((2,6,5,7),np.float32)
        calls=[]
        def execute(value):
            calls.append(value)
            return np.zeros((2,3,5,7),np.float32)
        execute_pointwise(x,meta(),execute)
        self.assertEqual(len(calls),1)

    def test_bad_shapes_dtype_or_nonfinite_rejected(self):
        def execute(value):
            return np.zeros((2,3,5,7),np.float32)
        for x in (np.zeros((1,6,5,7),np.float32),np.zeros((2,6,5,7),np.float64),
                  np.full((2,6,5,7),np.nan,np.float32)):
            with self.assertRaises(ValueError):execute_pointwise(x,meta(),execute)
        with self.assertRaises(ValueError):
            execute_pointwise(np.zeros((2,6,5,7),np.float32),meta(),
                              lambda x:np.full((2,3,5,7),np.inf,np.float32))

    def test_nonzero_halo_and_wrong_layer_paths_rejected(self):
        for key,value in (('halo',2),('path','backbone.backbone'),('partition','other')):
            bad=copy.deepcopy(meta())
            bad['ffn_adapter'][key]=value
            with self.assertRaises(ValueError):validate_adapter(bad)

    def test_wrong_mlp_structure_not_approximated(self):
        with self.assertRaises(ValueError):PointwiseFFN(nn.Sequential(nn.Linear(6,3)))
        with self.assertRaises(ValueError):PointwiseFFN(nn.Sequential(nn.Linear(6,12),nn.ReLU(),nn.Linear(12,3)))

    def test_six_layers_keep_independent_weights_and_token_abi(self):
        model=nn.Module()
        model.backbone=nn.Module()
        model.backbone.transformer=nn.Module()
        model.backbone.transformer.layers=nn.ModuleList()
        originals={}
        for i,name in enumerate(FFN_PARTITIONS):
            block=nn.Module()
            block.cross_attn_ffn=nn.Module()
            block.cross_attn_ffn.mlp=mlp()
            with torch.no_grad():block.cross_attn_ffn.mlp[0].weight.add_(i*.2)
            model.backbone.transformer.layers.append(block)
            originals[name]=PointwiseFFN(block.cross_attn_ffn.mlp)
        class FakeRuntime:
            parts={name:dict(meta=meta(name,tile=(1,6,5,7))) for name in FFN_PARTITIONS}
            def execute(self,name,value):
                return originals[name](torch.from_numpy(value)).detach().numpy()
        runtime=FakeRuntime()
        self.assertEqual(attach_ffn_partitions(model,runtime),list(FFN_PARTITIONS))
        x=torch.randn(2,35,6)
        for name in FFN_PARTITIONS:
            proxy=locate_ffn(model,name)
            self.assertIsInstance(proxy,TokenFFNProxy)
            expected=image_to_token(originals[name](token_to_image(x,5,7,6)))
            torch.testing.assert_close(proxy(x),expected)
        self.assertFalse(torch.equal(locate_ffn(model,'transformer_ffn_0')(x),
                                     locate_ffn(model,'transformer_ffn_5')(x)))

    def test_attach_validates_all_before_replacing_any(self):
        model=SimpleNamespace(backbone=SimpleNamespace(transformer=SimpleNamespace(layers=nn.ModuleList())))
        for name in FFN_PARTITIONS:
            block=nn.Module()
            block.cross_attn_ffn=nn.Module()
            block.cross_attn_ffn.mlp=mlp()
            model.backbone.transformer.layers.append(block)
        original=locate_ffn(model,'transformer_ffn_0')
        parts={name:dict(meta=meta(name)) for name in FFN_PARTITIONS}
        parts['transformer_ffn_5']['meta']['ffn_adapter']['full_input_shape'][1]=999
        with self.assertRaises(ValueError):attach_ffn_partitions(model,SimpleNamespace(parts=parts))
        self.assertIs(locate_ffn(model,'transformer_ffn_0'),original)


if __name__ == '__main__':
    unittest.main()
