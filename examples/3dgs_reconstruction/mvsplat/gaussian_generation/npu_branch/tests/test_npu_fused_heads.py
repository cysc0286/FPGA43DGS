"""Small synthetic interface/equivalence checks; no hardware performance claim."""
import unittest
import tempfile
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from gaussian_generation.npu_branch.fused_heads import attach_fused_heads, split_output, torch_fused_heads


@unittest.skipIf(torch is None, 'Torch is required for small interface checks')
class FusedHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(1)
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.refine_unet = torch.nn.Conv2d(3,2,1)
                self.to_gaussians = torch.nn.Sequential(torch.nn.Conv2d(5,4,3,padding=1),
                    torch.nn.GELU(),torch.nn.Conv2d(4,3,3,padding=1))
                self.to_disparity = torch.nn.Sequential(torch.nn.Conv2d(2,4,3,padding=1),
                    torch.nn.GELU(),torch.nn.Conv2d(4,1,3,padding=1))
            def forward(self, x, mode='normal'):
                refine=self.refine_unet(x)
                joined=torch.cat((refine,x),dim=1)
                gaussian=self.to_gaussians(joined)
                if mode == 'skip': return gaussian
                if mode == 'duplicate': self.to_gaussians(joined)
                if mode == 'copy': refine=refine.clone()
                if mode == 'mutate': refine.add_(.125)
                return gaussian,self.to_disparity(refine)
        self.model=Toy().eval()
        self.fused=torch_fused_heads(self.model.to_gaussians,self.model.to_disparity,2)
        self.calls=0
        def execute(value):
            self.calls+=1
            return self.fused(torch.from_numpy(value)).detach().numpy()
        self.execute=execute

    def attach(self, execute=None):
        return attach_fused_heads(self.model,execute or self.execute,input_channels=5,
                                  refine_channels=2,gaussian_channels=3,disparity_channels=1)

    def test_two_frames_exact_and_cache_consumed(self):
        inputs=[torch.randn(2,3,9,9),torch.randn(2,3,9,9)]
        with torch.inference_mode():
            expected=[self.model(x) for x in inputs]
            with self.attach() as adapter:
                for x,reference in zip(inputs,expected):
                    actual=self.model(x)
                    self.assertTrue(all(torch.equal(a,b) for a,b in zip(actual,reference)))
                    self.assertIsNone(adapter.pending)
                    self.assertFalse(adapter.active)
        self.assertEqual(self.calls,2)

    def test_close_restores_original_modules_and_hooks(self):
        original=(self.model.to_gaussians,self.model.to_disparity)
        adapter=self.attach()
        adapter.close();adapter.close()
        self.assertIs(self.model.to_gaussians,original[0])
        self.assertIs(self.model.to_disparity,original[1])
        self.assertEqual(len(self.model._forward_pre_hooks),0)
        self.assertEqual(len(self.model.refine_unet._forward_hooks),0)

    def test_refinement_copy_rejected_even_when_values_match(self):
        with self.attach(),torch.inference_mode():
            with self.assertRaisesRegex(ValueError,'mismatched'):
                self.model(torch.randn(1,3,7,7),mode='copy')

    def test_inference_tensor_inplace_edit_rejected(self):
        with self.attach(),torch.inference_mode():
            with self.assertRaisesRegex(ValueError,'mismatched'):
                self.model(torch.randn(1,3,7,7),mode='mutate')

    def test_skip_disparity_cannot_reuse_next_frame(self):
        with self.attach(),torch.inference_mode():
            with self.assertRaisesRegex(RuntimeError,'consumed exactly once'):
                self.model(torch.randn(1,3,7,7),mode='skip')
            with self.assertRaisesRegex(RuntimeError,'unfinished'):
                self.model(torch.randn(1,3,7,7))

    def test_duplicate_gaussian_call_rejected(self):
        with self.attach(),torch.inference_mode():
            with self.assertRaisesRegex(ValueError,'bound inference frame'):
                self.model(torch.randn(1,3,7,7),mode='duplicate')

    def test_nonfinite_executor_output_rejected(self):
        def bad(value):
            return np.full((value.shape[0],4,*value.shape[2:]),np.nan,np.float32)
        with self.attach(bad),torch.inference_mode():
            with self.assertRaisesRegex(ValueError,'Invalid fused'):
                self.model(torch.randn(1,3,7,7))

    def test_double_attachment_rejected(self):
        with self.attach():
            with self.assertRaisesRegex(ValueError,'already attached'):
                self.attach()

    def test_small_disparity_cache_is_owning_copy(self):
        source=np.arange(2*4*7*7,dtype=np.float32).reshape(2,4,7,7)
        gaussian,disparity=split_output(source,3,1)
        self.assertTrue(np.shares_memory(gaussian,source))
        self.assertFalse(np.shares_memory(disparity,source))
        self.assertEqual(disparity.nbytes,2*1*7*7*4)

    def test_composed_onnx_tiles_match_both_heads_at_all_edges(self):
        try:
            import onnx
            import onnxruntime
        except ImportError:
            self.skipTest('ONNX test dependencies unavailable')
        from gaussian_generation.npu_branch.fused_heads_export import fused_model
        from gaussian_generation.npu_branch.head_probe import _session
        from gaussian_generation.npu_branch.spatial import execute_spatial
        with tempfile.TemporaryDirectory() as directory,torch.inference_mode():
            root=Path(directory)
            value=torch.randn(2,5,19,19)
            for name,module,x in [('gaussian',self.model.to_gaussians,value),
                                  ('disparity',self.model.to_disparity,value[:,:2])]:
                torch.onnx.export(module,(x,),str(root/(name+'.onnx')),
                    input_names=['features'],output_names=['result'],
                    opset_version=17,dynamo=False)
            source=[onnx.load(str(root/(name+'.onnx'))) for name in ('gaussian','disparity')]
            combined=fused_model(*source,batch=1,size=9,refine_channels=2)
            initializers={x.name:x for x in combined.graph.initializer}
            for prefix,model in zip(('gaussian/','disparity/'),source):
                for initializer in model.graph.initializer:
                    self.assertTrue(np.array_equal(onnx.numpy_helper.to_array(initializer),
                        onnx.numpy_helper.to_array(initializers[prefix+initializer.name])))
            meta=dict(input_shape=[1,5,9,9],output_shape=[1,4,9,9],
                spatial_tiling=dict(schema='mvsplat_local_head_tiles_v1',halo=2,core_size=5,
                    input_shape=[2,5,19,19],output_shape=[2,4,19,19]))
            session=_session(combined)
            actual=execute_spatial(value.numpy(),meta,
                lambda tile:session.run(None,{'features':tile})[0])
            expected=self.fused(value).numpy()
            self.assertTrue(np.allclose(actual,expected,rtol=2e-5,atol=2e-6))


if __name__=='__main__':
    unittest.main()
