"""Context-only appearance baking: coverage, immutable geometry and bad frames."""
import struct
import unittest

import numpy as np

from gaussian_generation.appearance import bake_context_residual
from gaussian_generation.blending import AngularFeathering
from gaussian_generation.test_ring_blending import camera
from rendering.resident_runtime import framebuffer_linear


class AppearanceTests(unittest.TestCase):
    def fixture(self):
        g=dict(means=np.array([[0,0,2],[0,0,-2]],np.float32),
               covariances=np.repeat(np.eye(3,dtype=np.float32)[None]*.01,2,axis=0),
               harmonics=np.zeros((2,3,25),np.float32),opacities=np.full(2,.8,np.float32))
        view=camera(0)
        feather=AngularFeathering([[0,0,1]],[view])
        return g,view,feather

    def test_covered_colour_error_is_baked_but_geometry_and_unobserved_rays_are_untouched(self):
        g,view,f=self.fixture();before={k:v.copy() for k,v in g.items()}
        rgb=np.full((128,128,3),.2,np.float32)
        coverage=np.full((128,128),.8,np.float32)
        target=np.full((128,128,3),.36,np.float32)
        out,record=bake_context_residual(g,[view],f,lambda _:target,lambda _: (rgb,coverage),strength=.5)
        # A .2 radiance correction at half strength produces .08 pixel gain
        # for coverage .8. It must not be multiplied by coverage a second time.
        expected=.08
        pixel_gain=.28209479177387814*out['harmonics'][0,:,0]*.8
        np.testing.assert_allclose(pixel_gain,expected,rtol=2e-6,atol=1e-7)
        np.testing.assert_array_equal(out['harmonics'][1],g['harmonics'][1])
        np.testing.assert_array_equal(out['harmonics'][:,:,1:],g['harmonics'][:,:,1:])
        for k in g:
            np.testing.assert_array_equal(g[k],before[k])
            if k!='harmonics':self.assertIs(out[k],g[k])
        self.assertEqual(record['contexts'],1)
        self.assertEqual(record['policy'],'fixed_centre_context_residual_dc_v1')

    def test_zero_coverage_cannot_create_scene_content_and_zero_residual_is_identity(self):
        for empty in (False,True):
            g,v,f=self.fixture();rgb=np.zeros((128,128,3),np.float32)
            coverage=np.zeros((128,128),np.float32) if empty else np.ones((128,128),np.float32)
            target=np.ones_like(rgb) if empty else rgb.copy()
            out,_=bake_context_residual(g,[v],f,lambda _:target,lambda _: (rgb,coverage))
            np.testing.assert_array_equal(out['harmonics'],g['harmonics'])

    def test_invalid_frame_cannot_return_a_calibrated_scene_or_modify_original(self):
        g,v,f=self.fixture();before=g['harmonics'].copy()
        for bad in (float('nan'),-1.,1.01):
            with self.assertRaises(ValueError):
                bake_context_residual(g,[v],f,lambda _:np.zeros((128,128,3)),
                    lambda _: (np.zeros((128,128,3)),np.full((128,128),bad)))
        with self.assertRaises(ValueError):
            bake_context_residual(g,[v],f,lambda _:np.zeros((64,64,3)),
                lambda _: (np.zeros((128,128,3)),np.ones((128,128))))
        np.testing.assert_array_equal(g['harmonics'],before)

    def test_bad_strength_and_mismatched_source_camera_are_rejected(self):
        g,v,f=self.fixture()
        for strength in (True,-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):
                bake_context_residual(g,[v],f,lambda _:None,lambda _:None,strength=strength)
        with self.assertRaises(ValueError):
            bake_context_residual(g,[camera(10)],f,lambda _:None,lambda _:None)

    def test_frozen_fourth_float_is_transmittance_and_status_is_not_float_data(self):
        raw=b'GSSOUT01'+struct.pack('<II',1,1)+struct.pack('<4fI',.2,.3,.4,.25,0xffffffff)
        rgb,coverage=framebuffer_linear(raw)
        np.testing.assert_allclose(rgb,[[[.2,.3,.4]]],atol=1e-7)
        np.testing.assert_allclose(coverage,[[.75]])
        for bad in (raw[:-1],raw[:16]+struct.pack('<4fI',.2,.3,.4,-.1,0)):
            with self.assertRaises(ValueError):framebuffer_linear(bad)


if __name__=='__main__':unittest.main()
