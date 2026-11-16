"""One context-only, bounded appearance pass for an observed fixed-centre ring.

This is not bilateral-grid training or a depth correction. A render callback
supplies premultiplied RGB and coverage, so the board can use its resident FPGA
renderer without importing the slow numerical reference into the live path.
"""
import time

import numpy as np

from common import validate_gaussians
from gaussian_generation.sampling import _source_geometry


APPEARANCE_BAKE = 'fixed_centre_context_residual_dc_v1'
C0 = .28209479177387814


def _blur(value, sigma):
    """Separable Gaussian with half-sample symmetric edges; NumPy only."""
    radius = int(4*sigma+.5)
    x = np.arange(-radius,radius+1,dtype=np.float64)
    kernel = np.exp(-.5*(x/sigma)**2)
    kernel /= kernel.sum()
    result = np.asarray(value,np.float64)
    for axis in (0,1):
        padding = [(0,0)]*result.ndim
        padding[axis] = (radius,radius)
        padded = np.pad(result,padding,mode='symmetric')
        output = np.zeros_like(result)
        for offset,weight in enumerate(kernel):
            select = [slice(None)]*result.ndim
            select[axis] = slice(offset,offset+result.shape[axis])
            output += weight*padded[tuple(select)]
        result = output
    return result


def _sample(image,u,v):
    u,v = np.clip(u,0,image.shape[1]-1),np.clip(v,0,image.shape[0]-1)
    x,y = u.astype(np.intp),v.astype(np.intp)
    xx,yy = np.minimum(x+1,image.shape[1]-1),np.minimum(y+1,image.shape[0]-1)
    a,b = (u-x)[:,None],(v-y)[:,None]
    return ((1-b)*((1-a)*image[y,x]+a*image[y,xx])+
            b*((1-a)*image[yy,x]+a*image[yy,xx]))


def bake_context_residual(gaussians,cameras,feather,read_context,render_context,*,
                          strength=1.,sigma=2.,limit=.2,minimum_coverage=.6):
    """Bake shared low-frequency RGB residuals into SH DC coefficients once.

    Both callbacks receive an actual context camera. ``read_context`` returns
    normalized RGB; ``render_context`` returns linear premultiplied RGB and
    coverage (1 minus transmittance), never display-rounded bytes. The same
    original scene must be used for every render in the pass. Targets are not
    part of this interface. Geometry, opacity and higher SH terms stay intact.
    """
    if (type(strength) not in (int,float) or not np.isfinite(strength) or not 0<=strength<=1
            or type(sigma) not in (int,float) or not np.isfinite(sigma) or not .5<=sigma<=4
            or type(limit) not in (int,float) or not np.isfinite(limit) or not 0<limit<=.2
            or type(minimum_coverage) not in (int,float) or not np.isfinite(minimum_coverage)
            or not .5<=minimum_coverage<1):
        raise ValueError('Invalid bounded appearance controls')
    g = validate_gaussians(gaussians)
    if any(v.dtype!=np.float32 for v in g.values()):
        raise ValueError('Appearance baking requires original FP32 Gaussian fields')
    if len(cameras)!=len(feather.axes):
        raise ValueError('Appearance source count differs from angular support')
    for i,view in enumerate(cameras):
        rotation,_,_ = _source_geometry(view)
        k = np.asarray(view['intrinsics_normalized'],np.float64)
        frustum = feather.cameras[i]
        if (not np.allclose(rotation,frustum[0],atol=1e-7,rtol=0)
                or not np.allclose(np.diag(k)[:2],frustum[1],atol=1e-7,rtol=0)
                or not np.allclose(k[:2,2],frustum[2],atol=1e-7,rtol=0)):
            raise ValueError('Appearance camera differs from the observed source frustum')
    started = time.monotonic()
    rays = np.asarray(g['means'],np.float64)
    rays = rays/np.maximum(np.linalg.norm(rays,axis=1,keepdims=True),1e-30)
    weights = feather.weights(rays)
    correction = np.zeros((len(rays),3),np.float64)
    records = []
    for i,view in enumerate(cameras):
        target = np.asarray(read_context(view),np.float64)
        rgb,coverage = render_context(view)
        rgb,coverage = np.asarray(rgb,np.float64),np.asarray(coverage,np.float64)
        h,w = view['height'],view['width']
        if (target.shape!=(h,w,3) or rgb.shape!=target.shape or coverage.shape!=(h,w)
                or not np.isfinite(target).all() or not np.isfinite(rgb).all()
                or not np.isfinite(coverage).all() or np.any(coverage<0) or np.any(coverage>1)
                or np.any(target<0) or np.any(target>1) or np.any(rgb<0)):
            raise ValueError('Invalid source colour, premultiplied RGB or coverage')
        valid = coverage>minimum_coverage
        residual = np.clip((target-rgb)/np.maximum(coverage[:,:,None],.5),-limit,limit)
        residual = _blur(residual*valid[:,:,None],sigma)/np.maximum(
            _blur(valid.astype(np.float64),sigma)[:,:,None],1e-6)
        local = rays@np.asarray(view['c2w'],np.float64)[:3,:3]
        k = np.asarray(view['intrinsics_normalized'],np.float64)
        u = (k[0,0]*local[:,0]/np.maximum(local[:,2],1e-8)+k[0,2])*w-.5
        v = (k[1,1]*local[:,1]/np.maximum(local[:,2],1e-8)+k[1,2])*h-.5
        correction += _sample(residual,u,v)*weights[:,i,None]
        records.append(dict(source_index=view.get('index'),valid_pixels=int(valid.sum()),
                            input_mae_before=float(np.mean(abs(target-rgb)))))
    harmonics = g['harmonics'].copy()
    harmonics[:,:,0] += (strength*correction/C0).astype(np.float32)
    result = dict(g,harmonics=harmonics)
    validate_gaussians(result)
    return result,dict(policy=APPEARANCE_BAKE,contexts=len(cameras),strength=float(strength),
        sigma_pixels=float(sigma),radiance_correction_limit=float(limit),
        minimum_coverage=float(minimum_coverage),seconds=time.monotonic()-started,
        max_applied_radiance_correction=float(np.max(abs(strength*correction))),
        contexts_record=records,targets_used=False,per_frame_extra_passes=0,
        scope='Fixed-centre appearance correction; no geometry recovery or missing-content fill')
