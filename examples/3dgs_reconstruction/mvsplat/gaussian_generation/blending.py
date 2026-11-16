"""Source-frustum feathering for a fixed-centre scene; no cross-view depth fit."""
import numpy as np

from gaussian_generation.sampling import _source_geometry


SOURCE_FEATHER = 'observed_frustum_smoothstep_optical_depth_v1'
MIN_SOURCE_WEIGHT = .001


def _smoothstep(t):
    t = np.clip(t,0,1)
    return t*t*(3-2*t)


class AngularFeathering:
    """Partition observed angular overlap without a nearest-axis hard switch.

    Camera eligibility uses the actual cropped source intrinsics and rotation.
    Relative optical-axis angle supplies a narrow smooth overlap, while image
    edge confidence reduces abrupt changes at the cropped frustum boundaries.
    Unsupported directions receive zero weight, never extrapolated coverage.
    """
    def __init__(self,directions,cameras,transition_degrees=8.,edge_margin=.04):
        if (type(transition_degrees) not in (int,float) or not np.isfinite(transition_degrees)
                or not 0<transition_degrees<=12
                or type(edge_margin) not in (int,float) or not np.isfinite(edge_margin)
                or not 0<=edge_margin<=.25):
            raise ValueError('Invalid angular transition or source edge margin')
        axes = np.asarray(directions,np.float64)
        if (axes.ndim!=2 or axes.shape[1]!=3 or not 1<=len(axes)<=16
                or len(cameras)!=len(axes) or not np.isfinite(axes).all()
                or not np.allclose(np.linalg.norm(axes,axis=1),1,atol=1e-5)):
            raise ValueError('Invalid source axes or camera count')
        self.axes = axes.copy()
        self.band,self.edge = float(transition_degrees),float(edge_margin)
        self.cameras = []
        for axis,view in zip(axes,cameras):
            rotation,_,_ = _source_geometry(view)
            if not np.allclose(rotation[:,2],axis,atol=1e-6):
                raise ValueError('Source frustum differs from its angular axis')
            k = np.asarray(view['intrinsics_normalized'],np.float64)
            if not np.all((k[:2,2]>=0)&(k[:2,2]<=1)):
                raise ValueError('Source principal point is outside the cropped image')
            self.cameras.append((rotation.copy(),np.diag(k)[:2].copy(),k[:2,2].copy()))

    def weights(self,points):
        points = np.asarray(points)
        if (points.ndim!=2 or points.shape[1]!=3 or points.dtype.kind!='f'
                or not np.isfinite(points).all()):
            raise ValueError('Angular feathering requires finite Gaussian means')
        rays = points/np.maximum(np.linalg.norm(points,axis=1,keepdims=True),1e-30)
        angle = np.degrees(np.arccos(np.clip(rays@self.axes.T,-1,1)))
        support = np.zeros(angle.shape,bool)
        confidence = np.zeros_like(angle)
        for i,(rotation,focal,principal) in enumerate(self.cameras):
            camera = rays@rotation
            positive = camera[:,2]>1e-8
            uv = camera[:,:2]/np.maximum(camera[:,2,None],1e-8)*focal+principal
            distance = np.min(np.column_stack([uv,1-uv]),axis=1)
            support[:,i] = positive&(distance>0)
            confidence[:,i] = (_smoothstep(distance/self.edge)*positive
                               if self.edge else support[:,i])
        best = np.where(support,angle,np.inf).min(axis=1)
        finite = np.isfinite(best)
        delta = np.full_like(angle,np.inf)
        delta[finite] = angle[finite]-best[finite,None]
        weights = _smoothstep(1-delta/self.band)*confidence
        weights /= np.maximum(weights.sum(axis=1,keepdims=True),1e-30)
        weights[weights<MIN_SOURCE_WEIGHT] = 0
        weights /= np.maximum(weights.sum(axis=1,keepdims=True),1e-30)
        return weights


def attenuate_opacity(opacity,weights):
    """Split peak optical depth rather than linearly scale an opaque peak.

    This preserves total transmittance for coincident constant-colour samples
    whose weights sum to one. Gaussian footprints and ordered multi-depth
    composition are nonlinear, so general images remain an approximation.
    """
    opacity,weights = np.asarray(opacity),np.asarray(weights)
    if (opacity.ndim!=1 or weights.shape!=opacity.shape
            or not np.isfinite(opacity).all() or not np.isfinite(weights).all()
            or np.any(opacity<0) or np.any(opacity>1)
            or np.any(weights<0) or np.any(weights>1)):
        raise ValueError('Invalid opacity or angular blending weights')
    return (-np.expm1(weights*np.log1p(-np.minimum(opacity,.99)))).astype(np.float32)
