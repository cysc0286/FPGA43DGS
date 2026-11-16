"""Bounded spatial retention of row-major, one-Gaussian-per-source-pixel outputs."""
import numpy as np


SPATIAL_SAMPLING = 'source_pixel_stratified_2x2_v1'
BALANCED_SAMPLING = 'avoid_empty_supported_source_rows_columns_v1'
FOOTPRINT_COMPENSATION = 'retention_density_projected_footprint_v1'
MOMENT_MERGE = 'source_projected_neighbour_moments_v1'
SCREEN_VARIANCE = .3  # Matches cached_projection.hpp and the frozen projector.


def _source_geometry(view):
    if view is None:
        raise ValueError('Thinned footprints require actual source camera metadata')
    pose = np.asarray(view['c2w'],np.float64)
    intrinsic = np.asarray(view['intrinsics_normalized'],np.float64)
    h,w = view['height'],view['width']
    if (type(h) is not int or type(w) is not int or min(h,w)<1
            or pose.shape!=(4,4) or intrinsic.shape!=(3,3)
            or not np.isfinite(pose).all() or not np.isfinite(intrinsic).all()
            or not np.allclose(pose[:3,3],0,atol=1e-8)
            or not np.allclose(pose[3],[0,0,0,1],atol=1e-8)
            or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-5)
            or not np.isclose(np.linalg.det(pose[:3,:3]),1,atol=1e-5)
            or intrinsic[0,0]<=0 or intrinsic[1,1]<=0
            or not np.allclose(intrinsic[2],[0,0,1],atol=1e-8)
            or not np.allclose(intrinsic[[0,1],[1,0]],0,atol=1e-8)):
        raise ValueError('Reduction requires valid fixed-centre source cameras')
    return pose[:3,:3],intrinsic[0,0]*w,intrinsic[1,1]*h


def compensate_covariances(gaussians, selected, source_views, owned_counts):
    """Widen retained footprints to compensate for deleted Gaussian samples.

    For a retained fraction p, target 2D covariance is (Sigma2D + 0.3 I)/p,
    including the projector's existing low-pass filter. Scaling only the world
    covariance misses that filter, especially for small/far-away Gaussians.
    Lift the extra filter variance through the source camera's right/down axes.
    This is an approximate image-quality correction for fixed-centre views;
    it does not infer depth, blend groups, or preserve arbitrary fine details.
    Positions, SH colours, opacity and source arrays are never modified.
    """
    selected = np.asarray(selected)
    per_view = len(gaussians['means'])//2
    if (selected.ndim != 1 or selected.dtype.kind not in 'iu'
            or len(gaussians['means']) % 2 or len(source_views) != 2
            or len(owned_counts) != 2
            or np.any(selected < 0) or np.any(selected >= 2*per_view)
            or (len(selected)>1 and np.any(selected[1:]<=selected[:-1]))):
        raise ValueError('Invalid two-view Gaussian retention for footprint compensation')
    covariance = gaussians['covariances'][selected].copy()
    means = gaussians['means'][selected]
    records = []
    for local, (view, count) in enumerate(zip(source_views,owned_counts)):
        mask = selected//per_view==local
        retained = int(np.count_nonzero(mask))
        if type(count) is not int or not retained <= count <= per_view:
            raise ValueError('Invalid pre-thinning ownership count')
        if count and not retained:
            raise ValueError('Cannot compensate a source with no retained Gaussians')
        factor = count/retained if retained else 1.
        records.append(dict(owned_gaussians=count, retained_gaussians=retained,
                            footprint_area_factor=factor))
        if factor <= 1:
            continue
        rotation,fx,fy = _source_geometry(view)
        right,down,forward = rotation.T.astype(np.float32)
        fx,fy = np.float32(fx),np.float32(fy)
        depth = means[mask]@forward
        if np.any(depth<=0):
            raise ValueError('Retained Gaussian is behind its source camera')
        kernel = np.outer(right,right)/(fx*fx)+np.outer(down,down)/(fy*fy)
        covariance[mask] = np.float32(factor)*covariance[mask]+(
            np.float32(SCREEN_VARIANCE*(factor-1))*depth*depth)[:,None,None]*kernel[None]
    if not np.isfinite(covariance).all():
        raise ValueError('Nonfinite compensated Gaussian covariance')
    return covariance,records


def merge_retained_neighbours(gaussians, selected, owned, source_views):
    """Coalesce discarded neighbours rather than discard their colour/coverage.

    Group by nearest retained projected centre within each actual source view.
    Match Gaussian screen-space mass, mean and covariance, then lift the result
    back into the existing world-covariance ABI at the weighted predicted depth.
    This is a local approximation: ordered alpha composition is nonlinear, and
    opacity clipping / averaging may lose fine detail. No target image is used.
    Unthinned views and zero-mass clusters preserve their original parameters.
    """
    from common import validate_gaussians
    validate_gaussians(gaussians)
    selected,owned = np.asarray(selected),np.asarray(owned)
    total = len(gaussians['means'])
    if total % 2 or len(source_views)!=2:
        raise ValueError('Neighbour reduction requires two source blocks')
    for ids in (selected,owned):
        if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or not len(ids)
                or np.any(ids<0) or np.any(ids>=total)
                or (len(ids)>1 and np.any(ids[1:]<=ids[:-1]))):
            raise ValueError('Reduction indices must be ordered unique Gaussian IDs')
    membership = np.searchsorted(owned,selected)
    if np.any(membership>=len(owned)) or not np.array_equal(owned[membership],selected):
        raise ValueError('Retained Gaussians must be a subset of angular ownership')
    result = {key:value[selected].copy() for key,value in gaussians.items()}
    records = []
    for local,view in enumerate(source_views):
        mask = selected//(total//2)==local
        keep = selected[mask]
        available = owned[owned//(total//2)==local]
        if len(available) and not len(keep):
            raise ValueError('Cannot merge a source without a retained representative')
        record = dict(owned_gaussians=len(available),retained_gaussians=len(keep),
                      footprint_area_factor=len(available)/len(keep) if len(keep) else 1.,
                      opacity_clipped_clusters=0,max_neighbour_distance_pixels=0.)
        records.append(record)
        if len(keep)==len(available):
            continue
        from scipy.spatial import cKDTree
        rotation,fx,fy = _source_geometry(view)
        camera = gaussians['means'][available].astype(np.float64)@rotation
        z = camera[:,2]
        if np.any(z<=0):
            raise ValueError('Owned Gaussian is behind its source camera')
        uv = camera[:,:2]/z[:,None]*[fx,fy]
        representatives = np.searchsorted(available,keep)
        distance,owner = cKDTree(uv[representatives]).query(uv,k=1,workers=1)
        # Exact ties must not leave an existing representative with no members.
        owner[representatives] = np.arange(len(keep))
        record['max_neighbour_distance_pixels'] = float(distance.max())
        jacobian = np.zeros((len(z),2,3),np.float64)
        jacobian[:,0,0],jacobian[:,1,1] = fx/z,fy/z
        jacobian[:,0,2],jacobian[:,1,2] = -fx*camera[:,0]/z**2,-fy*camera[:,1]/z**2
        camera_cov = rotation.T[None]@gaussians['covariances'][available].astype(np.float64)@rotation[None]
        projected = jacobian@camera_cov@jacobian.swapaxes(1,2)+SCREEN_VARIANCE*np.eye(2)[None]
        determinant = np.linalg.det(projected)
        if np.any(determinant<=0) or np.any(projected[:,0,0]<=0):
            raise ValueError('Invalid source-projected Gaussian covariance')
        mass = gaussians['opacities'][available]*np.sqrt(determinant)
        sums = np.bincount(owner,weights=mass,minlength=len(keep))

        def aggregate(value):
            flat = value.reshape(len(value),-1)
            averaged = np.empty((len(keep),flat.shape[1]),np.float64)
            for column in range(flat.shape[1]):
                averaged[:,column] = np.bincount(owner,weights=mass*flat[:,column],minlength=len(keep))
            averaged /= np.maximum(sums[:,None],1e-30)
            return averaged.reshape(len(keep),*value.shape[1:])

        center,depth = aggregate(uv),aggregate(z)
        zero = sums<=1e-30
        # Keep zero-opacity representatives finite while the original fields
        # are restored below; their arbitrary zero weighted depth is not used.
        depth[zero] = z[representatives[zero]]
        offset = uv-center[owner]
        covariance2 = aggregate(projected+offset[:,:,None]*offset[:,None,:])
        means_camera = np.column_stack([center[:,0]/fx*depth,center[:,1]/fy*depth,depth])
        lift = rotation[:,[0,1]][None]*np.stack([depth/fx,depth/fy],axis=1)[:,None,:]
        covariance = lift@(covariance2-SCREEN_VARIANCE*np.eye(2)[None])@lift.swapaxes(1,2)
        radial = (means_camera@rotation.T)/depth[:,None]
        depth_variance = aggregate(camera_cov[:,2,2]+(z-depth[owner])**2)
        covariance += depth_variance[:,None,None]*radial[:,:,None]*radial[:,None,:]
        opacity = sums/np.sqrt(np.maximum(np.linalg.det(covariance2),1e-30))
        record['opacity_clipped_clusters'] = int(np.count_nonzero(opacity>.99))
        merged = dict(means=means_camera@rotation.T,covariances=covariance,
                      harmonics=aggregate(gaussians['harmonics'][available]),
                      opacities=np.minimum(.99,opacity))
        for key,value in merged.items():
            value[zero] = gaussians[key][keep][zero]
            result[key][mask] = value.astype(np.float32)
    validate_gaussians(result)
    return result,records


def spatial_sample_indices(index, budget, source_shape):
    """Retain an exact budget without periodic deletion of whole pixel columns.

    Each occupied 2x2 source-pixel block receives a proportional integer quota.
    Largest remainders allocate the leftover budget; a 2D low-discrepancy phase
    breaks equal-remainder ties. Within a block, opposite corners come first,
    with the first corner alternating between blocks. Ownership holes stay
    holes: this function selects existing parameters and never synthesizes them.
    """
    index = np.asarray(index)
    if (index.ndim != 1 or index.dtype.kind not in 'iu'
            or type(budget) is not int or budget < 1
            or not isinstance(source_shape, (tuple, list)) or len(source_shape) != 2
            or any(type(x) is not int or x < 1 for x in source_shape)):
        raise ValueError('Invalid spatial sampling index, budget or source shape')
    h, w = source_shape
    if (h*w > 1000000 or np.any(index < 0) or np.any(index >= h*w)
            or (len(index) > 1 and np.any(index[1:] <= index[:-1]))):
        raise ValueError('Spatial sampling requires ordered unique source-pixel indices')
    if len(index) <= budget:
        return index.copy()
    y, x = np.divmod(index, w)
    block_width, block_height = (w+1)//2, (h+1)//2
    block = (y//2)*block_width+x//2
    sizes = np.bincount(block.astype(np.intp), minlength=block_width*block_height)
    # Integer arithmetic keeps the quota exact even at one-third budgets.
    scaled = sizes.astype(np.int64)*budget
    quotas, remainders = np.divmod(scaled, len(index))
    remaining = budget-int(quotas.sum())
    if remaining:
        occupied = np.flatnonzero(sizes)
        by, bx = np.divmod(occupied, block_width)
        phase = np.remainder((bx+.5)*.7548776662466927+(by+.5)*.5698402909980532, 1.)
        extra = occupied[np.lexsort((occupied, phase, -remainders[occupied]))[:remaining]]
        quotas[extra] += 1
    # Every half-retained full block covers both its rows and its columns.
    # Cross the block axes: using x to choose the local x corner would still
    # erase alternating source columns when the budget is one point per block.
    first = ((y//2) % 2)+2*((x//2) % 2)
    corner = 2*(y % 2)+(x % 2)
    rank = np.where(corner == first, 0, np.where(corner == (first ^ 3), 1,
                    np.where(corner == (first ^ 1), 2, 3)))
    order = np.lexsort((rank, block))
    starts = np.cumsum(sizes)-sizes
    offset = np.arange(len(index))-starts[block[order]]
    selected = index[order[offset < quotas[block[order]]]]
    return np.sort(selected)


def balanced_sample_indices(index, budget, source_shape):
    """Keep the historical sample unless it empties supported image lines.

    Regular decimation is often good for fine texture, but rational strides can
    delete complete columns/rows. Use the spatial alternative only when it
    leaves fewer such holes. This decision uses source-pixel coverage, not GT
    images, scene names, target views or an offline quality score.
    """
    spatial = spatial_sample_indices(index,budget,source_shape)
    if len(index)<=budget:
        return spatial
    legacy = index[np.linspace(0,len(index)-1,budget,dtype=int)]
    h,w = source_shape
    fraction = budget/len(index)
    def holes(selected):
        missing = 0
        for all_coord,coord,length in ((index//w,selected//w,h),(index % w,selected % w,w)):
            available = np.bincount(all_coord.astype(np.intp),minlength=length)
            retained = np.bincount(coord.astype(np.intp),minlength=length)
            supported = (available>=4) & (available*fraction>=2)
            missing += int(np.count_nonzero(supported & (retained==0)))
        return missing
    return spatial if holes(spatial)<holes(legacy) else legacy
