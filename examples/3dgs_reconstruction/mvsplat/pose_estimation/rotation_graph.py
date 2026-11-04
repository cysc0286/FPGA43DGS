"""Rotation averaging using observed image edges, with an anchored SO(3) gauge.

No heading, turn rate or 360-degree endpoint is supplied to this solver. Failed
consistency checks return the original sequence without changing translations.
"""
import numpy as np


def _angle(rotation):
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1))))


def _project(matrix):
    u,_,vt=np.linalg.svd(matrix)
    return u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt


def _residual(poses,edge):
    i,j,relative,_=edge
    return _angle(relative@poses[i][:3,:3].T@poses[j][:3,:3])


def refine_observed_rotations(poses,local_edges,loop_edges):
    """Edges are (source, target, source-to-target ray rotation, weight)."""
    info=dict(applied=False,method='anchored_observed_rotation_graph',
              loop_count=len(loop_edges),reason='insufficient_loop_evidence')
    if len(loop_edges)<2:return poses,info
    ids=sorted(poses);anchor=ids[0]
    before=[_residual(poses,e) for e in loop_edges]
    info['max_loop_before_degrees']=max(before)
    if max(before)<=3.:
        info['reason']='already_consistent';return poses,info
    edges=list(local_edges)+list(loop_edges)
    for i,j,r,weight in edges:
        if (i not in poses or j not in poses or i==j or not np.isfinite(weight) or weight<=0
                or np.asarray(r).shape!=(3,3) or not np.isfinite(r).all()
                or not np.allclose(r.T@r,np.eye(3),atol=1e-6) or np.linalg.det(r)<.999999):
            raise ValueError('Invalid observed rotation edge')
    if any(not np.array_equal(p[:3,3],np.zeros(3)) for p in poses.values()):
        raise ValueError('Rotation graph cannot refine translated cameras')
    if not np.allclose(poses[anchor],np.eye(4),atol=1e-10):
        raise ValueError('Rotation graph requires the original identity anchor')
    positions={i:3*n for n,i in enumerate(ids[1:])}
    count=3*(len(ids)-1)
    normal=np.zeros((count,count),np.float64);rhs=np.zeros((count,3),np.float64)
    identity=np.eye(3)
    neighbours={i:[] for i in ids}
    for i,j,r,weight in edges:
        neighbours[i].append((j,r,weight));neighbours[j].append((i,r.T,weight))
        if i!=anchor:
            a=positions[i];normal[a:a+3,a:a+3]+=weight*identity
        if j!=anchor:
            b=positions[j];normal[b:b+3,b:b+3]+=weight*identity
        if i==anchor:
            rhs[b:b+3]+=weight*r
        elif j==anchor:
            rhs[a:a+3]+=weight*r.T
        else:
            normal[a:a+3,b:b+3]-=weight*r.T
            normal[b:b+3,a:a+3]-=weight*r
    try:
        solved=np.linalg.solve(normal,rhs)
    except np.linalg.LinAlgError:
        info['reason']='disconnected_or_singular_graph';return poses,info
    if not np.isfinite(solved).all():
        info['reason']='nonfinite_solution';return poses,info
    rotations={anchor:np.eye(3)}
    for index,offset in positions.items():rotations[index]=_project(solved[offset:offset+3].T)
    # Projected coordinate descent keeps every camera on SO(3), using actual
    # edge weights. This is not interpolation of a desired endpoint angle.
    for _ in range(40):
        largest=0.
        for index in ids[1:]:
            value=sum(weight*(rotations[other]@transport)
                      for other,transport,weight in neighbours[index])
            updated=_project(value)
            largest=max(largest,float(np.linalg.norm(updated-rotations[index])))
            rotations[index]=updated
        if largest<1e-9:break
    candidate={i:poses[i].copy() for i in ids}
    for i in ids:candidate[i][:3,:3]=rotations[i]
    after=[_residual(candidate,e) for e in loop_edges]
    local=[_residual(candidate,e) for e in local_edges]
    corrections=[_angle(candidate[i][:3,:3]@poses[i][:3,:3].T) for i in ids]
    def objective(value):
        return float(sum(weight*np.linalg.norm(relative@value[i][:3,:3].T@value[j][:3,:3]-identity)**2
                         for i,j,relative,weight in edges))
    info.update(max_loop_after_degrees=max(after),max_local_after_degrees=max(local),
                p95_local_after_degrees=float(np.percentile(local,95)),
                max_correction_degrees=max(corrections),
                objective_before=objective(poses),objective_after=objective(candidate))
    if (max(after)>3. or max(local)>5. or np.percentile(local,95)>2.
            or max(corrections)>45. or info['objective_after']>=info['objective_before']):
        info['reason']='observed_edges_disagree';return poses,info
    info.update(applied=True,reason='observed_loop_and_local_edges_consistent')
    return candidate,info
