"""Fixed-reference local registration; no independently normalized window fusion."""
from collections import deque
import cv2
import numpy as np
from pose_estimation.geometry import image_features,select_views,estimate_target,mutual_ratio_matches


class LocalAnchor:
    def __init__(self, intrinsic):
        self.intrinsic=np.asarray(intrinsic,np.float64)
        self.reference=None;self.landmarks={};self.trajectory=deque(maxlen=128)

    def select(self, frames):
        if self.reference is None:
            chosen=select_views(frames,self.intrinsic)
            context,targets,pair,poses,_=chosen
            self.reference=(context[0],frames[context[0]].copy(),pair[2],pair[3])
            self.landmarks={k:p.copy() for k,p in list(pair[1].items())[:2048]}
            self.trajectory.append([0.,0.,0.]);self.trajectory.append(pair[0][:3,3].tolist())
            return chosen
        first,image,keys,descriptors=self.reference
        frames[first]=image
        registered={};features={}
        for index in sorted(frames,reverse=True):
            if index==first:continue
            frame=frames[index]
            if cv2.Laplacian(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()<15:continue
            try:
                features[index]=image_features(frame)
                pose,info=estimate_target(frame,keys,descriptors,self.landmarks,self.intrinsic,30,features=features[index])
                if info['median_reprojection_px']>2.:continue
                world=np.array(list(self.landmarks.values()))
                ray=world-pose[:3,3]
                cosine=np.sum(world*ray,axis=1)/(np.linalg.norm(world,axis=1)*np.linalg.norm(ray,axis=1))
                parallax=float(np.median(np.degrees(np.arccos(np.clip(cosine,-1,1)))))
                if not np.isfinite(parallax) or parallax<1.:continue
                registered[index]=(pose,dict(info,median_parallax_deg=parallax))
            except (ValueError,cv2.error):continue
        candidates=[i for i in sorted(registered,reverse=True) if any(first<j<i for j in registered)]
        if not candidates:raise ValueError('No registered overlapping context and held-out target; return towards the initial scene')
        second=candidates[0]
        target=min((j for j in registered if first<j<second),key=lambda j:abs(j-(first+second)/2))
        pose,info=registered[second]
        self._extend(pose,features[second])
        self.trajectory.append(pose[:3,3].tolist())
        pair=(pose,self.landmarks,keys,descriptors,dict(info,translation_scale=1.,coordinate_frame='fixed_initial_anchor'))
        selection=dict(kind='anchored_local_live_v1',anchor_index=first,candidates=sorted(registered),
                       thresholds=dict(min_inliers=30,min_parallax_deg=1.,max_target_median_reprojection_px=2.))
        return [first,second],[target],pair,{target:registered[target]},selection

    def _extend(self, pose, features):
        first,image,keys,descriptors=self.reference
        matches=mutual_ratio_matches(descriptors,features[1])
        fresh=[m for m in matches if m.queryIdx not in self.landmarks]
        if not fresh:return
        p0=np.float64([keys[m.queryIdx].pt for m in fresh]);p1=np.float64([features[0][m.trainIdx].pt for m in fresh])
        rotation=pose[:3,:3].T;translation=-rotation@pose[:3,3]
        a=self.intrinsic@np.column_stack((np.eye(3),np.zeros(3)))
        b=self.intrinsic@np.column_stack((rotation,translation))
        homogeneous=cv2.triangulatePoints(a,b,p0.T,p1.T).T
        with np.errstate(divide='ignore',invalid='ignore'):
            world=homogeneous[:,:3]/homogeneous[:,3:];other=world@rotation.T+translation
            q0=world@self.intrinsic.T;q1=other@self.intrinsic.T
            error=np.maximum(np.linalg.norm(q0[:,:2]/q0[:,2:]-p0,axis=1),np.linalg.norm(q1[:,:2]/q1[:,2:]-p1,axis=1))
            rays=world-pose[:3,3]
            angle=np.degrees(np.arccos(np.clip(np.sum(world*rays,axis=1)/(np.linalg.norm(world,axis=1)*np.linalg.norm(rays,axis=1)),-1,1)))
        for match,point,depth,err,parallax in zip(fresh,world,other[:,2],error,angle):
            if len(self.landmarks)>=2048:break
            if np.isfinite(point).all() and point[2]>0 and depth>0 and err<=2 and parallax>=1:
                self.landmarks[match.queryIdx]=point

    def snapshot(self):
        return dict(points=[p.tolist() for p in self.landmarks.values()],trajectory=list(self.trajectory),
                    coordinate_frame='fixed_initial_anchor',scale='arbitrary monocular units',global_mapping=False)
