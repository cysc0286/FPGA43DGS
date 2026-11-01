"""Bounded live JPEG sampling and prepared input, without a full-video archive."""
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
from urllib.request import urlopen


@dataclass(frozen=True)
class FramePacket:
    sequence: int
    monotonic: float
    jpeg: bytes


class FrameWindow:
    def __init__(self, capacity=12, seconds=6., byte_limit=16*1024**2):
        if not 3<=capacity<=12 or not 1<=seconds<=10 or byte_limit<1024:
            raise ValueError('Invalid live window limits')
        self.frames=deque();self.capacity=capacity;self.seconds=seconds;self.byte_limit=byte_limit
        self.bytes=0;self.received=0;self.resets=0

    def push(self, packet, now):
        if (packet.sequence<0 or not math.isfinite(packet.monotonic)
                or not 0<=now-packet.monotonic<=3 or not packet.jpeg
                or len(packet.jpeg)>min(8*1024**2,self.byte_limit)):
            raise ValueError('Invalid or stale source frame')
        if self.frames:
            last=self.frames[-1]
            if packet.sequence==last.sequence:return False
            if packet.sequence<last.sequence or packet.monotonic<=last.monotonic:
                self.frames.clear();self.bytes=0;self.resets+=1
        self.frames.append(packet);self.bytes+=len(packet.jpeg);self.received+=1
        while (len(self.frames)>self.capacity or self.bytes>self.byte_limit
               or packet.monotonic-self.frames[0].monotonic>self.seconds):
            self.bytes-=len(self.frames.popleft().jpeg)
        return True

    def latest(self, after=-1, minimum=8, span=3.):
        if (len(self.frames)<minimum or self.frames[-1].sequence<=after
                or self.frames[-1].monotonic-self.frames[0].monotonic<span):return None
        return tuple(self.frames)


class LiveSampler:
    def __init__(self, url, interval=.5, stop_path=None):
        parsed=urlsplit(url)
        if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','::1') or parsed.username or parsed.password:
            raise ValueError('Live source must be a loopback HTTP camera endpoint')
        if not math.isfinite(interval) or not .1<=interval<=1:raise ValueError('Invalid sampling interval')
        self.url,self.interval=url,interval
        self.stop_path=Path(stop_path) if stop_path else None
        self.window=FrameWindow();self.lock=threading.Lock();self.stopped=threading.Event()
        self.error=None;self.thread=None

    def start(self):
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start();return self

    def run(self):
        while not self.stopped.is_set():
            if self.stop_path and self.stop_path.exists():break
            begin=time.monotonic()
            try:
                with urlopen(self.url,timeout=3) as response:
                    payload=response.read(8*1024**2+1)
                    packet=FramePacket(int(response.headers['X-Frame-Id']),
                        float(response.headers['X-Capture-Monotonic']),payload)
                with self.lock:self.window.push(packet,time.monotonic());self.error=None
            except (OSError,ValueError,TypeError) as exc:
                with self.lock:self.error=str(exc)
            self.stopped.wait(max(0,self.interval-(time.monotonic()-begin)))

    def snapshot(self, after=-1):
        with self.lock:
            frames=self.window.latest(after)
            if frames and time.monotonic()-frames[-1].monotonic>3:frames=None
            return frames,dict(buffer_frames=len(self.window.frames),buffer_bytes=self.window.bytes,
                               sampled_frames=self.window.received,source_resets=self.window.resets,error=self.error)

    def close(self):
        self.stopped.set()
        if self.thread:self.thread.join(4)


def decode_window(packets, calibration=None, width=480):
    import cv2
    import numpy as np
    result={};shape=None
    for packet in packets:
        image=cv2.imdecode(np.frombuffer(packet.jpeg,np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError('Invalid JPEG in live window')
        if shape is not None and image.shape!=shape:raise ValueError('Camera mode changed')
        shape=image.shape
        if calibration:image=calibration.rectify(image,width)
        else:
            h,w=image.shape[:2];scale=min(1.,width/w)
            image=cv2.resize(image,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
        result[packet.sequence]=image
    return result


def prepare_window(frames, chosen, intrinsic, calibration_meta, out, packets, size=128):
    """Serialize the established two-context ABI from registered live views."""
    import cv2
    import numpy as np
    from common import save,sha
    out=Path(out);out.mkdir(parents=True,exist_ok=False);(out/'images').mkdir()
    context,targets,pair,target_poses,selection=chosen
    if len(context)!=2 or len(targets)!=1 or len(set(context+targets))!=3:
        raise ValueError('Live context and held-out target must be distinct')
    h,w=frames[context[0]].shape[:2]
    if any(frame.shape[:2]!=(h,w) for frame in frames.values()):raise ValueError('Prepared frame sizes differ')
    scale=size/min(w,h)
    affine=np.array([[scale,0,(size-1)/2-scale*(w-1)/2],
                     [0,scale,(size-1)/2-scale*(h-1)/2]],np.float64)
    k=np.vstack((affine,[0,0,1]))@intrinsic;k[:2]/=size;k[:2,2]+=.5/size
    if not np.allclose(k[:2,2],.5,atol=1e-8):raise ValueError('Live renderer needs centered intrinsics')
    poses={context[0]:np.eye(4),context[1]:pair[0],targets[0]:target_poses[targets[0]][0]}
    pixels=[];views=[]
    for index in context+targets:
        name=f'frame_{index:09d}.png'
        image=cv2.warpAffine(frames[index],affine,(size,size),flags=cv2.INTER_LINEAR)
        if not cv2.imwrite(str(out/'images'/name),image):raise OSError('Cannot save live frame')
        views.append(dict(name=name,index=index,role='context' if index in context else 'target',
                          c2w=poses[index].tolist(),intrinsics_normalized=k.tolist(),width=size,height=size,
                          prepared_sha256=sha(out/'images'/name)))
        if index in context:pixels.append(image[:,:,::-1].transpose(2,0,1).astype(np.float32)/255)
    np.savez(out/'context.npz',image=np.array(pixels,np.float32),
             extrinsics=np.array([poses[i] for i in context],np.float32),intrinsics=np.array([k,k],np.float32),
             near=np.ones(2,np.float32),far=np.full(2,100,np.float32))
    meta=dict(schema='mvsplat_context_v1',pose_source='anchored_local_live_pnp',views=views,
        context_indices=context,target_indices=targets,context_sha256=sha(out/'context.npz'),
        calibration=calibration_meta,frame_selection=selection,near=1,far=100,
        source_frames=[dict(index=p.sequence,monotonic=p.monotonic,sha256=hashlib.sha256(p.jpeg).hexdigest()) for p in packets],
        normalization=dict(kind='fixed_initial_anchor',median_triangulated_depth=10.),
        target_scope='Held out from Gaussian context; local anchor map only, no global SLAM')
    save(out/'input.json',meta);return meta
