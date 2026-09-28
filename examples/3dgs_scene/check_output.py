"""Compare board pixels with PC-only official CUDA reference; fixed gates."""
import hashlib
import math
import struct
from pathlib import Path
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pixels(path):
    raw=Path(path).read_bytes()
    if raw[:8]!=b'GSSOUT01':raise ValueError('Output ABI')
    w,h=struct.unpack_from('<2I',raw,8)
    if len(raw)!=16+w*h*20:raise ValueError('Output length')
    p=np.frombuffer(raw,offset=16,dtype=[('rgba','<f4',(4,)),('last','<u4')])
    if not np.isfinite(p['rgba']).all():raise ValueError('Nonfinite output')
    return w,h,p


def compare(actual,expected):
    w,h,a=pixels(actual);ew,eh,b=pixels(expected)
    if (w,h)!=(ew,eh):raise ValueError('Image dimensions')
    d=np.abs(a['rgba'].astype(np.float64)-b['rgba'])
    last=a['last']!=b['last'];worst=int(np.argmax(d[:,:3]))
    mse=float(np.mean((np.clip(a['rgba'][:,:3],0,1).astype(np.float64)-np.clip(b['rgba'][:,:3],0,1))**2))
    return dict(pixels=w*h,rgb_max=float(d[:,:3].max()),rgb_mean=float(d[:,:3].mean()),
        transmittance_max=float(d[:,3].max()),last_mismatches=int(last.sum()),
        worst_xy=[int((worst//3)%w),int((worst//3)//w)],
        psnr_vs_matching_official_db=None if mse==0 else -10*math.log10(mse),
        bitwise_equal=sha(actual)==sha(expected),
        passed=bool(d[:,:3].max()<=1e-4 and d[:,:3].mean()<=1e-5 and d[:,3].max()<=1e-5 and not last.any()))


def timing(path):
    x=np.atleast_1d(np.genfromtxt(path,delimiter=',',names=True))
    total=x['total_us']
    if not np.isfinite(total).all() or (total<=0).any():raise ValueError('Timing values')
    return dict(samples=len(total),mean_ms=float(total.mean()/1000),median_ms=float(np.median(total)/1000),
        min_ms=float(total.min()/1000),max_ms=float(total.max()/1000),
        p95_ms=float(np.quantile(total,.95)/1000),p99_ms=float(np.quantile(total,.99)/1000),
        raster_fps=float(1e6/total.mean()),cpu_percent=float(100*x['process_cpu_us'].sum()/total.sum()),
        means={k:float(x[k].mean()) for k in x.dtype.names if k!='sample'})
