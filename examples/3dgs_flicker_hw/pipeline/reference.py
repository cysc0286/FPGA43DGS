"""Frozen HLS C executable reference, parallelized only over independent tiles."""
import argparse,concurrent.futures,hashlib,json,os,subprocess,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--modes',type=int,nargs='+',default=[0,1,2,3,4,5]);p.add_argument('--workers',type=int,default=4);p.add_argument('--project',default='hls_pipeline_workset');a=p.parse_args()
 folder=a.folder.resolve();meta=json.loads((folder/'input.json').read_text());raw=(folder/'input.flk').read_bytes();counts=meta['counts'];sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
 if hashlib.sha256(raw).hexdigest()!=meta['input_sha256']:raise ValueError('Input changed')
 built=ROOT/'build'/a.project/'solution1/csim/build/csim.exe'
 exe=folder/'reference.exe'
 if exe.exists():raise ValueError('Frozen reference already exists')
 shutil.copy2(built,exe)
 env=dict(os.environ);x='D:/Xilinx/Vivado/2018.3';env['PATH']=';'.join([x+'/msys64/mingw64/bin',x+'/win64/tools/fpo_v7_0',x+'/win64/lib/csim',env.get('PATH','')])
 groups=[[] for _ in range(min(a.workers,len(counts)))];loads=[0]*len(groups)
 for t in sorted(range(len(counts)),key=lambda i:counts[i],reverse=True):
  k=min(range(len(groups)),key=lambda i:loads[i]);groups[k].append(t);loads[k]+=counts[t]
 offsets=np.concatenate(([0],np.cumsum(np.array(counts)+1)))*64
 for k,ts in enumerate(groups):
  d=folder/str(k);d.mkdir(exist_ok=True)
  (d/'input.flk').write_bytes(b''.join(raw[offsets[t]:offsets[t+1]] for t in ts))
 report={'project':a.project,'exe_sha256':sha(exe),'input_sha256':meta['input_sha256'],'groups':groups,'source_hashes':{str(f.relative_to(ROOT)):sha(f) for path in ['pipeline','ctu','hls'] for f in (ROOT/path).glob('*') if f.suffix in ['.cpp','.hpp']},'complete':False,'modes':{}}
 reportpath=folder/'execution.json'
 if reportpath.exists():raise ValueError('Execution already recorded; preserve it before rerunning')
 def save():reportpath.write_text(json.dumps(report,indent=2))
 save()
 def run(job):
  k,m=job;d=folder/str(k);out=d/f'mode{m}.raw'
  with (d/f'mode{m}.log').open('w') as log:r=subprocess.run([str(exe),str(d/'input.flk'),str(out),str(len(groups[k])),str(m)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=2400)
  if r.returncode or out.stat().st_size!=len(groups[k])*4096:raise RuntimeError(f'Reference failed group={k} mode={m} exit={r.returncode}')
  print(f'DONE group={k} mode={m}',flush=True)
 dtype=np.dtype([('rgba','<u2',(4,)),('last','<u4'),('tag','<u4')])
 for m in a.modes:
  with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(run,[(k,m) for k in range(len(groups))]))
  merged=np.zeros(len(counts)*256,dtype=dtype)
  for k,ts in enumerate(groups):
   data=np.fromfile(folder/str(k)/f'mode{m}.raw',dtype=dtype)
   if not np.array_equal(data['tag'],np.arange(len(data),dtype=np.uint32)):raise ValueError('Tag mismatch')
   for i,t in enumerate(ts):merged[t*256:(t+1)*256]=data[i*256:(i+1)*256]
  merged['tag']=np.arange(len(merged),dtype=np.uint32);out=folder/f'expected_mode{m}.raw';merged.tofile(out)
  report['modes'][str(m)]={'output_sha256':sha(out)};save()
 report['complete']=True;save()
if __name__=='__main__':main()
