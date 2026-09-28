"""Parallel independent tiles through the SAME frozen HLS C executable.

Only golden generation is parallelized; this is not a CPU performance result.
"""
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--case',default='n559263_v10_w320');p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 d=a.output.resolve();d.mkdir(parents=True,exist_ok=True)
 subprocess.run([sys.executable,str(ROOT/'prepare_reference.py'),'--case',a.case,'--tiles','all','--limit','0','--output',str(d/'input.flk')],check=True,stdout=subprocess.DEVNULL)
 meta=json.loads((d/'input.json').read_text());raw=(d/'input.flk').read_bytes();counts=meta['counts'];groups=[[] for _ in range(a.workers)];loads=[0]*a.workers
 for t in sorted(range(len(counts)),key=lambda i:counts[i],reverse=True):
  k=min(range(a.workers),key=lambda i:loads[i]);groups[k].append(t);loads[k]+=counts[t]
 offsets=np.concatenate(([0],np.cumsum(np.array(counts)+1)))*64
 exe=ROOT/'build/hls_renderer/solution1/csim/build/csim.exe'
 env=dict(os.environ);x='D:/Xilinx/Vivado/2018.3'
 env['PATH']=';'.join([x+'/msys64/mingw64/bin',x+'/win64/tools/fpo_v7_0',x+'/win64/lib/csim',env.get('PATH','')])
 report={'exe_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'groups':groups,'complete':False}
 (d/'execution.json').write_text(json.dumps(report,indent=2))
 def work(k):
  folder=d/str(k);folder.mkdir(exist_ok=True);parts=groups[k];input=folder/'input.flk';output=folder/'expected.raw'
  with input.open('wb') as f:
   for t in parts:f.write(raw[offsets[t]:offsets[t+1]])
  with (folder/'run.txt').open('w') as log:r=subprocess.run([str(exe),str(input),str(output),str(len(parts))],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=2400)
  if r.returncode or output.stat().st_size!=len(parts)*4096:raise RuntimeError('Reference partition failed '+str(k)+' exit='+str(r.returncode))
  print('REFERENCE_PART_DONE='+str(k),flush=True);return k
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(work,range(a.workers)))
 merged=np.zeros(len(counts)*256,dtype=[('rgba','<u2',(4,)),('last','<u4'),('tag','<u4')])
 for k,parts in enumerate(groups):
  data=np.fromfile(d/str(k)/'expected.raw',dtype=merged.dtype)
  if not np.array_equal(data['tag'],np.arange(len(data),dtype=np.uint32)):raise ValueError('Partition tags')
  for i,t in enumerate(parts):merged[t*256:(t+1)*256]=data[i*256:(i+1)*256]
 merged['tag']=np.arange(len(merged),dtype=np.uint32);merged.tofile(d/'expected.raw')
 report['complete']=True;report['expected_sha256']=hashlib.sha256((d/'expected.raw').read_bytes()).hexdigest();(d/'execution.json').write_text(json.dumps(report,indent=2))
 subprocess.run([sys.executable,str(ROOT/'evaluate_reference.py'),str(d)],check=True)
if __name__=='__main__':main()
