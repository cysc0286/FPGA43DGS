"""FP32 feature stream for FLK1; no Gaussian-pixel expansion or CPU CAT."""
import argparse,hashlib,json,struct
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--case',default='n559263_v0_w320');p.add_argument('--scene',type=Path);p.add_argument('--tiles',default='0,110,230');p.add_argument('--limit',type=int,default=256);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 source=a.scene if a.scene else REPO/'examples/3dgs_scene/data/20260926T135717'/a.case/'scene.bin';raw=source.read_bytes()
 if raw[:8]!=b'GSSCN001':raise ValueError('Scene ABI mismatch')
 w,h,tile,n,active,entries,tiles=struct.unpack_from('<7I',raw,8);bg=struct.unpack_from('<3f',raw,36)
 gs=np.frombuffer(raw,np.dtype([('id','<u4'),('q','<f4',(10,))]),active,48)['q']
 ranges=np.frombuffer(raw,'<u4',tiles*2,48+active*44).reshape(-1,2);ids=np.frombuffer(raw,'<u4',entries,48+active*44+tiles*8)
 selected=list(range(tiles)) if a.tiles=='all' else list(map(int,a.tiles.split(',')));counts=[]
 a.output.parent.mkdir(parents=True,exist_ok=True)
 if a.output.exists():raise ValueError('Input already exists; preserve frozen evidence')
 with a.output.open('wb') as f:
  for t in selected:
   b,e=map(int,ranges[t]);e=min(e,b+a.limit) if a.limit else e;count=e-b;counts.append(count)
   x=t%((w+15)//16)*16;y=t//((w+15)//16)*16;hdr=bytearray(64)
   struct.pack_into('<IHHH',hdr,0,count,x,y,min(16,w-x)|(min(16,h-y)<<5));hdr[10:16]=np.array(bg,dtype='<f2').tobytes();f.write(hdr)
   data=np.zeros((count,64),np.uint8);data[:,:36]=gs[ids[b:e],:9].astype('<f4').view(np.uint8).reshape(count,36);f.write(data.tobytes())
 meta={'schema':'FLK1-FP32-features','scene_sha256':hashlib.sha256(raw).hexdigest(),'case':a.case,'tiles':selected,'counts':counts,'width':w,'height':h,'records':sum(counts)+len(counts),'input_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()}
 a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2));print('PREPARED',a.output,meta['records'])
if __name__=='__main__':main()
