"""Expand frozen real Tile attributes to CTU-only sub-tile validation packets."""
import hashlib,json,struct
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
 source=ROOT/'evidence/real_reference/input.flk';raw=source.read_bytes()
 out=ROOT/'evidence/ctu_real';out.mkdir(exist_ok=True)
 at=0;packets=[];gaussians=0
 while at<len(raw):
  n,ox,oy,shape=struct.unpack_from('<IHHH',raw,at);width=shape&31;height=(shape>>5)&31;at+=64
  for j in range(n):
   fields=np.frombuffer(raw,dtype='<f2',count=9,offset=at).astype(np.float64)
   a,b,c=fields[2:5];tr=a+c;delta=np.hypot(a-c,2*b);spiky=(tr+delta)>=9*(tr-delta)
   for sy in [0,8]:
    for sx in [0,8]:
     if sx>=width or sy>=height:continue
     q=bytearray(raw[at:at+64]);struct.pack_into('<HHH',q,18,ox+sx,oy+sy,min(8,width-sx)|(min(8,height-sy)<<5)|(int(spiky)<<10))
     struct.pack_into('<I',q,24,gaussians);packets.append(q)
   at+=64;gaussians+=1
 if at!=len(raw):raise ValueError('Source length')
 data=b''.join(packets);(out/'input.flk').write_bytes(data)
 meta={'scope':'CTU masks only; all valid sub-tiles, no AABB filtering','input_gaussians':gaussians,'subtile_packets':len(packets),
       'spiky_flag':'axis ratio >=3 computed from encoded FP16 conic; not a preprocessing reproduction',
       'source_sha256':hashlib.sha256(raw).hexdigest(),'input_sha256':hashlib.sha256(data).hexdigest()}
 (out/'input.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))

if __name__=='__main__':main()
