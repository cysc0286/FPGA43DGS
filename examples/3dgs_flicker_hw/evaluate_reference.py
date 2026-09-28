import argparse,json,struct,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'examples/3dgs_scene'))
from check_output import compare
def main():
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();d=a.directory
 meta=json.loads((d/'input.json').read_text());w,h=meta['width'],meta['height']
 if meta['tiles']!=list(range(((w+15)//16)*((h+15)//16))):raise ValueError('Full frame required')
 data=np.fromfile(d/'expected.raw',dtype=np.dtype([('rgba','<f2',(4,)),('last','<u4'),('tag','<u4')]))
 if len(data)!=len(meta['tiles'])*256:raise ValueError('HLS reference incomplete')
 if not np.array_equal(data['tag'],np.arange(len(data),dtype=np.uint32)):raise ValueError('Reference tags')
 frame=np.zeros(w*h,dtype=[('rgba','<f4',(4,)),('last','<u4')])
 for t in meta['tiles']:
  for py in range(16):
   x=t%((w+15)//16)*16;y=t//((w+15)//16)*16+py
   if y>=h:continue
   width=min(16,w-x);src=data[t*256+py*16:t*256+py*16+width]
   frame[y*w+x:y*w+x+width]['rgba']=src['rgba'];frame[y*w+x:y*w+x+width]['last']=src['last']
 out=d/'frame.bin';out.write_bytes(b'GSSOUT01'+struct.pack('<II',w,h)+frame.tobytes())
 official=REPO/'examples/3dgs_scene/data/20260926T135717'/meta['case']/'official.bin'
 report={'scope':'FP16 HLS C reference, not board measurement','official_comparison':compare(out,official)}
 (d/'quality.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
