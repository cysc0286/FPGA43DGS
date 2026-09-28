"""Full-frame quality for each frozen HLS C mode; never a board benchmark."""
import argparse,hashlib,json,struct,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'examples/3dgs_scene'))
from check_output import compare
def main():
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();d=a.folder
 meta=json.loads((d/'input.json').read_text());execution=json.loads((d/'execution.json').read_text());w,h=meta['width'],meta['height'];tiles=meta['tiles']
 if tiles!=list(range(((w+15)//16)*((h+15)//16))) or not execution['complete']:raise ValueError('Complete full-frame reference required')
 base={'n559263_v0_w320':'full_reference_v0','n559263_v10_w320':'full_reference_v10','n10000_v0_w320':'full_reference_10k'}[meta['case']]
 rows=[]
 for mode in range(6):
  source=d/f'expected_mode{mode}.raw'
  if hashlib.sha256(source.read_bytes()).hexdigest()!=execution['modes'][str(mode)]['output_sha256']:raise ValueError('Golden changed')
  data=np.fromfile(source,dtype=np.dtype([('rgba','<f2',(4,)),('last','<u4'),('tag','<u4')]))
  if not np.array_equal(data['tag'],np.arange(len(tiles)*256,dtype=np.uint32)):raise ValueError('Tags or size')
  frame=np.zeros(w*h,dtype=[('rgba','<f4',(4,)),('last','<u4')])
  for t in tiles:
   for py in range(16):
    x=t%((w+15)//16)*16;y=t//((w+15)//16)*16+py
    if y>=h:continue
    width=min(16,w-x);src=data[t*256+py*16:t*256+py*16+width]
    frame[y*w+x:y*w+x+width]['rgba']=src['rgba'];frame[y*w+x:y*w+x+width]['last']=src['last']
  out=d/f'frame_mode{mode}.bin';out.write_bytes(b'GSSOUT01'+struct.pack('<II',w,h)+frame.tobytes())
  official=REPO/'examples/3dgs_scene/data/20260926T135717'/meta['case']/'official.bin'
  rows.append({'mode':mode,'official_fp32':compare(out,official),'flk0_fp16':compare(out,ROOT/'evidence'/base/'frame.bin')})
  if mode==0 and not rows[-1]['flk0_fp16']['bitwise_equal']:raise ValueError('Mode 0 changed FLK0 numerics')
 report={'scope':'Full-frame HLS C only, not physical performance','case':meta['case'],'modes':rows}
 (d/'quality.json').write_text(json.dumps(report,indent=2))
 for r in rows:print(r['mode'],'official_psnr',r['official_fp32']['psnr_vs_matching_official_db'],'FP16_base_psnr',r['flk0_fp16']['psnr_vs_matching_official_db'],'last_vs_base',r['flk0_fp16']['last_mismatches'])
if __name__=='__main__':main()
