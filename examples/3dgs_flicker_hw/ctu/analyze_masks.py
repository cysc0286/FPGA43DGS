"""Separate FP16 mask error from CAT's inherent sparse sampling loss."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
 d=ROOT/'evidence/ctu_real';raw=(d/'input.flk').read_bytes();n=len(raw)//64
 rows=np.frombuffer(raw,np.uint8).reshape(n,64)
 q=rows[:,:18].copy().view('<f2').reshape(n,9).astype(np.float64)
 meta=rows[:,18:24].copy().view('<u2').reshape(n,3).astype(np.uint32)
 ox,oy=meta[:,0],meta[:,1];width=meta[:,2]&31;height=(meta[:,2]>>5)&31;spiky=((meta[:,2]>>10)&1)!=0
 if (width==0).any() or (height==0).any():raise ValueError('Invalid extents')
 with np.errstate(divide='ignore'):limit=np.log(255*q[:,5])
 def hit(x,y):
  dx=x-q[:,0];dy=y-q[:,1];energy=.5*(q[:,2]*dx*dx+q[:,4]*dy*dy)+q[:,3]*dx*dy
  return (energy>=0)&(energy<=limit)
 leader_dense=np.zeros((n,4),bool);leader_sparse=np.zeros((n,4),bool)
 qualified=np.zeros((n,4),np.uint32);valid_mini=np.zeros((n,4),bool);candidate_pairs=0
 for m in range(4):
  mx=(m%2)*4;my=(m//2)*4
  left=ox+mx;top=oy+my;right=np.minimum(left+3,ox+width-1);bottom=np.minimum(top+3,oy+height-1)
  valid=(mx<width)&(my<height);valid_mini[:,m]=valid
  a=hit(left,top);b=hit(right,top);c=hit(left,bottom);e=hit(right,bottom)
  leader_dense[:,m]=valid&(a|b|c|e);leader_sparse[:,m]=valid&(a|e)
  for py in range(4):
   for px in range(4):
    v=(mx+px<width)&(my+py<height);candidate_pairs+=int(v.sum());qualified[:,m]+=(v&hit(left+px,top+py)).astype(np.uint32)
 report={'scope':'CTU-only diagnostic on 768 selected real Gaussian records, not complete-frame quality or timing',
         'analytic_reference':'FP64 alpha threshold on encoded FP16 Gaussian attributes, before transmittance/early termination',
         'input_sha256':hashlib.sha256(raw).hexdigest(),'candidate_pixel_pairs':candidate_pairs,'runs':[]}
 for mode in range(5):
  data=(d/f'expected_mode{mode}.raw').read_bytes();out=np.frombuffer(data,np.uint8).reshape(n,64)
  hardware=np.column_stack([((out[:,28]>>m)&1)!=0 for m in range(4)])
  dense=np.full(n,mode==1)|((mode==3)&~spiky)|((mode==4)&spiky)
  analytic=valid_mini if mode==0 else np.where(dense[:,None],leader_dense,leader_sparse)
  report['runs'].append({'mode':mode,'output_sha256':hashlib.sha256(data).hexdigest(),
   'valid_mini_pairs':int(valid_mini.sum()),'kept_mini_pairs':int(hardware.sum()),
   'fp16_additional_rejected_mini_pairs':int((analytic&~hardware).sum()),
   'fp16_additional_kept_mini_pairs':int((~analytic&hardware).sum()),
   'analytic_sampling_missed_alpha_qualified_pixels':int(qualified[~analytic].sum()),
   'fp16_cat_missed_alpha_qualified_pixels':int(qualified[~hardware].sum()),
   'alpha_qualified_pixels_without_cat':int(qualified.sum())})
 (d/'mask_analysis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
