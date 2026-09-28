"""Independent integer oracle and real-tile input replay. Never modifies baseline."""
import hashlib
import json
import random
import struct
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'3dgs_tile_cpu'
sys.path.insert(0,str(BASE))
from check_output import compare

ONE=1<<30
COLOR=1<<24
CLAMP=1063004406
MIN_ALPHA=4210753
MIN_T=107375
MAX_RGB=16*COLOR

def mul(a,b): return (a*b+(1<<29))>>30
def quant(x,scale): return int(float(x)*scale+0.5)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

class Model:
    def __init__(self): self.clear()
    def clear(self):
        self.rgb=[0]*3;self.t=ONE;self.last=self.seen=0
        self.terminated=self.finished=False;self.error=False
    def step(self,cmd):
        op,a,r,g,b,ordinal=cmd
        self.error=False
        if op==0:self.clear()
        elif (op not in (1,2) or self.finished or max(r,g,b)>MAX_RGB or
              (op==1 and (a>ONE or ordinal==0 or ordinal<=self.seen))):self.error=True
        elif op==2:
            self.rgb=[v+mul(c,self.t) for v,c in zip(self.rgb,[r,g,b])]
            self.finished=True
        else:
            self.seen=ordinal
            if not self.terminated and a>=MIN_ALPHA:
                weight=mul(self.t,min(a,CLAMP));nt=self.t-weight
                if nt<MIN_T:self.terminated=True
                else:
                    self.rgb=[v+mul(c,weight) for v,c in zip(self.rgb,[r,g,b])]
                    self.t=nt;self.last=ordinal
        return (*self.rgb,self.t,self.last,
                2*int(self.error)+4*int(self.terminated)+8*int(self.finished))

def main():
    data=ROOT/'data';data.mkdir(parents=True,exist_ok=True)
    ev=ROOT/'evidence';ev.mkdir(exist_ok=True)
    contract={'version':1,'alpha_fraction_bits':30,'rgb_fraction_bits':24,
      'rounding':'positive round-half-up','alpha_clamp':CLAMP,'alpha_min':MIN_ALPHA,
      'transmittance_min':MIN_T,'maximum_rgb':MAX_RGB,
      'thresholds':{'rgb_max_abs':1e-4,'rgb_mean_abs':1e-5,
                    'transmittance_max_abs':1e-5,'last_contributor':'exact'}}
    # This contract is fixed, not selected by searching for a passing result.
    model=Model();synthetic=[]
    def push(c): synthetic.append((c,model.step(c)))
    push((0,0,0,0,0,0))
    for i,a in enumerate([0,MIN_ALPHA-1,MIN_ALPHA,ONE//2,CLAMP,ONE],1):
        push((1,a,COLOR,COLOR//2,2*COLOR,i))
    push((2,0,COLOR//4,COLOR//2,COLOR,0))
    push((2,0,0,0,0,0));push((1,ONE//2,0,0,0,100))
    push((0,0,0,0,0,0));push((255,0,0,0,0,0))
    push((1,ONE+1,0,0,0,1));push((1,1,MAX_RGB+1,0,0,1))
    push((1,ONE//2,COLOR,0,0,1));push((1,ONE//2,0,COLOR,0,1))
    push((1,ONE//2,0,COLOR,0,0));push((1,ONE//2,0,COLOR,0,2))
    push((2,0,COLOR//2,0,COLOR,0))
    rng=random.Random(20260925)
    for pix in range(64):
        push((0,0,0,0,0,0))
        for i in range(1,33):
            push((1,rng.choice([rng.randrange(ONE+1),MIN_ALPHA,MIN_ALPHA-1]),
                  rng.randrange(MAX_RGB+1),rng.randrange(MAX_RGB+1),rng.randrange(MAX_RGB+1),i))
        push((2,0,rng.randrange(COLOR+1),rng.randrange(COLOR+1),rng.randrange(COLOR+1),0))
    # Hand-computable order / premultiplication sanity tests independent of RTL.
    m=Model();m.step((1,ONE//2,COLOR,0,0,1));v=m.step((1,ONE//2,0,COLOR,0,2))
    assert v[:5]==(COLOR//2,COLOR//4,0,ONE//4,2)
    assert m.step((2,0,0,0,COLOR,0))[:3]==(COLOR//2,COLOR//4,COLOR//4)
    base=json.loads((BASE/'data/manifest.json').read_text())
    inp=BASE/'data'/base['input_file'];expected=BASE/'data'/base['expected_file']
    assert sha(inp)==base['input_sha256'] and sha(expected)==base['expected_sha256']
    a=inp.read_text().split();assert a[0]=='GS_TILE_V1'
    x0,y0,w,h,n=map(int,a[1:6]);bg=np.array(a[6:9],dtype=np.float32)
    rows=np.array(a[9:],dtype=np.float32).reshape(n,11)
    xy=rows[:,1:3];conic=rows[:,3:6];opacity=rows[:,6];rgb=rows[:,7:10]
    assert np.all(rgb>=0) and np.all(rgb<=16)
    colors=[[quant(c,COLOR) for c in row] for row in rgb]
    qbg=[quant(c,COLOR) for c in bg]
    real=[];pixels=[];candidates=0
    for y in range(h):
        for x in range(w):
            model=Model();cmd=(0,0,0,0,0,0);real.append((cmd,model.step(cmd)))
            dx=xy[:,0]-np.float32(x0+x);dy=xy[:,1]-np.float32(y0+y)
            power=np.float32(-.5)*(conic[:,0]*dx*dx+conic[:,2]*dy*dy)-conic[:,1]*dx*dy
            # CPU/offline Gaussian evaluation. Do not use expected outputs here.
            alpha=np.minimum(np.float32(.99),opacity*np.exp(power))
            mask=(power<=0)&(alpha>=np.float32(1/255))
            for i in np.flatnonzero(mask):
                cmd=(1,quant(alpha[i],ONE),*colors[i],int(i)+1)
                real.append((cmd,model.step(cmd)));candidates+=1
            cmd=(2,0,*qbg,0);real.append((cmd,model.step(cmd)))
            pixels.append((*[c/COLOR for c in model.rgb],model.t/ONE,model.last))
    def write_vectors(name,records):
        p=data/name
        p.write_text(''.join(' '.join(f'{v:08x}' for v in (*c,*s))+'\n' for c,s in records),encoding='ascii')
        return {'file':name,'commands':len(records),'sha256':sha(p)}
    vectors=synthetic+real
    allinfo=write_vectors('vectors.txt',vectors);syninfo=write_vectors('synthetic.txt',synthetic)
    realinfo=write_vectors('tile_vectors.txt',real)
    # Input-only file: actual runtime never receives expected states.
    replay=data/'tile_input.txt'
    replay.write_text(''.join(' '.join(f'{v:08x}' for v in c)+'\n' for c,_ in real),encoding='ascii')
    syninput=data/'synthetic_input.txt'
    syninput.write_text(''.join(' '.join(f'{v:08x}' for v in c)+'\n' for c,_ in synthetic),encoding='ascii')
    blob=b'GSTO0001'+struct.pack('<II',w,h)+b''.join(struct.pack('<ffffI',*p) for p in pixels)
    actual=ev/'quantized_tile.bin';actual.write_bytes(blob)
    result=compare(actual,json.loads(expected.read_text()),contract['thresholds'])
    manifest={'schema':'GSC1','contract':contract,'vectors':allinfo,'synthetic':syninfo,'tile':realinfo,
      'tile_input_sha256':sha(replay),'synthetic_input_sha256':sha(syninput),
      'source_input_sha256':sha(inp),'source_expected_sha256':sha(expected),
      'generator_sha256':sha(Path(__file__)),'tile_size':[w,h],
      'gaussian_candidates_per_pixel':n,'alpha_qualified_pixel_pairs':candidates,
      'scope':'Offline CPU alpha evaluation; ordered PL compositing replay only'}
    (data/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ev/'quantization.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'vectors':allinfo,'synthetic':syninfo,'tile':realinfo,'quality':result},indent=2))
    if not result['passed']:raise SystemExit('Frozen quantization quality limits failed')

if __name__=='__main__':main()
