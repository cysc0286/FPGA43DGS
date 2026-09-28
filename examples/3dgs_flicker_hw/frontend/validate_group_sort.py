"""Compare every board-generated tile list to the official CUDA export."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
GAUSSIAN=np.dtype([('id','<u4'),('q','<f4',(10,))])


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def load(path):
    data=path.read_bytes()
    if data[:8]!=b'GSSCN001':raise ValueError('Scene ABI')
    w,h,tile,n,active,entries,tiles=struct.unpack_from('<7I',data,8)
    if tile!=16 or tiles!=((w+15)//16)*((h+15)//16):raise ValueError('Tile ABI')
    expected=48+active*44+tiles*8+entries*4
    if len(data)!=expected:raise ValueError('Scene length')
    gs=np.frombuffer(data,GAUSSIAN,active,48)
    ranges=np.frombuffer(data,'<u4',tiles*2,48+active*44).reshape(-1,2)
    ids=np.frombuffer(data,'<u4',entries,48+active*44+tiles*8)
    if np.any(ids>=active):raise ValueError('Out-of-range list index')
    return (w,h,n,active,entries,tiles),gs,ranges,gs['id'][ids]


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args()
    stage=json.loads((a.evidence/'result.json').read_text())
    if not stage['complete']:raise ValueError('Incomplete grouping')
    report={'stage':str(a.evidence.resolve()),'cases':[]}
    for item in stage['cases']:
        name=f'n559263_{item["name"]}_w320'
        board=a.evidence/(item['name']+'.scene')
        official=REPO/'examples/3dgs_scene/data/20260926T135717'/name/'scene.bin'
        if sha(board)!=item['scene_sha256']:raise ValueError('Board scene drift')
        b,bg,br,bi=load(board)
        o,og,orr,oi=load(official)
        if b[:4]!=o[:4]:raise ValueError('Dimension/active drift')
        if not np.array_equal(bg['id'],og['id']):raise ValueError('Active Gaussian IDs differ')
        range_match=int(np.count_nonzero(np.all(br==orr,axis=1)))
        differing_tiles=0;missing=0;extra=0;order_mismatch=0;duplicate=0
        examples=[]
        for tile in range(b[5]):
            x=bi[br[tile,0]:br[tile,1]]
            y=oi[orr[tile,0]:orr[tile,1]]
            if not len(x) and not len(y):continue
            if len(x)!=len(np.unique(x)):duplicate+=1
            sx=np.sort(x);sy=np.sort(y)
            shared=np.intersect1d(sx,sy,assume_unique=True)
            missing+=len(y)-len(shared);extra+=len(x)-len(shared)
            if not np.array_equal(sx,sy):differing_tiles+=1
            if not np.array_equal(x,y):
                order_mismatch+=1
                if len(examples)<8:examples.append({'tile':tile,'board_count':len(x),
                                                    'official_count':len(y),'same_members':np.array_equal(sx,sy),
                                                    'board_head':x[:8].tolist(),'official_head':y[:8].tolist()})
        result={'case':name,'board_entries':b[4],'official_entries':o[4],
                'range_exact_tiles':range_match,'tiles':b[5],
                'member_differing_tiles':differing_tiles,'missing_instances':missing,
                'extra_instances':extra,'order_differing_tiles':order_mismatch,
                'duplicate_tiles':duplicate,'examples':examples,
                'board_sha256':item['scene_sha256'],'official_sha256':sha(official),
                'board_timing':item['timing']}
        report['cases'].append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='examples'},indent=2))
    (a.evidence/'validation.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
