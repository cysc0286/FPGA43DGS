"""Check real runtime state against PC-only expected records."""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'3dgs_tile_cpu'))
import importlib.util
spec=importlib.util.spec_from_file_location('tile_compare',ROOT.parent/'3dgs_tile_cpu/check_output.py')
tile=importlib.util.module_from_spec(spec);spec.loader.exec_module(tile)

def check(actual,kind,mode,hardware_cycles=False):
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    v=manifest[kind];vp=ROOT/'data'/v['file']
    if hashlib.sha256(vp.read_bytes()).hexdigest()!=v['sha256']:raise ValueError('golden hash mismatch')
    vectors=[[int(x,16) for x in s.split()] for s in vp.read_text().splitlines()]
    expected=[r[6:] for r in vectors if mode=='trace' or r[0]==2]
    states=[[int(x,16) for x in s.split()] for s in Path(actual).read_text().splitlines()]
    if len(states)!=len(expected) or any(len(s)!=8 for s in states):raise ValueError('invalid output count/shape')
    bad=[i for i,(a,e) in enumerate(zip(states,expected)) if a[:6]!=e]
    report={'states':len(states),'mismatches':bad[:20],'mismatch_count':len(bad),'exact_passed':not bad,
            'actual_sha256':hashlib.sha256(Path(actual).read_bytes()).hexdigest()}
    if hardware_cycles:
        sim_path=ROOT/'evidence/simulation'/f'{kind}_trace.txt'
        sim_report=json.loads((ROOT/'evidence/simulation/result.json').read_text())
        if hashlib.sha256(sim_path.read_bytes()).hexdigest()!=sim_report[kind]['actual_sha256']:
            raise ValueError('simulation counter reference changed')
        sim=[[int(x,16) for x in line.split()] for line in sim_path.read_text().splitlines()]
        wanted=[s for v,s in zip(vectors,sim) if mode=='trace' or v[0]==2]
        cycle_bad=[i for i,(s,e) in enumerate(zip(states,wanted)) if s[6:]!=e[6:]]
        report['cycle_counter_mismatches']=cycle_bad[:20]
        report['cycle_counter_passed']=not cycle_bad
    if kind=='tile':
        pixels=[s for s in states] if mode=='tile' else [s for r,s in zip(vectors,states) if r[0]==2]
        w,h=manifest['tile_size']
        blob=b'GSTO0001'+struct.pack('<II',w,h)+b''.join(
            struct.pack('<ffffI',s[0]/(1<<24),s[1]/(1<<24),s[2]/(1<<24),s[3]/(1<<30),s[4]) for s in pixels)
        path=Path(actual).with_suffix('.bin');path.write_bytes(blob)
        base=ROOT.parent/'3dgs_tile_cpu/data';bm=json.loads((base/'manifest.json').read_text())
        report['official_quality']=tile.compare(path,json.loads((base/bm['expected_file']).read_text()),manifest['contract']['thresholds'])
    report['passed']=(report['exact_passed'] and report.get('official_quality',{}).get('passed',True)
                      and report.get('cycle_counter_passed',True))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('actual',type=Path);p.add_argument('--kind',choices=['synthetic','tile'],required=True)
    p.add_argument('--mode',choices=['trace','tile'],default='trace');a=p.parse_args()
    r=check(a.actual,a.kind,a.mode);a.actual.with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2));sys.exit(0 if r['passed'] else 1)
