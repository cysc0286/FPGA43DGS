"""Run paired full-frame measurements against a frozen staged executable."""
import argparse,datetime,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'examples/3dgs_compositor/board'))
from remote import connect,run
sys.path.insert(0,str(REPO/'examples/3dgs_scene'))
from check_output import compare
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--backends',nargs='+',default=['cpu1','cpu4']);p.add_argument('--cases',nargs='+',default=['n559263_v0_w320','n559263_v10_w320']);p.add_argument('--repeats',type=int,default=3);p.add_argument('--warmup',type=int,default=1);a=p.parse_args()
 stage=json.loads((a.stage/'result.json').read_text());dest=stage['remote']
 if not stage['compiled']:raise ValueError('Stage not built')
 ev=ROOT/'evidence'/('measure_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S'));ev.mkdir(parents=True);(ev/'received').mkdir()
 report={'stage':str(a.stage),'stage_record':stage,'runs':[],'complete':False};c=connect()
 def save():(ev/'result.json').write_text(json.dumps(report,indent=2))
 save()
 try:
  if run(c,'sha256sum '+dest+'/render')[1].split()[0]!=stage['binary_sha256']:raise ValueError('Binary drift')
  for case in a.cases:
   match=next(x for x in stage['cases'] if x['name']==case)
   if run(c,'sha256sum '+dest+'/'+case+'.bin')[1].split()[0]!=match['sha256']:raise ValueError('Input drift')
   for backend in a.backends:
    if backend not in ['cpu1','cpu4','serial','pipeline']:raise ValueError('Backend')
    stem=ev.name+'_'+case+'_'+backend
    if backend.startswith('cpu'):args=f'cpu {case}.bin {stem} base {backend[-1]} {a.repeats} {a.warmup} 0'
    else:args=f'{case}.bin {stem} {backend} 8 {a.repeats} {a.warmup}'
    print('RUN '+stem,flush=True)
    run(c,f'cd {dest} && timeout 600 ./render {args}',timeout=610,log=ev/(stem+'.txt'))
    suffixes=['.bin','_timing.csv']+([] if backend.startswith('cpu') else ['_trace.csv'])
    with c.open_sftp() as s:
     for suffix in suffixes:s.get(dest+'/'+stem+suffix,str(ev/'received'/(stem+suffix)))
    official=REPO/'examples/3dgs_scene/data/20260926T135717'/case/'official.bin'
    item={'case':case,'backend':backend,'stem':stem,'official':compare(ev/'received'/(stem+'.bin'),official)}
    reference={'n559263_v0_w320':'full_reference_v0','n559263_v10_w320':'full_reference_v10','n10000_v0_w320':'full_reference_10k'}.get(case)
    if reference and (ROOT/'evidence'/reference/'frame.bin').exists():
     item['fp16_hls']=compare(ev/'received'/(stem+'.bin'),ROOT/'evidence'/reference/'frame.bin')
    report['runs'].append(item);save();print(json.dumps(item),flush=True)
    if not backend.startswith('cpu') and reference and not item['fp16_hls']['bitwise_equal']:
     raise RuntimeError('FP16 hardware/HLS drift: stop further measurements')
  report['complete']=True;save();print('EVIDENCE='+str(ev))
 finally:save();c.close()
if __name__=='__main__':main()
