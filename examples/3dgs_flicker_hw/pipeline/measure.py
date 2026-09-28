"""FLK1 CPU/base/AABB/CAT ablation, frozen scene and executable per run."""
import argparse,datetime,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'examples/3dgs_compositor/board'))
from remote import connect,run
sys.path.insert(0,str(REPO/'examples/3dgs_scene'))
from check_output import compare
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True,type=Path);p.add_argument('--backends',nargs='+',default=['cpu1','cpu4','mode0','mode1','mode2','mode3','mode4','mode5']);p.add_argument('--cases',nargs='+',default=['n559263_v0_w320','n559263_v10_w320']);p.add_argument('--repeats',type=int,default=3);p.add_argument('--warmup',type=int,default=1);p.add_argument('--schedule',choices=['serial','pipeline'],default='pipeline');a=p.parse_args()
 stage=json.loads((a.stage/'result.json').read_text());dest=stage['remote']
 if not stage['compiled']:raise ValueError('Unbuilt stage')
 ev=ROOT/'evidence'/('cat_measure_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S'));ev.mkdir();(ev/'received').mkdir()
 report={'stage':str(a.stage),'stage_record':stage,'schedule':a.schedule,'runs':[],'complete':False,'scope':'prepared frame including packing/transport/synchronization/readback, not pose-to-frame'}
 def save():(ev/'result.json').write_text(json.dumps(report,indent=2))
 save();c=connect()
 try:
  if run(c,'sha256sum '+dest+'/render')[1].split()[0]!=stage['binary_sha256']:raise ValueError('Executable drift')
  run(c,'uname -a; cat /proc/sys/kernel/random/boot_id; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor',log=ev/'environment.txt',check=False)
  for case in a.cases:
   match=next(x for x in stage['cases'] if x['name']==case)
   if run(c,'sha256sum '+dest+'/'+case+'.bin')[1].split()[0]!=match['sha256']:raise ValueError('Scene drift')
   for backend in a.backends:
    stem=ev.name+'_'+case+'_'+backend
    if backend in ['cpu1','cpu4']:args=f'cpu {case}.bin {stem} base {backend[-1]} {a.repeats} {a.warmup} 0'
    elif backend=='cpu_dense4':args=f'cpu {case}.bin {stem} dense 4 {a.repeats} {a.warmup} 0'
    elif backend in [f'mode{x}' for x in range(6)]:args=f'mode {backend[-1]} {case}.bin {stem} {a.schedule} 8 {a.repeats} {a.warmup}'
    else:raise ValueError('Unknown backend')
    print('RUN '+stem,flush=True)
    run(c,f'cd {dest} && timeout 600 ./render {args}',timeout=610,log=ev/(stem+'.txt'))
    suffixes=['.bin','_timing.csv']+([] if backend.startswith('cpu') else ['_trace.csv'])
    with c.open_sftp() as s:
     for suffix in suffixes:s.get(dest+'/'+stem+suffix,str(ev/'received'/(stem+suffix)))
    actual=ev/'received'/(stem+'.bin');official=REPO/'examples/3dgs_scene/data/20260926T135717'/case/'official.bin'
    item={'case':case,'backend':backend,'stem':stem,'official':compare(actual,official)}
    if not backend.startswith('cpu'):
     folder=ROOT/'evidence'/('pipeline_full_v0' if case=='n559263_v0_w320' else 'pipeline_full_v10')
     expected=folder/f'frame_mode{backend[-1]}.bin';item['fp16_hls']=compare(actual,expected);item['golden_sha256']=hashlib.sha256(expected.read_bytes()).hexdigest()
    report['runs'].append(item);save();print(json.dumps(item),flush=True)
    if not backend.startswith('cpu') and not item['fp16_hls']['bitwise_equal']:raise RuntimeError('Hardware differs from selected-mode HLS golden')
  report['complete']=True;save();print('EVIDENCE='+str(ev))
 finally:save();c.close()
if __name__=='__main__':main()
