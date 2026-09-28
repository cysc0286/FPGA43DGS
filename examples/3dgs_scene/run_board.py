"""Timestamped whole-image board workloads; only inputs uploaded, no Golden."""
import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'examples/3dgs_compositor/board'))
import remote
from check_output import sha,compare,timing
SDK='/root/heterogs_npu/sdk_3.36.1/usr'
LIB=SDK+'/lib/aarch64-linux-gnu'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',choices=['calibrate','measure'],default='calibrate')
    ap.add_argument('--reuse-build',type=Path)
    ap.add_argument('--cases',nargs='*')
    ap.add_argument('--backends',nargs='*',default=['float1','fixed1','float4','fixed4','fpga1'])
    ap.add_argument('--cpu-repeats',type=int,default=10)
    ap.add_argument('--fpga-repeats',type=int,default=3)
    args=ap.parse_args()
    data=Path(json.loads((ROOT/'data/current.json').read_text())['directory'])
    manifest=json.loads((data/'manifest.json').read_text())
    stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    ev=ROOT/'evidence'/(args.phase+'_'+stamp);ev.mkdir(parents=True)
    (ev/'sources').mkdir();(ev/'received').mkdir()
    dest='/root/fpga43dgs_scene/'+stamp
    sources=[ROOT/'scene_renderer.cpp',REPO/'examples/3dgs_compositor/board/replay.cpp',
             REPO/'examples/3dgs_batch/board/batch.hpp']
    flags=(f'-O3 -std=gnu++17 -ffp-contract=off -fopenmp -Wall -Wextra '
        f'-I{SDK}/include -L{LIB} -Wl,-rpath,{LIB} '
        '-licraft_zg330backend -licraft_hostbackend -licraft_xrt -licraft_xir -licraft_utils -ldw -ldl -pthread')
    report=dict(schema='gs-large-prepared-frame-v1',phase=args.phase,remote_directory=dest,
        data=str(data.resolve()),manifest_sha256=sha(data/'manifest.json'),source_hashes={p.name:sha(p) for p in sources},
        compile_flags=flags,completed=False,correctness_passed=False,scope='complete prepared-frame rasterization; GPU preprocessing excluded',runs=[])
    def save():(ev/'result.json').write_text(json.dumps(report,indent=2))
    save();c=remote.connect()
    try:
        report['identity']=remote.run(c,'uname -a; cat /proc/sys/kernel/random/boot_id; free -k; nproc')[1]
        remote.run(c,'mkdir -p '+dest)
        with c.open_sftp() as s:
            for p in sources:
                shutil.copy2(p,ev/'sources'/p.name);s.put(str(p),dest+'/'+p.name)
                if remote.run(c,'sha256sum '+dest+'/'+p.name)[1].split()[0]!=sha(p):raise ValueError('Source upload')
            for name in ('run_board.py','check_output.py'):shutil.copy2(ROOT/name,ev/'sources'/name)
            if args.reuse_build:
                old=json.loads((args.reuse_build/'result.json').read_text())
                if old['source_hashes']!=report['source_hashes'] or old['compile_flags']!=flags:raise ValueError('Compile provenance')
                binary=old['remote_directory']+'/scene_renderer'
                if remote.run(c,'sha256sum '+binary)[1].split()[0]!=old['binary_sha256']:raise ValueError('Binary provenance')
                remote.run(c,f'cp {binary} {dest}/scene_renderer')
            else:
                print('Native build '+str(ev),flush=True)
                remote.run(c,f'cd {dest} && g++ scene_renderer.cpp -o scene_renderer {flags}',timeout=180,log=ev/'build.log')
            report['binary_sha256']=remote.run(c,'sha256sum '+dest+'/scene_renderer')[1].split()[0];save()
            for case in manifest['cases']:
                name=case['name']
                if args.cases and name not in args.cases:continue
                p=data/name/'scene.bin'
                if sha(p)!=case['input_sha256'] or sha(data/name/'official.bin')!=case['official_sha256']:raise ValueError('Input/Golden drift')
                s.put(str(p),dest+'/'+name+'.bin')
                if remote.run(c,'sha256sum '+dest+'/'+name+'.bin')[1].split()[0]!=case['input_sha256']:raise ValueError('Input upload')
                for variant in args.backends:
                    if variant not in ('float1','fixed1','float4','fixed4','fpga1'):raise ValueError('Backend')
                    mode,threads=variant[:-1],int(variant[-1])
                    repeats=1 if args.phase=='calibrate' else (args.fpga_repeats if mode=='fpga' else args.cpu_repeats)
                    warmup=0 if args.phase=='calibrate' else 1
                    stem=name+'_'+variant
                    print('RUN '+stem,flush=True)
                    command=(f'cd {dest} && timeout 600s env LD_LIBRARY_PATH={LIB} ./scene_renderer '
                             f'{name}.bin {stem} {mode} {threads} {repeats} {warmup}')
                    remote.run(c,command,timeout=610,log=ev/(stem+'.log'))
                    suffixes=['.bin','_timing.csv','_meta.json']+(['_states.u32'] if mode!='float' else [])
                    for suffix in suffixes:s.get(dest+'/'+stem+suffix,str(ev/'received'/(stem+suffix)))
                    output=ev/'received'/(stem+'.bin')
                    result=dict(case=case,variant=variant,stem=stem,
                        official=compare(output,data/name/'official.bin'),timing=timing(ev/'received'/(stem+'_timing.csv')),
                        metadata=json.loads((ev/'received'/(stem+'_meta.json')).read_text()),
                        files={suffix:sha(ev/'received'/(stem+suffix)) for suffix in suffixes})
                    # A failed official gate stays visible; collect all backends to diagnose causality.
                    if mode!='float':
                        baseline=ev/'received'/(name+'_fixed1_states.u32')
                        if baseline.exists():result['fixed1_integer_exact']=sha(baseline)==sha(ev/'received'/(stem+'_states.u32'))
                    report['runs'].append(result);save()
                    print(json.dumps({'run':stem,'mean_ms':result['timing']['mean_ms'],
                        'official_pass':result['official']['passed'],'rgb_max':result['official']['rgb_max'],
                        'last_mismatches':result['official']['last_mismatches'],
                        'unique_contributors':result['timing']['means']['unique_contributors'],
                        'fixed_exact':result.get('fixed1_integer_exact')}),flush=True)
        report['completed']=True
        report['correctness_passed']=all(r['official']['passed'] and r.get('fixed1_integer_exact',True) for r in report['runs'])
        save();print('EVIDENCE='+str(ev),flush=True)
    except Exception as e:report['error']=str(e);save();raise
    finally:c.close()


if __name__=='__main__':main()
