"""One board invocation from official model/camera through the FPGA framebuffer."""
import argparse
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'examples/3dgs_compositor/board'))
from remote import connect,run
sys.path.insert(0,str(REPO/'examples/3dgs_scene'))
from check_output import compare
sys.path.insert(0,str(REPO/'examples/3dgs_flicker_cat'))
from analyze_results import quality


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--attributes',type=Path,required=True)
    p.add_argument('--group',type=Path,required=True)
    p.add_argument('--backend',choices=['fpga','cpu_dense','cpu_base'],required=True)
    a=p.parse_args()
    attr=json.loads((a.attributes/'result.json').read_text())
    grouped=json.loads((a.group/'result.json').read_text())
    program=json.loads((ROOT/'evidence/stagecat_20260927T122707/result.json').read_text())
    acceptance_path=ROOT/'evidence/frontend_full_acceptance_20260927.json'
    acceptance=json.loads(acceptance_path.read_text())
    if not attr['complete'] or not grouped['complete']:
        raise ValueError('Missing board stage')
    stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    ev=ROOT/'evidence'/('full_pipeline_'+a.backend+'_'+stamp);ev.mkdir()
    script=ROOT/'frontend/full_pipeline.sh'
    model='/root/fpga43dgs_flicker/model_8212d03d/point_cloud.ply'
    attr_bin=attr['remote']+'/attributes'
    group_bin=grouped['remote']+'/group_sort'
    render_bin=program['remote']+'/render'
    remote_root='/root/fpga43dgs_flicker/'+ev.name
    result={'script_sha256':sha(script),'acceptance_sha256':sha(acceptance_path),
            'backend':a.backend,
            'attribute_binary_sha256':attr['binary_sha256'],
            'group_binary_sha256':grouped['binary_sha256'],
            'render_binary_sha256':program['binary_sha256'],
            'model_sha256':attr['model_sha256'],'cases':[],'complete':False}
    c=connect()
    try:
        run(c,'mkdir -p '+remote_root)
        run(c,'cat /proc/sys/kernel/random/boot_id; uname -a',log=ev/'environment.txt')
        for remote_path,digest in ((model,attr['model_sha256']),
                                   (attr_bin,attr['binary_sha256']),
                                   (group_bin,grouped['binary_sha256']),
                                   (render_bin,program['binary_sha256'])):
            if run(c,'sha256sum '+remote_path)[1].split()[0]!=digest:
                raise ValueError('Board source/binary drift: '+remote_path)
        with c.open_sftp() as s:s.put(str(script),remote_root+'/full_pipeline.sh')
        if run(c,'sha256sum '+remote_root+'/full_pipeline.sh')[1].split()[0]!=result['script_sha256']:
            raise ValueError('Orchestration script drift')
        for item in attr['camera_runs']:
            camera=Path(item['camera']);view=int(camera.stem[1:]);name=f'n559263_v{view}_w320'
            if sha(camera)!=item['camera_sha256']:
                raise ValueError('Camera drift')
            work=remote_root+'/'+name
            with c.open_sftp() as s:s.put(str(camera),remote_root+'/'+camera.name)
            if run(c,'sha256sum '+remote_root+'/'+camera.name)[1].split()[0]!=item['camera_sha256']:
                raise ValueError('Camera upload drift')
            command=('sh '+remote_root+'/full_pipeline.sh '+model+' '+remote_root+'/'+camera.name
                     +' '+attr_bin+' '+group_bin+' '+render_bin+' '+work+' '+a.backend)
            text=run(c,command,timeout=120,log=ev/(name+'_run.txt'))[1]
            stamp_line=re.search(r'TIMING_NS start=(\d+) after_attributes=(\d+) after_group=(\d+) after_render=(\d+)',text)
            if not stamp_line:raise ValueError('Missing end-to-end timing')
            times=list(map(int,stamp_line.groups()))
            if times!=sorted(times):raise ValueError('Nonmonotonic timestamps')
            with c.open_sftp() as s:
                for remote_name,local_name in [('frame.bin',name+'.bin'),('frame_timing.csv',name+'_timing.csv'),
                                               ('scene.bin',name+'.scene')]:
                    s.get(work+'/'+remote_name,str(ev/local_name))
            official=REPO/'examples/3dgs_scene/data/20260926T135717'/name/'official.bin'
            frame=ev/(name+'.bin');q=quality(frame,official)
            minimum_psnr=acceptance['quality_vs_official_min_psnr_db'][f'view{view}']
            minimum_ssim=acceptance['quality_vs_official_min_ssim'][f'view{view}']
            passed=q['psnr_clamped_rgb_db']>=minimum_psnr and q['ssim_gaussian11_valid']>=minimum_ssim
            entry={'case':name,'camera_sha256':item['camera_sha256'],
                   'frame_sha256':sha(frame),'scene_sha256':sha(ev/(name+'.scene')),
                   'timing_sha256':sha(ev/(name+'_timing.csv')),
                   'attributes_ms':(times[1]-times[0])/1e6,
                   'group_sort_ms':(times[2]-times[1])/1e6,
                   'render_wall_ms':(times[3]-times[2])/1e6,
                   'pose_to_frame_ms':(times[3]-times[0])/1e6,
                   'render_invocations':1,'render_warmup':0,
                   'official_compare':compare(frame,official),
                   'quality':q,'passed':bool(passed)}
            result['cases'].append(entry)
            (ev/'result.json').write_text(json.dumps(result,indent=2))
            print(json.dumps(entry,indent=2),flush=True)
        result['complete']=all(x['passed'] for x in result['cases'])
        if not result['complete']:raise RuntimeError('Frozen full-pipeline quality gate failed')
    finally:
        (ev/'result.json').write_text(json.dumps(result,indent=2))
        c.close()
        print('EVIDENCE='+str(ev))


if __name__=='__main__':main()
