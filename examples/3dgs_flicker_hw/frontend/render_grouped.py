"""Render board-built GSSCN001 scenes through the installed FPGA path."""
import argparse
import hashlib
import json
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
    p=argparse.ArgumentParser();p.add_argument('--group',type=Path,required=True);a=p.parse_args()
    stage=json.loads((a.group/'result.json').read_text())
    check=json.loads((a.group/'validation.json').read_text())
    acceptance_path=ROOT/'evidence/frontend_full_acceptance_20260927.json'
    acceptance=json.loads(acceptance_path.read_text())
    program=json.loads((ROOT/'evidence/stagecat_20260927T122707/result.json').read_text())
    if not stage['complete'] or len(check['cases'])!=2:
        raise ValueError('Incomplete board grouping')
    output_dir=a.group/'rendered'
    output_dir.mkdir(exist_ok=False)
    report={'group_evidence':str(a.group.resolve()),'acceptance_sha256':sha(acceptance_path),
            'program_sha256':program['binary_sha256'],'cases':[],'complete':False}
    c=connect()
    try:
        run(c,'cat /proc/sys/kernel/random/boot_id',log=output_dir/'environment.txt')
        if run(c,'sha256sum '+program['remote']+'/render')[1].split()[0]!=program['binary_sha256']:
            raise ValueError('Program drift')
        for item in stage['cases']:
            view=int(item['name'][1:]);name=f'n559263_v{view}_w320'
            comparison=next(x for x in check['cases'] if x['case']==name)
            scene=a.group/(item['name']+'.scene')
            if sha(scene)!=item['scene_sha256']:raise ValueError('Scene drift')
            dest=program['remote']+'/group_'+a.group.name+'_'+name
            run(c,'mkdir -p '+dest)
            with c.open_sftp() as s:s.put(str(scene),dest+'/scene.bin')
            if run(c,'sha256sum '+dest+'/scene.bin')[1].split()[0]!=item['scene_sha256']:
                raise ValueError('Scene upload drift')
            run(c,f'cd {dest} && timeout 90 {program["remote"]}/render mode 2 scene.bin frame pipeline 8 1 0',
                timeout=100,log=output_dir/(name+'_run.txt'))
            with c.open_sftp() as s:
                s.get(dest+'/frame.bin',str(output_dir/(name+'.bin')))
                s.get(dest+'/frame_timing.csv',str(output_dir/(name+'_timing.csv')))
            actual=output_dir/(name+'.bin')
            official=REPO/'examples/3dgs_scene/data/20260926T135717'/name/'official.bin'
            baseline=ROOT/'evidence/cat_measure_20260927T224323/received'/('cat_measure_20260927T224323_'+name+'_mode2.bin')
            q=quality(actual,official);prior=quality(baseline,official)
            passed=(comparison['missing_instances']<=acceptance['maximum_missing_tile_instances_per_view']
                    and comparison['duplicate_tiles']==acceptance['required_duplicate_tiles']
                    and q['psnr_clamped_rgb_db']>=acceptance['quality_vs_official_min_psnr_db'][f'view{view}']
                    and q['ssim_gaussian11_valid']>=acceptance['quality_vs_official_min_ssim'][f'view{view}'])
            result={'case':name,'scene_sha256':item['scene_sha256'],
                    'frame_sha256':sha(actual),'timing_sha256':sha(output_dir/(name+'_timing.csv')),
                    'official_compare':compare(actual,official),'quality':q,'prior_quality':prior,
                    'list_validation':comparison,'passed':bool(passed)}
            report['cases'].append(result)
            (output_dir/'result.json').write_text(json.dumps(report,indent=2))
            print(json.dumps({'case':name,'quality':q,'prior':prior,'passed':passed},indent=2),flush=True)
        report['complete']=all(x['passed'] for x in report['cases'])
        if not report['complete']:raise RuntimeError('Full board-grouped quality gate failed')
    finally:
        (output_dir/'result.json').write_text(json.dumps(report,indent=2))
        c.close()
        print('EVIDENCE='+str(output_dir))


if __name__=='__main__':main()
