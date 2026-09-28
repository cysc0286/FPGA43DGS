"""Isolate board-computed attributes with the frozen official tile order."""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO/'examples/3dgs_compositor/board'))
from remote import connect, run
sys.path.insert(0, str(REPO/'examples/3dgs_scene'))
from check_output import compare
sys.path.insert(0, str(REPO/'examples/3dgs_flicker_cat'))
from analyze_results import quality
from validate_attributes import ATTR, GAUSSIAN, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--attributes', type=Path, required=True)
    p.add_argument('--label', required=True)
    a = p.parse_args()
    stage = json.loads((a.attributes/'result.json').read_text())
    validation = json.loads((a.attributes/'validation.json').read_text())
    if not stage['complete'] or len(validation['cases']) != 2:
        raise ValueError('Unvalidated board attributes')
    acceptance = json.loads((ROOT/'evidence/frontend_acceptance_20260927.json').read_text())
    program = json.loads((ROOT/'evidence/stagecat_20260927T122707/result.json').read_text())
    target = a.attributes/('hybrid_'+a.label)
    target.mkdir(exist_ok=False)
    record = {'attributes': str(a.attributes.resolve()), 'acceptance_sha256': sha(ROOT/'evidence/frontend_acceptance_20260927.json'),
              'program_sha256': program['binary_sha256'], 'cases': [], 'complete': False}
    client = connect()
    try:
        run(client, 'cat /proc/sys/kernel/random/boot_id', log=target/'environment.txt')
        if run(client, 'sha256sum '+program['remote']+'/render')[1].split()[0] != program['binary_sha256']:
            raise ValueError('Render executable drift')
        for item in stage['camera_runs']:
            camera = json.loads(Path(item['camera']).with_suffix('.json').read_text())
            view = camera['view']; key = f'view{view}'
            name = f'n559263_v{view}_w320'
            scene = REPO/'examples/3dgs_scene/data/20260926T135717'/name/'scene.bin'
            raw = bytearray(scene.read_bytes())
            w,h,tile,n,active,entries,tiles = struct.unpack_from('<7I', raw, 8)
            gs = np.frombuffer(raw, GAUSSIAN, active, 48)
            attrs = np.memmap(a.attributes/item['output'], dtype=ATTR, mode='r', offset=24, shape=(n,))
            ids = gs['id']
            if np.any(attrs['valid'][ids] != 1):
                raise ValueError('Missing official active point')
            # The frozen list is ordered by GPU depth. Replacing that key with
            # slightly different ARM depths would violate the sorted-list ABI.
            gs['q'][:,:9] = attrs['q'][ids,:9]
            prepared = target/(name+'.bin')
            prepared.write_bytes(raw)
            dest = program['remote']+'/frontend_'+a.attributes.name+'_'+name
            run(client, 'mkdir -p '+dest)
            with client.open_sftp() as sftp:
                sftp.put(str(prepared), dest+'/scene.bin')
            if run(client, 'sha256sum '+dest+'/scene.bin')[1].split()[0] != sha(prepared):
                raise ValueError('Hybrid input drift')
            prefix=dest+'/mode2'
            run(client, f'cd {dest} && timeout 90 {program["remote"]}/render mode 2 scene.bin mode2 pipeline 8 1 0',
                timeout=100,log=target/(name+'_run.txt'))
            with client.open_sftp() as sftp:
                sftp.get(prefix+'.bin',str(target/(name+'_render.bin')))
                sftp.get(prefix+'_timing.csv',str(target/(name+'_timing.csv')))
            output=target/(name+'_render.bin')
            official=scene.parent/'official.bin'
            prior=ROOT/'evidence/cat_measure_20260927T224323/received'/('cat_measure_20260927T224323_'+name+'_mode2.bin')
            check=compare(output,official)
            picture=quality(output,official)
            old=quality(prior,official)
            passed=(picture['psnr_clamped_rgb_db']>=acceptance['quality_vs_official_min_psnr_db'][key]
                    and picture['ssim_gaussian11_valid']>=acceptance['quality_vs_official_min_ssim'][key])
            entry={'case':name,'prepared_sha256':sha(prepared),'output_sha256':sha(output),
                   'depth_source':'official sorted-list key; board depth assessed in validation.json',
                   'timing_sha256':sha(target/(name+'_timing.csv')),
                   'official_compare':check,'quality':picture,'prior_quality':old,
                   'pass_quality_gate':bool(passed)}
            record['cases'].append(entry)
            (target/'result.json').write_text(json.dumps(record,indent=2))
            print(json.dumps({'case':name,'quality':picture,'prior':old,'passed':passed},indent=2),flush=True)
        record['complete']=all(x['pass_quality_gate'] for x in record['cases'])
        if not record['complete']:
            raise RuntimeError('Pre-frozen frontend quality gate failed; retain evidence')
    finally:
        (target/'result.json').write_text(json.dumps(record,indent=2))
        client.close()
        print('EVIDENCE='+str(target))


if __name__=='__main__':
    main()
