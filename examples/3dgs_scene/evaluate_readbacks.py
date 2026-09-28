"""Upload frozen calibration readbacks to the existing GPU for image metrics."""
import datetime
import json
import os
import shutil
import sys
from pathlib import Path
import paramiko
from check_output import sha
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'examples/3dgs_compositor/board'))
from remote import run


def main():
    stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    ev=ROOT/'evidence'/('quality_'+stamp);ev.mkdir(parents=True)
    base='/root/autodl-tmp/3dgs_reproduction'
    dest=base+'/scene_quality_'+stamp
    data=Path(json.loads((ROOT/'data/current.json').read_text())['directory'])
    frames=json.loads((data/'receipt.json').read_text())['remote']+'/frames'
    plan=[]
    for name in ['calibrate_20260926T140258','calibrate_20260926T140613']:
        source=ROOT/'evidence'/name
        for r in json.loads((source/'result.json').read_text())['runs']:
            if r['variant'] not in ('float4','fpga1'):continue
            p=source/'received'/(r['stem']+'.bin')
            if sha(p)!=r['files']['.bin']:raise ValueError('Input readback drift')
            plan.append(dict(file=p.name,case=r['case']['name'],variant=r['variant'],
                local_source=str(p),actual_sha256=sha(p)))
    (ev/'plan.json').write_text(json.dumps(plan,indent=2))
    c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
    c.connect('connect.bjb1.seetacloud.com',port=32310,username='root',password=os.environ['GS_CLOUD_PASSWORD'],
        look_for_keys=False,allow_agent=False,timeout=8,banner_timeout=8,auth_timeout=8)
    try:
        run(c,'mkdir -p '+dest)
        with c.open_sftp() as s:
            for filename in ('quality_gpu.py','check_output.py'):
                shutil.copy2(ROOT/filename,ev/filename)
                s.put(str(ev/filename),dest+'/'+filename)
            s.put(str(ev/'plan.json'),dest+'/plan.json')
            for row in plan:s.put(row['local_source'],dest+'/'+row['file'])
            cmd=(f'cd {base} && TORCH_HOME={base}/cache/torch ./venv/bin/python {dest}/quality_gpu.py '
                f'--source {base}/source_pinned --frames {frames} --directory {dest}')
            print(run(c,cmd,timeout=180,log=ev/'quality.log')[1],flush=True)
            s.get(dest+'/quality.json',str(ev/'quality.json'))
        (ev/'receipt.json').write_text(json.dumps(dict(remote=dest,source_sha256=sha(ev/'quality_gpu.py'),
            output_sha256=sha(ev/'quality.json')),indent=2))
        print('QUALITY='+str(ev),flush=True)
    finally:c.close()


if __name__=='__main__':main()
