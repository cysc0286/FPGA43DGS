"""Freeze dependencies, benchmark the same full board pipeline and collect results."""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'examples/3dgs_compositor/board'))
from remote import connect,run


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--attributes',type=Path,required=True)
    p.add_argument('--group',type=Path,required=True);p.add_argument('--repeats',type=int,default=5);a=p.parse_args()
    attr=json.loads((a.attributes/'result.json').read_text())
    group=json.loads((a.group/'result.json').read_text())
    program=json.loads((ROOT/'evidence/stagecat_20260927T122707/result.json').read_text())
    if not attr['complete'] or not group['complete'] or not 1<=a.repeats<=10:raise ValueError('Invalid completed stages/repeat count')
    stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S');ev=ROOT/'evidence'/('full_benchmark_'+stamp);ev.mkdir()
    dest='/root/fpga43dgs_flicker/'+ev.name
    model='/root/fpga43dgs_flicker/model_8212d03d/point_cloud.ply'
    config={'model':model,'attributes':attr['remote']+'/attributes','group':group['remote']+'/group_sort',
            'renderer':program['remote']+'/render','cameras':{},'scene_hashes':{},
            'backends':['fpga','cpu_dense','cpu_base'],'repeats':a.repeats,'hashes':{}}
    for path,digest in [(model,attr['model_sha256']),(config['attributes'],attr['binary_sha256']),
                        (config['group'],group['binary_sha256']),(config['renderer'],program['binary_sha256'])]:config['hashes'][path]=digest
    for item in attr['camera_runs']:
        camera=Path(item['camera']);view=camera.stem[1:];path=dest+'/'+camera.name
        config['cameras'][view]=path;config['hashes'][path]=item['camera_sha256']
        config['scene_hashes'][view]=next(x['scene_sha256'] for x in group['cases'] if x['name']==camera.stem)
    local=ev/'config.json';local.write_text(json.dumps(config,indent=2))
    source=ROOT/'frontend/board_benchmark.py';(ev/source.name).write_bytes(source.read_bytes())
    record={'remote':dest,'config_sha256':sha(local),'script_sha256':sha(source),
            'attributes':str(a.attributes.resolve()),'group':str(a.group.resolve()),'complete':False}
    c=connect()
    try:
        run(c,'mkdir -p '+dest)
        run(c,'cat /proc/sys/kernel/random/boot_id; uname -a; df -h /dev/shm; free -m',log=ev/'environment.txt')
        with c.open_sftp() as s:
            s.put(str(local),dest+'/config.json');s.put(str(source),dest+'/board_benchmark.py')
            for item in attr['camera_runs']:
                camera=Path(item['camera']);s.put(str(camera),dest+'/'+camera.name)
        for path,digest in [(dest+'/config.json',record['config_sha256']),(dest+'/board_benchmark.py',record['script_sha256'])]:
            if run(c,'sha256sum '+path)[1].split()[0]!=digest:raise ValueError('Benchmark source drift')
        print('EVIDENCE='+str(ev),flush=True)
        run(c,'python3 '+dest+'/board_benchmark.py '+dest+'/config.json',timeout=600,log=ev/'board_console.txt')
        with c.open_sftp() as s:
            for name in s.listdir(dest):
                if name.endswith(('.csv','.txt','.bin')) or name=='result.json':
                    s.get(dest+'/'+name,str(ev/('board_result.json' if name=='result.json' else name)))
        board=json.loads((ev/'board_result.json').read_text())
        if not board['complete'] or len(board['rows'])!=a.repeats*6:raise ValueError('Incomplete benchmark')
        record['complete']=True;record['board_result_sha256']=sha(ev/'board_result.json')
        print('COMPLETED='+str(ev))
    finally:
        (ev/'result.json').write_text(json.dumps(record,indent=2));c.close()


if __name__=='__main__':main()
