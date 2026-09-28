"""Compile, run and retrieve board-side tile grouping/sorting."""
import argparse
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'examples/3dgs_compositor/board'))
from remote import connect,run


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--attributes',type=Path,required=True)
    choice=p.add_mutually_exclusive_group()
    choice.add_argument('--radix',action='store_true')
    choice.add_argument('--global-sort',action='store_true')
    choice.add_argument('--packed-sort',action='store_true')
    a=p.parse_args()
    prior=json.loads((a.attributes/'result.json').read_text())
    if not prior['complete'] or len(prior['camera_runs'])!=2:
        raise ValueError('Board attributes not complete')
    source=ROOT/'frontend/group_sort.cpp'
    stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    evidence=ROOT/'evidence'/('frontend_group_'+stamp)
    evidence.mkdir()
    shutil.copy2(source,evidence/'group_sort.cpp')
    dest='/root/fpga43dgs_flicker/frontend_group_'+stamp
    record={'attribute_evidence':str(a.attributes.resolve()),'source_sha256':sha(source),
            'sort_variant':('packed-depth-id-sort' if a.packed_sort else
                            'global-stable-sort' if a.global_sort else
                            'stable-radix4' if a.radix else 'std-stable-sort'),
            'remote':dest,'cases':[],'complete':False}
    c=connect()
    try:
        run(c,'mkdir -p '+dest)
        run(c,'cat /proc/sys/kernel/random/boot_id; uname -a',log=evidence/'environment.txt')
        with c.open_sftp() as s:s.put(str(source),dest+'/group_sort.cpp')
        if run(c,'sha256sum '+dest+'/group_sort.cpp')[1].split()[0]!=record['source_sha256']:
            raise ValueError('Source upload drift')
        flag=(' -DPACKED_SORT' if a.packed_sort else
              ' -DGLOBAL_SORT' if a.global_sort else
              ' -DRADIX_SORT' if a.radix else '')
        run(c,'g++ -O2 -std=gnu++17 -ffp-contract=off -Wall -Wextra'+flag+' '+dest+'/group_sort.cpp -o '+dest+'/group_sort',
            timeout=120,log=evidence/'compile.txt')
        record['binary_sha256']=run(c,'sha256sum '+dest+'/group_sort')[1].split()[0]
        for item in prior['camera_runs']:
            name=Path(item['output']).stem
            input_path=a.attributes/item['output']
            with c.open_sftp() as s:s.put(str(input_path),dest+'/'+item['output'])
            if run(c,'sha256sum '+dest+'/'+item['output'])[1].split()[0]!=item['output_sha256']:
                raise ValueError('Attribute upload drift')
            command=f'timeout 90 {dest}/group_sort {dest}/{item["output"]} {dest}/{name}.scene'
            run(c,command,timeout=100,log=evidence/(name+'_run.txt'))
            with c.open_sftp() as s:s.get(dest+'/'+name+'.scene',str(evidence/(name+'.scene')))
            entry={'name':name,'input_sha256':item['output_sha256'],
                   'scene_sha256':sha(evidence/(name+'.scene')),
                   'scene_bytes':(evidence/(name+'.scene')).stat().st_size,
                   'timing':(evidence/(name+'_run.txt')).read_text().strip()}
            record['cases'].append(entry)
            (evidence/'result.json').write_text(json.dumps(record,indent=2))
            print(json.dumps(entry),flush=True)
        record['complete']=True
    finally:
        (evidence/'result.json').write_text(json.dumps(record,indent=2))
        c.close()
        print('EVIDENCE='+str(evidence))


if __name__=='__main__':main()
