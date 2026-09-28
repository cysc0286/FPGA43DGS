"""Vendor adder and SDK block regression for a FLK1 image without old GSC1/GSB1.

Only the platform identification check is adapted. The existing arithmetic,
allocation guards, block sizes, repetitions and byte comparisons are preserved.
"""
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'examples/3dgs_compositor/board'))
from remote import connect,run


def main():
    stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    ev=ROOT/'evidence'/('lean_block_'+stamp)
    ev.mkdir()
    baseline=ROOT/'board/probe_block.cpp'
    source=baseline.read_text()
    old='rd(0xc0)!=0x20230628||rd(0x100)!=0x47534231'
    new='rd(0xc0)!=0x20230628||rd(0x200)!=0x464c4b31||rd(0x204)!=0x20000||rd(0x9c)!=0||rd(0x100)!=0xffffffff'
    if source.count(old)!=1:
        raise ValueError('Unexpected probe baseline')
    local=ev/'probe.cpp'
    local.write_text(source.replace(old,new),newline='\n')
    dest='/root/fpga43dgs_flicker/lean_block_'+stamp
    record={'baseline_sha256':hashlib.sha256(baseline.read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(local.read_bytes()).hexdigest(),
            'changes':['platform capability guard only'],'remote':dest,'passed':False}
    client=connect()
    try:
        run(client,'mkdir -p '+dest)
        run(client,'uname -a; cat /proc/sys/kernel/random/boot_id',log=ev/'environment.txt')
        with client.open_sftp() as sftp:
            sftp.put(str(local),dest+'/probe.cpp')
        sdk='/root/heterogs_npu/sdk_3.36.1/usr'
        lib=sdk+'/lib/aarch64-linux-gnu'
        command=f'g++ -O2 -std=gnu++17 -Wall -Wextra -I{sdk}/include {dest}/probe.cpp -L{lib} -Wl,-rpath,{lib} -licraft_zg330backend -licraft_hostbackend -licraft_xrt -licraft_xir -licraft_utils -ldw -ldl -pthread -o {dest}/probe'
        run(client,command,timeout=180,log=ev/'compile.txt')
        rc,out=run(client,f'flock -n /run/lock/fpga43dgs-gsc1.lock timeout 20 {dest}/probe',timeout=30,log=ev/'run.txt',check=False)
        record['returncode']=rc
        record['samples']=[json.loads(line) for line in out.splitlines() if line.startswith('{')]
        record['passed']=rc==0 and len(record['samples'])==18 and sum(row.get('exact',False) for row in record['samples'])==15
        print(out)
        if not record['passed']:
            raise RuntimeError('SDK regression failed; inspect before retry')
    finally:
        (ev/'result.json').write_text(json.dumps(record,indent=2))
        client.close()
        print('EVIDENCE='+str(ev))


if __name__=='__main__':
    main()
