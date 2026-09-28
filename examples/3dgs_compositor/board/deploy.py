"""Compile input-only ARM replay, run CPU or GSC1 PL, verify retrieved results."""
import argparse
import hashlib
import json
import os
import shlex
import sys
import time
from pathlib import Path
EXAMPLE=Path(__file__).resolve().parents[1];REPO=EXAMPLE.parents[1]
sys.path.insert(0,str(EXAMPLE))
from remote import connect,run
from check_output import check

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['cpu','fpga'],default='cpu')
    p.add_argument('--existing',help='Reuse a compiled remote directory');a=p.parse_args()
    stamp=time.strftime('%Y%m%d_%H%M%S');local=EXAMPLE/'evidence'/f'{a.backend}_{stamp}';local.mkdir(parents=True)
    remote=a.existing or '/root/fpga43dgs_compositor/'+stamp
    c=connect()
    try:
        _,identity=run(c,'uname -m; cat /proc/sys/kernel/random/boot_id; g++ --version | head -1',log=local/'identity.txt')
        if not identity.startswith('aarch64'):raise RuntimeError('Not the ARM board')
        run(c,'mkdir -p '+shlex.quote(remote))
        files=[EXAMPLE/'board/replay.cpp',EXAMPLE/'data/tile_input.txt',EXAMPLE/'data/synthetic_input.txt']
        with c.open_sftp() as s:
            for f in files:s.put(str(f),remote+'/'+f.name)
        prefix='cd '+shlex.quote(remote)+' && '
        _,hashes=run(c,prefix+'sha256sum replay.cpp tile_input.txt synthetic_input.txt',log=local/'input_hashes.txt')
        actual={line.split()[1]:line.split()[0] for line in hashes.splitlines()}
        for f in files:
            if actual.get(f.name)!=hashlib.sha256(f.read_bytes()).hexdigest():raise RuntimeError('upload mismatch')
        binary='replay_cpu' if a.backend=='cpu' else 'replay_fpga'
        flags='-DCPU_ONLY' if a.backend=='cpu' else '-licraft_xrt -licraft_xir -licraft_utils -ldw -pthread -ldl'
        run(c,prefix+f'timeout 180 g++ -std=gnu++17 -O2 -Wall -Wextra replay.cpp -o {binary} '+flags,
            timeout=195,log=local/'build.txt')
        results=[]
        for kind,mode,repeats in [('synthetic','trace',1),('tile','trace',1),('tile','tile',20 if a.backend=='cpu' else 5)]:
            name=f'{kind}_{mode}.txt'
            _,log=run(c,prefix+f'timeout 600 ./{binary} {a.backend} {kind}_input.txt {name} {mode} {repeats}',
                        timeout=620,log=local/f'{kind}_{mode}_timing.txt')
            with c.open_sftp() as s:s.get(remote+'/'+name,str(local/name))
            result=check(local/name,kind,mode,hardware_cycles=a.backend=='fpga')
            timing=json.loads(next(line for line in log.splitlines() if line.startswith('{"backend"')))
            result.update(kind=kind,mode=mode,timing=timing);results.append(result)
            (local/f'{kind}_{mode}.json').write_text(json.dumps(result,indent=2)+'\n')
            if not result['passed']:raise RuntimeError('Numerical check failed: '+name)
        run(c,prefix+'sha256sum '+binary,log=local/'binary_hash.txt')
        summary={'backend':a.backend,'remote':remote,'expected_uploaded':False,'source_sha256':actual['replay.cpp'],
                  'runs':results,'passed':all(r['passed'] for r in results)}
        (local/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2));print('EVIDENCE='+str(local))
    finally:c.close()

if __name__=='__main__':main()
