"""Preserve simulation evidence and check all final pixels against official data."""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from check_output import check

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--core',type=Path,required=True)
    p.add_argument('--register',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    count=manifest['vectors']['commands'];synthetic=manifest['synthetic']['commands']
    ev=ROOT/'evidence/simulation';ev.mkdir(parents=True,exist_ok=True)
    for path,name in [(a.core,'core'),(a.register,'register')]:
        log=(path/'simulation.log').read_text()
        assert f'{count} exact vectors' in log and 'FAIL:' not in log and 'FATAL:' not in log
        shutil.copy2(path/'simulation.log',ev/(name+'.txt'))
    actual=(a.core/'actual.txt').read_text().splitlines();assert len(actual)==count
    tile=ev/'tile_trace.txt';tile.write_text('\n'.join(actual[synthetic:])+'\n')
    syn=ev/'synthetic_trace.txt';syn.write_text('\n'.join(actual[:synthetic])+'\n')
    states=[[int(v,16) for v in s.split()] for s in actual[synthetic:]]
    cmds=[[int(v,16) for v in s.split()] for s in (ROOT/'data/tile_input.txt').read_text().splitlines()]
    cycles=Counter(s[6] for s in states)
    total=sum(s[7] for c,s in zip(cmds,states) if c[0]==2)
    assert total==sum(s[6] for s in states)
    result={'backend':'RTL simulation; not physical FPGA',
      'rtl_sha256':hashlib.sha256((ROOT/'rtl/gs_compositor.v').read_bytes()).hexdigest(),
      'core_and_register_exact_commands':count,'synthetic':check(syn,'synthetic','trace'),
      'tile':check(tile,'tile','trace'),'tile_command_cycles_histogram':dict(cycles),
      'tile_total_active_cycles':total,'clock_target_hz':100000000,
      'tile_active_time_estimate_ms_at_target':total/100000,
      'scope':'Active core cycles exclude host, transport, idle and all Gaussian evaluation'}
    (ev/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
