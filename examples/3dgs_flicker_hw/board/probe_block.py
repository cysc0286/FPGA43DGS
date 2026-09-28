"""Run bounded vendor block-transport diagnostic and retain raw evidence."""
import datetime,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'examples/3dgs_compositor/board'))
from remote import connect,run
def main():
 stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
 ev=ROOT/'evidence'/('block_'+stamp);ev.mkdir(parents=True)
 src=ROOT/'board/probe_block.cpp';dest='/root/fpga43dgs_flicker/block_'+stamp
 report={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'remote':dest,'passed':False}
 c=connect()
 try:
  run(c,'mkdir -p '+dest)
  run(c,'uname -a; cat /proc/sys/kernel/random/boot_id',log=ev/'environment.txt')
  with c.open_sftp() as s:s.put(str(src),dest+'/probe.cpp')
  sdk='/root/heterogs_npu/sdk_3.36.1/usr';lib=sdk+'/lib/aarch64-linux-gnu'
  cmd=f'g++ -O2 -std=gnu++17 -Wall -Wextra -I{sdk}/include {dest}/probe.cpp -L{lib} -Wl,-rpath,{lib} -licraft_zg330backend -licraft_hostbackend -licraft_xrt -licraft_xir -licraft_utils -ldw -ldl -pthread -o {dest}/probe'
  run(c,cmd,timeout=180,log=ev/'compile.txt')
  rc,out=run(c,f'flock -n /run/lock/fpga43dgs-gsc1.lock timeout 20 {dest}/probe',timeout=30,log=ev/'run.txt',check=False)
  report['returncode']=rc;report['samples']=[json.loads(l) for l in out.splitlines() if l.startswith('{')]
  report['passed']=rc==0 and len(report['samples'])==18 and sum(x.get('exact',False) for x in report['samples'])==15
  print(out);print('EVIDENCE='+str(ev))
  if not report['passed']:raise RuntimeError('Block diagnostic failed; inspect before retry')
 finally:
  (ev/'result.json').write_text(json.dumps(report,indent=2));c.close()
if __name__=='__main__':main()
