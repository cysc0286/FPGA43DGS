"""Compile the real-frame driver; upload frozen scenes; do not run new RTL yet."""
import datetime,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'examples/3dgs_compositor/board'))
from remote import connect,run
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S');ev=ROOT/'evidence'/('stage_'+stamp);ev.mkdir(parents=True)
 dest='/root/fpga43dgs_flicker/stage_'+stamp
 sources=[ROOT/'board/render.cpp',REPO/'examples/3dgs_flicker_cat/cat_reference.cpp']
 sdk='/root/heterogs_npu/sdk_3.36.1/usr';lib=sdk+'/lib/aarch64-linux-gnu'
 flags=f'-O3 -std=gnu++17 -ffp-contract=off -fopenmp -Wall -Wextra -I{sdk}/include -L{lib} -Wl,-rpath,{lib} -licraft_zg330backend -licraft_hostbackend -licraft_xrt -licraft_xir -licraft_utils -ldw -ldl -pthread'
 report={'remote':dest,'source_hashes':{p.name:sha(p) for p in sources},'flags':flags,'compiled':False,'cases':[]}
 c=connect()
 try:
  run(c,'mkdir -p '+dest)
  with c.open_sftp() as s:
   for p in sources:shutil.copy2(p,ev/p.name);s.put(str(p),dest+'/'+p.name)
  run(c,f'cd {dest} && g++ render.cpp -o render {flags}',timeout=180,log=ev/'compile.txt')
  report['compiled']=True;report['binary_sha256']=run(c,'sha256sum '+dest+'/render')[1].split()[0]
  data=REPO/'examples/3dgs_scene/data/20260926T135717'
  with c.open_sftp() as s:
   for case in ['n10000_v0_w320','n559263_v0_w320','n559263_v10_w320']:
    p=data/case/'scene.bin';s.put(str(p),dest+'/'+case+'.bin')
    if run(c,'sha256sum '+dest+'/'+case+'.bin')[1].split()[0]!=sha(p):raise RuntimeError('Scene upload hash')
    report['cases'].append({'name':case,'sha256':sha(p),'local':str(p)})
  print('STAGED='+str(ev))
 finally:
  (ev/'result.json').write_text(json.dumps(report,indent=2));c.close()
if __name__=='__main__':main()
