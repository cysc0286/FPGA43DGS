"""Synchronize reviewed RTL changes after tests, preserving the previous manifest."""
import argparse,datetime,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
TARGET=REPO/'platform/flicker_fpga'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--reason',required=True);p.add_argument('--tests',nargs='+',type=Path,required=True);a=p.parse_args()
 for test in a.tests:
  result=json.loads((test/'result.json').read_text())
  if not result['passed']:raise ValueError('Test not passed: '+str(test))
  for rel,digest in result['source_hashes'].items():
   if sha(ROOT/rel)!=digest:raise ValueError('Test source drift: '+rel)
 manifest=TARGET/'flicker_manifest.json';old=json.loads(manifest.read_text())
 for name,digest in old['sources'].items():
  if sha(TARGET/name)!=digest:raise ValueError('Unexpected platform source drift: '+name)
 stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
 backup=ROOT/'evidence'/('platform_sync_'+stamp);backup.mkdir(parents=True)
 shutil.copy2(manifest,backup/'previous_manifest.json')
 changed={}
 for source in sorted((ROOT/'rtl').glob('*.sv')):
  target=TARGET/'rtl/flicker'/source.name;name=str(target.relative_to(TARGET))
  if sha(source)!=sha(target):
   shutil.copy2(target,backup/target.name);changed[name]={'old':sha(target),'new':sha(source)}
   shutil.copy2(source,target);old['sources'][name]=sha(target)
 old.setdefault('revisions',[]).append({'stamp':stamp,'reason':a.reason,'changed':changed,'tests':list(map(str,a.tests))})
 manifest.write_text(json.dumps(old,indent=2))
 (backup/'result.json').write_text(json.dumps(old['revisions'][-1],indent=2))
 print('SYNCED='+str(backup))

if __name__=='__main__':main()
