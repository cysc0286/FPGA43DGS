"""Hash-guarded GSB1 -> FLK0 boot replacement. No automatic reboot."""
import argparse,datetime,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'examples/3dgs_compositor/board'))
from remote import connect,run
OLD='437f68944bcb060f6722a2715b6a67170790ab108818a72ae4fe2d6fd9f3a35f'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--boot-dir',type=Path,required=True);p.add_argument('--timing-review',type=Path);a=p.parse_args()
 status=(a.boot_dir.parent/'status.txt').read_text();boot=a.boot_dir/'BOOT.bin'
 if 'BITSTREAM_GENERATION=PASS' not in status:raise ValueError('Incomplete build')
 if 'TIMING_ACCEPTANCE=PASS_UNDER_CURRENT_CONSTRAINTS' not in status:
  if not a.timing_review:raise ValueError('Timing not accepted: analyze reports before any install')
  review=json.loads(a.timing_review.read_text())
  if review.get('new_render_paths_passed') is not True or review.get('prototype_use_justified') is not True or review.get('timing_sha256')!=sha(a.boot_dir.parent/'timing_summary.rpt'):raise ValueError('Invalid timing review')
 proof=json.loads((a.boot_dir/'comparison.json').read_text())
 if not all(proof[k] for k in ['passed','non_pl_partitions_exact','pl_payload_exact']) or sha(boot)!=proof['boot_sha256']:raise ValueError('Package proof mismatch')
 manifest=json.loads((REPO/'platform/flicker_fpga/flicker_manifest.json').read_text())
 for name,digest in manifest['sources'].items():
  if sha(REPO/'platform/flicker_fpga'/name)!=digest:raise ValueError('Source drift: '+name)
 stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S');ev=ROOT/'evidence'/('boot_'+stamp);ev.mkdir(parents=True);dest='/root/fpga43dgs_flicker/boot_'+stamp
 script=(REPO/'examples/3dgs_compositor/board/update_boot.sh').read_text().replace('old=ff350477e624c50d2f8180fb4b9130ec7688fbc7ca553412ed7c3dd2a68b31ef','old='+OLD)
 script=script.replace('/root/fpga43dgs_compositor/boot_*','/root/fpga43dgs_flicker/boot_*').replace('pre_gsc1','pre_flk0').replace('next_gsc1','next_flk0')
 (ev/'update_boot.sh').write_text(script,newline='\n')
 report={'previous_sha256':OLD,'new_sha256':sha(boot),'remote':dest,'installed_verified':False,'rebooted':False,'rollback_command':f'bash {dest}/update_boot.sh rollback {dest} {sha(boot)}'}
 c=connect()
 try:
  run(c,'mkdir -p '+dest)
  with c.open_sftp() as s:s.put(str(boot),dest+'/BOOT.new.bin');s.put(str(ev/'update_boot.sh'),dest+'/update_boot.sh')
  run(c,f'bash {dest}/update_boot.sh apply {dest} {sha(boot)}',timeout=90,log=ev/'apply.txt')
  with c.open_sftp() as s:s.get(dest+'/BOOT.previous.bin',str(ev/'BOOT.previous.bin'))
  if sha(ev/'BOOT.previous.bin')!=OLD:raise ValueError('Backup hash: do not reboot')
  report['installed_verified']=True;print('INSTALLED='+str(ev))
 finally:(ev/'result.json').write_text(json.dumps(report,indent=2));c.close()
if __name__=='__main__':main()
