"""Hash-guarded FLK0 -> FLK1 boot replacement. No automatic reboot."""
import argparse,datetime,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'examples/3dgs_compositor/board'))
from remote import connect,run
OLD='3838932193e5dfa60e6c714ead0eaa036a1d25f202986bc79e30736fcbcea5b0'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--boot-dir',type=Path,required=True);p.add_argument('--timing-review',type=Path);p.add_argument('--integration',type=Path,required=True);a=p.parse_args()
 integration=json.loads((a.integration/'result.json').read_text())
 if integration.get('passed') is not True:raise ValueError('Real-input DMA/CTU integration has not passed')
 dma=ROOT/('pipeline/rtl/flicker_dma.sv' if (ROOT/'pipeline/rtl/flicker_dma.sv').exists() else 'rtl/flicker_dma.sv')
 for name in [dma,ROOT/'pipeline/rtl/flicker_regs.sv',ROOT/'pipeline/rtl/flicker_top.sv']:
  digest=integration.get('platform_rtl_sha256',{}).get(name.name,integration['source_hashes'].get(str(name)))
  if digest!=sha(name):raise ValueError('Integration source drift: '+str(name))
 status=(a.boot_dir.parent/'status.txt').read_text();boot=a.boot_dir/'BOOT.bin'
 if 'BITSTREAM_GENERATION=PASS' not in status:raise ValueError('Incomplete build')
 if 'TIMING_ACCEPTANCE=PASS_UNDER_CURRENT_CONSTRAINTS' not in status:
  if not a.timing_review:raise ValueError('Timing not accepted: analyze reports before any install')
  review=json.loads(a.timing_review.read_text())
  if review.get('new_render_paths_passed') is not True or review.get('prototype_use_justified') is not True or review.get('timing_sha256')!=sha(a.boot_dir.parent/'timing_summary.rpt'):raise ValueError('Invalid timing review')
 proof=json.loads((a.boot_dir/'comparison.json').read_text())
 if not all(proof[k] for k in ['passed','non_pl_partitions_exact','pl_payload_exact']) or sha(boot)!=proof['boot_sha256']:raise ValueError('Package proof mismatch')
 platform=Path(next(line.split('=',1)[1] for line in status.splitlines() if line.startswith('PROJECT=')))
 manifest=json.loads((platform/'flicker_manifest.json').read_text())
 if not integration.get('hls_rtl_hashes') or integration['hls_rtl_hashes']!=manifest.get('hls_rtl_hashes'):raise ValueError('HLS integration/build provenance mismatch')
 for name,digest in manifest['sources'].items():
  if sha(platform/name)!=digest:raise ValueError('Source drift: '+name)
 if manifest.get('legacy_trim'):
  trim=manifest['legacy_trim'];verification=Path(trim['evidence'])/'result.json'
  if sha(verification)!=trim['record_sha256']:raise ValueError('Legacy trim evidence drift')
  tested=json.loads(verification.read_text())
  if tested.get('passed') is not True or sha(platform/'rtl/adder_op/legacy_adder_top.v')!=tested['sha256']['legacy_adder_top.v']:raise ValueError('Unverified legacy trim')
 stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S');ev=ROOT/'evidence'/('boot_'+stamp);ev.mkdir(parents=True);dest='/root/fpga43dgs_flicker/boot_'+stamp
 script=(REPO/'examples/3dgs_compositor/board/update_boot.sh').read_text().replace('old=ff350477e624c50d2f8180fb4b9130ec7688fbc7ca553412ed7c3dd2a68b31ef','old='+OLD)
 script=script.replace('/root/fpga43dgs_compositor/boot_*','/root/fpga43dgs_flicker/boot_*').replace('pre_gsc1','pre_flk1').replace('next_gsc1','next_flk1')
 (ev/'update_boot.sh').write_text(script,newline='\n')
 report={'previous_sha256':OLD,'new_sha256':sha(boot),'remote':dest,'installed_verified':False,'rebooted':False,'integration_evidence':str(a.integration),'integration_record_sha256':sha(a.integration/'result.json'),'rollback_command':f'bash {dest}/update_boot.sh rollback {dest} {sha(boot)}'}
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
