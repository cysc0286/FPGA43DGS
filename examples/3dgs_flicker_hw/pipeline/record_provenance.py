"""Read and bind the board BOOT hash/boot ID to completed measurements."""
import argparse, datetime, hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'examples/3dgs_compositor/board'))
from remote import connect,run
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('evidence',type=Path,nargs='+');p.add_argument('--physical-report',type=Path,required=True);p.add_argument('--boot-evidence',type=Path,required=True);p.add_argument('--variant',required=True);p.add_argument('--stage',type=Path,help='Explicit renderer stage for full-pipeline evidence');a=p.parse_args()
 boot=a.physical_report/'boot/BOOT.bin';expected=sha(boot)
 status=(a.physical_report/'status.txt').read_text();platform=Path(next(x.split('=',1)[1] for x in status.splitlines() if x.startswith('PROJECT=')))
 c=connect()
 try:
  command="""set -eu
if findmnt -rn -S /dev/mmcblk0p1 >/dev/null; then echo 'Boot partition unexpectedly mounted'; exit 1; fi
[ "$(blkid -s TYPE -o value /dev/mmcblk0p1)" = vfat ]
dir=$(mktemp -d /tmp/flk-provenance.XXXXXX)
trap 'umount "$dir"; rmdir "$dir"' EXIT
mount -t vfat -o ro /dev/mmcblk0p1 "$dir"
sha256sum "$dir/BOOT.bin"
cat /proc/sys/kernel/random/boot_id
"""
  text=run(c,command,timeout=20)[1];lines=text.strip().splitlines()
  if lines[0].split()[0]!=expected:raise ValueError('Installed BOOT differs from physical report')
  bootid=lines[1].strip()
  for folder in a.evidence:
   result=json.loads((folder/'result.json').read_text())
   if not result['complete'] or bootid not in (folder/'environment.txt').read_text():raise ValueError('Incomplete run or different running boot ID')
   stage=(a.stage if a.stage else Path(result['stage']))/'result.json'
   record={'recorded_local':datetime.datetime.now().isoformat(),'physical_report':str(a.physical_report.resolve()),'boot_evidence':str(a.boot_evidence.resolve()),'boot_sha256':expected,'boot_id':bootid,'fpga_compute_clock_mhz':200,'board_cpu_frequency':'not available: cpufreq sysfs absent','fpga_variant':a.variant,'manifest_sha256':sha(platform/'flicker_manifest.json'),'stage_result_sha256':sha(stage),'timing_review_sha256':sha(a.physical_report/'timing_review.json'),'readback_log':text}
   dest=folder/'physical_provenance.json'
   if dest.exists():raise ValueError('Provenance exists; preserve it')
   dest.write_text(json.dumps(record,indent=2));print('RECORDED '+str(folder))
 finally:c.close()
if __name__=='__main__':main()
