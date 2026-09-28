"""Back up and install a verified compositor BOOT via SSH; never reboot here."""
import argparse
import hashlib
import json
import shlex
import sys
import time
from pathlib import Path
EXAMPLE=Path(__file__).resolve().parents[1];REPO=EXAMPLE.parents[1]
from remote import connect,run

def main():
    p=argparse.ArgumentParser();p.add_argument('--boot-dir',type=Path,required=True)
    p.add_argument('--allow-reviewed-timing-failure',action='store_true');a=p.parse_args()
    boot=a.boot_dir/'BOOT.bin';comparison=json.loads((a.boot_dir/'comparison.json').read_text())
    status=(a.boot_dir.parent/'status.txt').read_text()
    if 'BITSTREAM_GENERATION=PASS' not in status:raise RuntimeError('No completed bitstream')
    if 'TIMING_ACCEPTANCE=PASS_UNDER_CURRENT_CONSTRAINTS' not in status and not a.allow_reviewed_timing_failure:
        raise RuntimeError('Review whole-design AND compositor timing before prototype installation')
    digest=hashlib.sha256(boot.read_bytes()).hexdigest()
    if digest!=comparison['boot_sha256'] or not all(comparison[k] for k in ['passed','non_pl_partitions_exact','pl_payload_exact']):
        raise RuntimeError('Packaged image validation mismatch')
    stamp=time.strftime('%Y%m%d_%H%M%S');local=EXAMPLE/'evidence'/('boot_'+stamp);local.mkdir(parents=True)
    remote='/root/fpga43dgs_compositor/boot_'+stamp
    c=connect()
    try:
        run(c,'cat /proc/sys/kernel/random/boot_id; findmnt /; df -h /root',log=local/'before.txt')
        run(c,'mkdir -p '+shlex.quote(remote))
        with c.open_sftp() as s:
            s.put(str(boot),remote+'/BOOT.new.bin')
            s.put(str(EXAMPLE/'board/update_boot.sh'),remote+'/update_boot.sh')
        run(c,'bash '+shlex.quote(remote+'/update_boot.sh')+' apply '+shlex.quote(remote)+' '+digest,
            timeout=90,log=local/'apply.txt')
        with c.open_sftp() as s:s.get(remote+'/BOOT.previous.bin',str(local/'BOOT.previous.bin'))
        if hashlib.sha256((local/'BOOT.previous.bin').read_bytes()).hexdigest()!=comparison['known_boot_sha256']:
            raise RuntimeError('Downloaded backup failed hash check; do not reboot')
        summary={'installed_sha256':digest,'previous_sha256':comparison['known_boot_sha256'],
                 'remote':remote,'boot_package':str(a.boot_dir.resolve()),'rebooted':False,
                 'rollback_command':'bash '+remote+'/update_boot.sh rollback '+remote+' '+digest,
                 'hardware_validation':'PENDING','reviewed_timing_failure':a.allow_reviewed_timing_failure}
        (local/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2));print('EVIDENCE='+str(local))
    finally:c.close()

if __name__=='__main__':main()
