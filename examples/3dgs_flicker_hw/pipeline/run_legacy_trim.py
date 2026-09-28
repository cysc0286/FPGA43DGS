"""Verify removal of unused historical cores preserves the actual vendor path."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, required=True)
    args = parser.parse_args()
    ev = ROOT/'evidence'/('legacy_trim_sim_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
    ev.mkdir()
    vendor = REPO/'platform/flicker_cat_compact_fpga/rtl/adder_op'
    sources = [args.candidate/'legacy_adder_top.v', args.candidate/'legacy_reference.v',
               ROOT/'pipeline/tb_legacy_trim.sv',
               *[vendor/name for name in ['reg_ctrl.v','pulse_cross.v','dma.v','adder.v',
                                          'gs_compositor.v','gs_batch_regs.sv','gs_batch_executor.sv']],
               *vendor.glob('AVR/*.v')]
    for path in sources:
        shutil.copy2(path, ev/path.name)
    record = {'candidate':str(args.candidate.resolve()), 'passed':False,
              'sha256':{path.name:hashlib.sha256((ev/path.name).read_bytes()).hexdigest() for path in sources}}
    commands = [['xvlog.bat','-sv',*[str(ev/path.name) for path in sources]],
                ['xelab.bat','tb_legacy_trim','-s','legacy_trim','-debug','typical','-mt','2','--timescale','1ns/1ps'],
                ['xsim.bat','legacy_trim','-runall']]
    env = dict(os.environ, RDI_PLATFORM='win64', PROCESSOR_ARCHITECTURE='AMD64')
    try:
        for name, *parameters in commands:
            result = subprocess.run([str(Path('D:/Xilinx/Vivado/2018.3/bin')/name),*parameters],
                                    cwd=ev,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            (ev/(name+'.log')).write_text(result.stdout)
            print(result.stdout[-2500:],flush=True)
            if result.returncode:
                raise RuntimeError(name+' failed')
        log = (ev/'xsim.bat.log').read_text()
        record['passed'] = 'PASS: removed capabilities absent;' in log and 'Fatal:' not in log
        if not record['passed']:
            raise RuntimeError('Missing simulator pass marker')
    finally:
        (ev/'result.json').write_text(json.dumps(record,indent=2))
        print('EVIDENCE='+str(ev))


if __name__ == '__main__':
    main()
