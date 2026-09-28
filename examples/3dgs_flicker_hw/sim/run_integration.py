import datetime,json,os,subprocess,argparse,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--real',action='store_true');args=parser.parse_args()
 tb='tb_real_flicker' if args.real else 'tb_flicker'
 dest=ROOT/'build/hls_renderer/solution1/sim/verilog'
 ev=ROOT/'evidence'/('integration_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S'));ev.mkdir(parents=True)
 bin=Path(r'D:\Xilinx\Vivado\2018.3\bin');env=dict(os.environ,RDI_PLATFORM='win64',PROCESSOR_ARCHITECTURE='AMD64')
 files=[*sorted((ROOT/'rtl').glob('*.sv')),ROOT/'sim'/(tb+'.sv')]
 (ev/'sources').mkdir()
 for f in files:shutil.copy2(f,ev/'sources'/f.name)
 hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
 commands=[('xvlog.bat',['--sv','--work','xil_defaultlib',*map(str,files)]),
  ('xelab.bat',['xil_defaultlib.'+tb,'glbl','-L','xil_defaultlib','-L','unisims_ver','-L','xpm','--initfile',str(bin.parent/'data/xsim/ip/xsim_ip.ini'),'-s','integration']),
  ('xsim.bat',['integration','-runall','-log',str(ev/'simulation.txt')])]
 ok=False
 try:
  for exe,args in commands:
   r=subprocess.run([str(bin/exe),*args],cwd=dest,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=600)
   (ev/(exe+'.txt')).write_text(r.stdout)
   if r.returncode:raise RuntimeError(r.stdout[-3000:])
  log=(ev/'simulation.txt').read_text()
  print('\n'.join(l for l in log.splitlines() if 'PASS:' in l or 'FAIL:' in l))
  ok=('PASS: full HLS' in log or 'PASS: real HLS' in log) and 'FAIL:' not in log
  if not ok:raise RuntimeError('Integration failed')
 finally:
  (ev/'result.json').write_text(json.dumps({'passed':ok,'files':list(map(str,files)),'source_hashes':hashes},indent=2));print('EVIDENCE='+str(ev))
if __name__=='__main__':main()
