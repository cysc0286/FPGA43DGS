"""Create an isolated platform with legacy regressions and 512-bit renderer."""
import hashlib,json,shutil,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
SOURCE=REPO/'platform/gs_batch_fpga';TARGET=REPO/'platform/flicker_fpga'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if TARGET.exists():raise RuntimeError('Platform exists; do not overwrite build/history')
 if not re.search(r'\|\s*Verilog\|\s*Pass\|',(ROOT/'build/hls_renderer/solution1/sim/report/flicker_render_cosim.rpt').read_text()):raise RuntimeError('HLS cosim not passed')
 before=sha(SOURCE/'rtl/adder_op/adder_top.v')
 shutil.copytree(SOURCE,TARGET,ignore=shutil.ignore_patterns('.Xil','*.runs','*.cache','*.sim','*.hw','*.log','*.jou'))
 for path in [TARGET/'fpai_demo_vivado.xpr',TARGET/'call_procise_disable_icap.bat',TARGET/'procise_run_disable_icap.tcl']:
  path.write_text(path.read_text().replace(SOURCE.as_posix(),TARGET.as_posix()),newline='\n')
 top=TARGET/'rtl/adder_op/adder_top.v'
 legacy=top.read_text().replace('module adder_top(','module legacy_adder_top(')
 (top.parent/'legacy_adder_top.v').write_text(legacy,newline='\n')
 top.write_text((ROOT/'rtl/platform_adder_top.v').read_text(),newline='\n')
 user=TARGET/'rtl/flicker';user.mkdir()
 for p in (ROOT/'rtl').glob('*.sv'):shutil.copy2(p,user/p.name)
 hls=ROOT/'build/hls_renderer/solution1/impl/verilog'
 for p in hls.iterdir():
  if p.is_file():shutil.copy2(p,user/p.name)
 for p in user.glob('*.v'):
  t=p.read_text()
  for dat in user.glob('*.dat'):t=t.replace('./'+dat.name,dat.as_posix())
  p.write_text(t,newline='\n')
 assert before==sha(SOURCE/'rtl/adder_op/adder_top.v')
 manifest={'baseline':str(SOURCE),'baseline_top_sha256':before,'sources':{str(p.relative_to(TARGET)):sha(p) for p in [top,top.parent/'legacy_adder_top.v',*user.iterdir()] if p.is_file()}}
 (TARGET/'flicker_manifest.json').write_text(json.dumps(manifest,indent=2))
 print('PREPARED='+str(TARGET))
if __name__=='__main__':main()
