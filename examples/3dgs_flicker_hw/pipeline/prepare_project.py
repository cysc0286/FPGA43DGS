"""Create an isolated platform with legacy regressions and 512-bit renderer."""
import argparse,hashlib,json,shutil,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
SOURCE=REPO/'platform/gs_batch_fpga';TARGET=REPO/'platform/flicker_cat_fpga'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 global TARGET
 parser=argparse.ArgumentParser()
 parser.add_argument('--project',default='hls_pipeline_workset')
 parser.add_argument('--console',type=Path,default=ROOT/'build/pipeline_workset_console.txt')
 parser.add_argument('--platform',default='flicker_cat_fpga')
 parser.add_argument('--legacy-trim-evidence',type=Path)
 args=parser.parse_args()
 if not re.fullmatch(r'[a-zA-Z0-9_]+',args.platform):raise ValueError('Invalid platform name')
 TARGET=REPO/'platform'/args.platform
 project=ROOT/'build'/args.project
 if TARGET.exists():raise RuntimeError('Platform exists; do not overwrite build/history')
 if not re.search(r'\|\s*Verilog\|\s*Pass\|',(project/'solution1/sim/report/flicker_render_pipeline_cosim.rpt').read_text()):raise RuntimeError('HLS cosim not passed')
 if '// ERROR : Due to pragma' in args.console.read_text(errors='replace'):raise RuntimeError('Unresolved dependence diagnostics')
 before=sha(SOURCE/'rtl/adder_op/adder_top.v')
 shutil.copytree(SOURCE,TARGET,ignore=shutil.ignore_patterns('.Xil','*.runs','*.cache','*.sim','*.hw','*.log','*.jou'))
 for path in [TARGET/'fpai_demo_vivado.xpr',TARGET/'call_procise_disable_icap.bat',TARGET/'procise_run_disable_icap.tcl']:
  path.write_text(path.read_text().replace(SOURCE.as_posix(),TARGET.as_posix()),newline='\n')
 top=TARGET/'rtl/adder_op/adder_top.v'
 legacy=top.read_text().replace('module adder_top(','module legacy_adder_top(')
 (top.parent/'legacy_adder_top.v').write_text(legacy,newline='\n')
 trim_record=None
 if args.legacy_trim_evidence:
  trim_record=json.loads((args.legacy_trim_evidence/'result.json').read_text())
  if trim_record.get('passed') is not True:raise ValueError('Legacy trim has not passed vendor-path regression')
  candidate=args.legacy_trim_evidence/'legacy_adder_top.v'
  reference=args.legacy_trim_evidence/'legacy_reference.v'
  for p in [candidate,reference]:
   if sha(p)!=trim_record['sha256'][p.name]:raise ValueError('Legacy verification source drift')
  if reference.read_text().replace('module legacy_reference(','module legacy_adder_top(')!=legacy:raise ValueError('Legacy baseline differs from verified reference')
  shutil.copy2(candidate,top.parent/'legacy_adder_top.v')
 top.write_text((ROOT/'rtl/platform_adder_top.v').read_text(),newline='\n')
 user=TARGET/'rtl/flicker';user.mkdir()
 for p in (ROOT/'rtl').glob('*.sv'):shutil.copy2(p,user/p.name)
 for p in (ROOT/'pipeline/rtl').glob('*.sv'):shutil.copy2(p,user/p.name)
 hls=project/'solution1/impl/verilog'
 for p in hls.iterdir():
  if p.is_file():shutil.copy2(p,user/p.name)
 for p in user.glob('*.v'):
  t=p.read_text()
  for dat in user.glob('*.dat'):t=t.replace('./'+dat.name,dat.as_posix())
  p.write_text(t,newline='\n')
 assert before==sha(SOURCE/'rtl/adder_op/adder_top.v')
 manifest={'baseline':str(SOURCE),'baseline_top_sha256':before,'hls_project':str(project),'hls_rtl_hashes':{p.name:sha(p) for p in hls.iterdir() if p.is_file()},'sources':{str(p.relative_to(TARGET)):sha(p) for p in [top,top.parent/'legacy_adder_top.v',*user.iterdir()] if p.is_file()}}
 if trim_record:manifest['legacy_trim']={'evidence':str(args.legacy_trim_evidence.resolve()),'record_sha256':sha(args.legacy_trim_evidence/'result.json'),'candidate_sha256':trim_record['sha256']['legacy_adder_top.v'],'removed_capabilities':['GSC1','GSB1'],'vendor_adder_retained':True}
 (TARGET/'flicker_manifest.json').write_text(json.dumps(manifest,indent=2))
 print('PREPARED='+str(TARGET))
if __name__=='__main__':main()
