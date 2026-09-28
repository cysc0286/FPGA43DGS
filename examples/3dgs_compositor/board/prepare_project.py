"""Create an isolated vendor project; refuse to overwrite existing work."""
from pathlib import Path
import hashlib
import json
import re
import shutil

EXAMPLE=Path(__file__).resolve().parents[1]
REPO=EXAMPLE.parents[1]
SOURCE=REPO/'platform/fpai_demo_fpga'
TARGET=REPO/'platform/gs_compositor_fpga'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    if TARGET.exists():raise SystemExit('Target exists; inspect before replacing: '+str(TARGET))
    relative=Path('rtl/adder_op/adder_top.v')
    frozen={str(p):sha(SOURCE/p) for p in [relative,Path('fpai_demo_vivado.xpr')]}
    shutil.copytree(SOURCE,TARGET,ignore=shutil.ignore_patterns('.Xil','*.log','*.jou'))
    xpr=TARGET/'fpai_demo_vivado.xpr'
    xpr.write_text(xpr.read_text().replace(SOURCE.as_posix(),TARGET.as_posix()),newline='\n')
    top=TARGET/relative;text=top.read_text();marker='  reg_ctrl   U_reg_ctrl('
    assert text.count(marker)==1
    wires='''    wire [31:0] gs_alpha,gs_r,gs_g,gs_b,gs_ordinal,gs_opcode,gs_request;
    wire [31:0] gs_out_r,gs_out_g,gs_out_b,gs_t,gs_last,gs_done,gs_status;
    wire [31:0] gs_cycles,gs_pixel_cycles;
    gs_compositor U_gs_compositor (
        .clk(ra_clk),.rst_n(ra_rst_n_1),
        .alpha(gs_alpha),.red(gs_r),.green(gs_g),.blue(gs_b),
        .ordinal(gs_ordinal),.opcode(gs_opcode),.request_seq(gs_request),
        .out_r(gs_out_r),.out_g(gs_out_g),.out_b(gs_out_b),.out_t(gs_t),
        .last(gs_last),.completed_seq(gs_done),.status(gs_status),
        .last_cycles(gs_cycles),.pixel_cycles(gs_pixel_cycles)
    );

'''
    text=text.replace(marker,wires+marker)
    ports={'rfu_wreg0':'gs_alpha','rfu_wreg1':'gs_r','rfu_wreg2':'gs_g',
      'rfu_wreg5':'gs_b','rfu_wreg6':'gs_ordinal','rfu_wreg7':'gs_opcode','rfu_wreg8':'gs_request',
      'rfu_rreg0':'gs_out_r','rfu_rreg1':'gs_out_g','rfu_rreg2':'gs_out_b',
      'rfu_rreg3':'gs_t','rfu_rreg4':'gs_done','rfu_rreg5':'gs_status',
      'rfu_rreg6':"32'h47534331",'rfu_rreg7':'gs_last','rfu_rreg8':'gs_cycles','rfu_rreg9':'gs_pixel_cycles'}
    for port,signal in ports.items():
        text,count=re.subn(r'(\.'+port+r'\s*\()([^)]*)(\))',lambda m:m[1]+' '+signal+' '+m[3],text)
        assert count==1,port
    top.write_text(text,newline='\n')
    rtl=EXAMPLE/'rtl/gs_compositor.v'
    shutil.copy2(rtl,TARGET/'rtl/adder_op/gs_compositor.v')
    assert frozen=={p:sha(SOURCE/p) for p in frozen}
    manifest={'source':str(SOURCE),'target':str(TARGET),'baseline_sha256':frozen,
      'rtl_sha256':sha(rtl),'patched_top_sha256':sha(top),'capability':'0x47534331'}
    (TARGET/'gs_compositor_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
