"""Real feature stream through CTU/VRUs, buffered DMA and asynchronous registers."""
import argparse,datetime,hashlib,json,os,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--reference',type=Path,default=ROOT/'evidence/pipeline_real_protocol');parser.add_argument('--project',default='hls_pipeline_workset');parser.add_argument('--dma-source',type=Path);parser.add_argument('--profile-split',action='store_true');args=parser.parse_args()
 if args.dma_source is None:args.dma_source=ROOT/('pipeline/rtl/flicker_dma.sv' if (ROOT/'pipeline/rtl/flicker_dma.sv').exists() else 'rtl/flicker_dma.sv')
 ref=args.reference;raw=(ref/'input.flk').read_bytes();meta=json.loads((ref/'input.json').read_text());records=meta['records']
 if len(meta['tiles'])!=3 or len(raw)!=records*64 or records>959:raise ValueError('Protocol test requires three tiles within guarded simulation memory')
 if hashlib.sha256(raw).hexdigest()!=meta['input_sha256']:raise ValueError('Input drift')
 project=ROOT/'build'/args.project
 dest=project/'solution1/sim/verilog'
 ev=ROOT/'evidence'/('pipeline_integration_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S'));ev.mkdir()
 for name,data in [('cat_input.hex',raw),*[(f'cat_mode{m}.hex',(ref/f'expected_mode{m}.raw').read_bytes()) for m in range(6)]]:
  (dest/name).write_text('\n'.join(data[i:i+64][::-1].hex() for i in range(0,len(data),64))+'\n')
 base=(ROOT/'sim/tb_real_flicker.sv').read_text();base=base[:base.index(' reg [511:0] expected')]
 base=base.replace('module tb_real_flicker;','module tb_pipeline;')
 # Force full output FIFO and require stable source payload while DDR is stalled.
 base=base.replace('awready<=awvalid&&(ticks%5!=0);','awready<=awvalid&&(ticks%23>8);')
 base+='''
 reg [511:0] expected[0:191];integer m,prior_reads,prior_writes;reg[31:0]cyc;
 reg stalled=0;reg[511:0]held_data;reg[31:0]held_address;
 always @(posedge hp)if(rst)begin
  if(stalled&&(!awvalid||awdata!==held_data||awaddr!==held_address))$fatal(1,"FAIL: stalled output changed");
  stalled<=awvalid&&!awready;held_data<=awdata;held_address<=awaddr;
 end
 initial begin #200000000;$fatal(1,"FAIL: CTU integration timeout");end
 initial begin
  for(i=0;i<4096;i=i+1)mem[i]={16{32'hdeadbeef}};
  $readmemh("cat_input.hex",mem,64,INPUT_LAST);
  repeat(8)@(negedge gp);rst=1;
  read_reg(0,value);if(value!=32'h464c4b31)$fatal(1,"FAIL: FLK1 cap");
  for(m=0;m<6;m=m+1)begin
   case(m)
    0:$readmemh("cat_mode0.hex",expected);1:$readmemh("cat_mode1.hex",expected);
    2:$readmemh("cat_mode2.hex",expected);3:$readmemh("cat_mode3.hex",expected);
    4:$readmemh("cat_mode4.hex",expected);5:$readmemh("cat_mode5.hex",expected);
   endcase
   prior_reads=read_seen;prior_writes=write_seen;
   for(i=1023;i<=1216;i=i+1)mem[i]={16{32'hdeadbeef}};
   write_reg('h10,'h1000);write_reg('h14,INPUT_RECORDS);write_reg('h18,'h10000);write_reg('h1c,192);
   write_reg('h20,3);write_reg('h24,m+1);write_reg('h28,m);write_reg('h2c,1);
   s=0;guard=0;
   while(!(s&4)&&guard<2000000)begin read_reg(8,s);guard=guard+1;end
   if(!(s&4)||(s&8))$fatal(1,"FAIL: job status");
   read_reg('h30,value);if(value!=m+1)$fatal(1,"FAIL: job ID");
   read_reg('h48,value);if(value)$fatal(1,"FAIL: DMA error");
   for(i=0;i<192;i=i+1)if(mem[1024+i]!==expected[i])begin
    $display("FAIL: mode=%d word=%d actual=%h expected=%h",m,i,mem[1024+i],expected[i]);$fatal(1,"FAIL: golden mismatch");
   end
   if(read_seen-prior_reads!=FENCED_READS||write_seen-prior_writes!=192)$fatal(1,"FAIL: wire accounting");
   if(mem[1023]!=={16{32'hdeadbeef}}||mem[1216]!=={16{32'hdeadbeef}})$fatal(1,"FAIL: guard");
   read_reg('h34,cyc);$display("MODE_PASS mode=%d cycles=%d input=INPUT_RECORDS output=192",m,cyc);
   repeat(12)@(negedge gp);read_reg(8,s);if(!(s&4))$fatal(1,"FAIL: completion ownership");
   write_reg('h2c,2);s=1;while(s!=0)read_reg(8,s);
  end
  write_reg('h28,6);write_reg('h2c,1);read_reg(8,s);if(!(s&8))$fatal(1,"FAIL: invalid mode accepted");
  $display("PASS: CTU+VRU+DMA+CDC six real modes 4608 pixels, stalls, guards and ownership");$finish;
 end
endmodule
'''
 base=base.replace('INPUT_LAST',str(63+records)).replace('INPUT_RECORDS',str(records)).replace('FENCED_READS',str(records+1))
 if args.profile_split:
  # These simulation-only probes count actual HLS blocking conditions, rather
  # than treating an empty FIFO while idle as a consumer stall. No RTL edits.
  probes=['dut.render.split_U0.prepare_geometry_U0','dut.render.split_U0.expand_subtiles_U0'] if (project/'solution1/syn/verilog/prepare_geometry.v').exists() else ['dut.render.split_U0']
  declarations=''
  displays=''
  for idx,probe in enumerate(probes):
   declarations+=f'''integer split_in{idx}=0,split_out{idx}=0,split_read{idx}=0,split_write{idx}=0;
 always @(posedge hp)begin
  if(!rst||dut.start)begin split_in{idx}<=0;split_out{idx}<=0;split_read{idx}<=0;split_write{idx}<=0;end
  else if(dut.busy)begin
   if(!{probe}.input_V_V_blk_n)split_in{idx}<=split_in{idx}+1;
   if(!{probe}.output_V_V_blk_n)split_out{idx}<=split_out{idx}+1;
   if({probe}.input_V_V_read&&{probe}.input_V_V_empty_n)split_read{idx}<=split_read{idx}+1;
   if({probe}.output_V_V_write&&{probe}.output_V_V_full_n)split_write{idx}<=split_write{idx}+1;
  end
 end
'''
   displays+=f'$display("SPLIT_PROFILE mode=%d stage={probe} input_block=%d output_block=%d reads=%d writes=%d",m,split_in{idx},split_out{idx},split_read{idx},split_write{idx});\n'
  base=base.replace(' reg [511:0] expected[0:191];',declarations+' reg [511:0] expected[0:191];')
  base=base.replace('repeat(12)@(negedge gp);read_reg(8,s);',displays+'repeat(12)@(negedge gp);read_reg(8,s);')
 tb=ev/'tb_pipeline.sv';tb.write_text(base)
 files=[args.dma_source.resolve(),ROOT/'pipeline/rtl/flicker_regs.sv',ROOT/'pipeline/rtl/flicker_top.sv',tb]
 (ev/'sources').mkdir()
 for f in files:shutil.copy2(f,ev/'sources'/f.name)
 bin=Path('D:/Xilinx/Vivado/2018.3/bin');env=dict(os.environ,RDI_PLATFORM='win64',PROCESSOR_ARCHITECTURE='AMD64')
 commands=[('xvlog.bat',['--sv','--work','xil_defaultlib',*map(str,files)]),('xelab.bat',['xil_defaultlib.tb_pipeline','glbl','-L','xil_defaultlib','-L','unisims_ver','-L','xpm','--initfile',str(bin.parent/'data/xsim/ip/xsim_ip.ini'),'-s','integration_cat']),('xsim.bat',['integration_cat','-runall','-log',str(ev/'simulation.txt')])]
 report={'passed':False,'reference':str(ref),'hls_project':str(project),'hls_rtl_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (project/'solution1/impl/verilog').iterdir() if p.is_file()},'input_records':records,'input_sha256':hashlib.sha256(raw).hexdigest(),'source_hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'scope':'Bounded real-input RTL protocol integration, not full-scene performance; full-scene correctness and timing require physical board tests'}
 report['platform_rtl_sha256']={name:hashlib.sha256(f.read_bytes()).hexdigest() for name,f in zip(['flicker_dma.sv','flicker_regs.sv','flicker_top.sv'],files[:3])}
 try:
  for exe,args in commands:
   with (ev/(exe+'.txt')).open('w') as log:r=subprocess.run([str(bin/exe),*args],cwd=dest,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
   if r.returncode:raise RuntimeError('Simulation command failed: '+exe)
  log=(ev/'simulation.txt').read_text();report['passed']='PASS: CTU+VRU' in log and 'FAIL:' not in log
  report['split_profile']=[l for l in log.splitlines() if 'SPLIT_PROFILE' in l]
  print('\n'.join(l for l in log.splitlines() if 'PASS' in l or 'FAIL:' in l or 'SPLIT_PROFILE' in l))
  if not report['passed']:raise RuntimeError('Pipeline integration failed')
 finally:(ev/'result.json').write_text(json.dumps(report,indent=2));print('EVIDENCE='+str(ev))
if __name__=='__main__':main()
