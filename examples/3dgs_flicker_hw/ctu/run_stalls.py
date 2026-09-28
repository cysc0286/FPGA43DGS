"""Actual CTU RTL: sustained output stalls, input gaps, empty/repeated calls."""
import datetime,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
 ref=ROOT/'evidence/ctu_real';raw=(ref/'input.flk').read_bytes();n=len(raw)//64
 expected=b''.join((ref/f'expected_mode{m}.raw').read_bytes() for m in range(5))
 if len(raw)%64 or len(expected)!=len(raw)*5:raise ValueError('Incomplete CTU references')
 work=ROOT/'build/hls_ctu/solution1/sim/verilog'
 ev=ROOT/'evidence'/('ctu_stalls_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S'));ev.mkdir()
 for name,data in [('ctu_inputs.hex',raw),('ctu_expected.hex',expected)]:
  (work/name).write_text('\n'.join(data[i:i+64][::-1].hex() for i in range(0,len(data),64))+'\n')
 tb='''`timescale 1ns/1ps
module tb_ctu_stalls;
 localparam N=__N__;
 reg clk=0,rst=1,start=0,active=0;always #2.5 clk=~clk;
 reg [31:0] count=0,mode=0;wire done,kr,kw;wire[511:0] result;
 reg[511:0] inputs[0:N-1],expected[0:N*5-1];
 integer ticks=0,ip=0,op=0,begin_tick=0,paused_ip=0,proved_stalls=0,total_out=0;
 wire input_valid=active&&ip<count&&(ticks%7!=0);
 wire output_ready=active&&(count==0||ticks-begin_tick>=5000)&&(ticks%13>3);
 flicker_ctu dut(.ap_clk(clk),.ap_rst(rst),.ap_start(start),.ap_done(done),.ap_ready(),.ap_idle(),
  .input_V_V_dout(inputs[ip<N?ip:0]),.input_V_V_empty_n(input_valid),.input_V_V_read(kr),
  .output_V_V_din(result),.output_V_V_full_n(output_ready),.output_V_V_write(kw),.count(count),.mode(mode));
 always @(posedge clk)begin
  ticks<=ticks+1;
  if(kr&&input_valid)ip<=ip+1;
  if(kw&&output_ready)begin
   if(op>=count||result!==expected[mode*N+op])$fatal(1,"FAIL: CTU mask/order/attribute mode=%d output=%d",mode,op);
   op<=op+1;total_out<=total_out+1;
  end
  if(active&&count!=0&&ticks-begin_tick==4000)paused_ip<=ip;
  if(active&&count!=0&&ticks-begin_tick==5000)begin
   if(ip==0||ip>=count||ip!=paused_ip)$fatal(1,"FAIL: CTU did not backpressure upstream during long stall");
   proved_stalls<=proved_stalls+1;
  end
 end
 task invoke(input integer amount,setting);
  begin
   @(negedge clk);count=amount;mode=setting;ip=0;op=0;begin_tick=ticks;active=1;start=1;
   @(negedge clk);start=0;
   repeat(3)@(negedge clk);
   while(!done)@(negedge clk);
   if(ip!=amount||op!=amount)$fatal(1,"FAIL: CTU completion counts");
   active=0;repeat(8)@(negedge clk);
  end
 endtask
 integer m;
 initial begin #2000000;$fatal(1,"FAIL: CTU stalled timeout");end
 initial begin
  $readmemh("ctu_inputs.hex",inputs);$readmemh("ctu_expected.hex",expected);
  repeat(10)@(negedge clk);rst=0;repeat(4)@(negedge clk);
  for(m=0;m<5;m=m+1)begin invoke(0,m);invoke(N,m);end
  invoke(0,0);
  if(proved_stalls!=5||total_out!=N*5)$fatal(1,"FAIL: CTU coverage");
  $display("PASS: CTU %d real outputs, five sustained backpressure events, input gaps and six empty invocations",total_out);
  $finish;
 end
endmodule
'''.replace('__N__',str(n))
 source=ev/'tb_ctu_stalls.sv';source.write_text(tb)
 bin=Path('D:/Xilinx/Vivado/2018.3/bin');env=dict(os.environ,RDI_PLATFORM='win64',PROCESSOR_ARCHITECTURE='AMD64')
 commands=[('xvlog.bat',['--sv','--work','xil_defaultlib',str(source)]),
  ('xelab.bat',['xil_defaultlib.tb_ctu_stalls','glbl','-L','xil_defaultlib','-L','unisims_ver','-L','xpm','--initfile',str(bin.parent/'data/xsim/ip/xsim_ip.ini'),'-s','ctu_stalls']),
  ('xsim.bat',['ctu_stalls','-runall','-log',str(ev/'simulation.txt')])]
 report={'passed':False,'scope':'standalone CTU RTL, not frame or physical FPGA','inputs':n,'modes':5,
         'input_sha256':hashlib.sha256(raw).hexdigest(),'expected_sha256':hashlib.sha256(expected).hexdigest(),
         'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
         'ctu_cpp_sha256':hashlib.sha256((ROOT/'ctu/ctu.cpp').read_bytes()).hexdigest()}
 try:
  for exe,args in commands:
   run=subprocess.run([str(bin/exe),*args],cwd=work,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=600)
   (ev/(exe+'.txt')).write_text(run.stdout)
   if run.returncode:raise RuntimeError(run.stdout[-2000:])
  log=(ev/'simulation.txt').read_text();report['passed']='PASS: CTU' in log and 'FAIL:' not in log
  print('\n'.join(x for x in log.splitlines() if 'PASS:' in x or 'FAIL:' in x))
  if not report['passed']:raise RuntimeError('CTU backpressure test failed')
 finally:(ev/'result.json').write_text(json.dumps(report,indent=2));print('EVIDENCE='+str(ev))

if __name__=='__main__':main()
