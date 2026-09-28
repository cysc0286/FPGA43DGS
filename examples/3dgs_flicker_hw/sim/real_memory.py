"""Derive an RTL DDR test from frozen real Gaussian input and HLS C golden."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 ref=ROOT/'evidence/real_reference';inp=(ref/'input.flk').read_bytes();out=(ref/'expected.raw').read_bytes()
 assert len(inp)%64==0 and len(out)==3*4096
 dest=ROOT/'build/hls_renderer/solution1/sim/verilog'
 for name,data in [('real_input.hex',inp),('real_expected.hex',out)]:
  (dest/name).write_text('\n'.join(data[i:i+64][::-1].hex() for i in range(0,len(data),64))+'\n')
 tb=(ROOT/'sim/tb_flicker.sv').read_text()
 tb=tb[:tb.index(' initial begin #10000000;')]
 tb=tb.replace('module tb_flicker;','module tb_real_flicker;')
 tb+='''
 reg [511:0] expected[0:191];
 initial begin #100000000; $fatal(1,"FAIL: real integration timeout"); end
 initial begin
  for(i=0;i<4096;i=i+1)mem[i]={16{32'hdeadbeef}};
  $readmemh("real_input.hex",mem,64,834);
  $readmemh("real_expected.hex",expected);
  repeat(8)@(negedge gp);rst=1;
  write_reg('h10,'h1000);write_reg('h14,771);write_reg('h18,'h10000);write_reg('h1c,192);
  write_reg('h20,3);write_reg('h24,123);write_reg('h28,0);write_reg('h2c,1);
  s=0;guard=0;
  while(!(s&4)&&guard<1000000)begin read_reg(8,s);guard=guard+1;end
  if(!(s&4)||(s&8))$fatal(1,"FAIL: real job status");
  read_reg('h30,value);if(value!=123)$fatal(1,"FAIL: real job ID");
  read_reg('h48,value);if(value)$fatal(1,"FAIL: real DMA error");
  for(i=0;i<192;i=i+1)if(mem[1024+i]!==expected[i])begin
   $display("FAIL: real output word %d actual=%h expected=%h",i,mem[1024+i],expected[i]);$fatal(1,"FAIL: real golden mismatch");
  end
  if(read_seen!=772||write_seen!=192)$fatal(1,"FAIL: real wire counts");
  if(mem[1216]!=={16{32'hdeadbeef}})$fatal(1,"FAIL: real output overrun");
  read_reg('h40,value);if(value==0)$fatal(1,"FAIL: prefetch did not overlap");
  $display("PASS: real HLS + DMA + CDC 768 Gaussians 768 pixels bit-exact with HLS C, prefetch overlaps=%d",value);
  $finish;
 end
endmodule
'''
 (ROOT/'sim/tb_real_flicker.sv').write_text(tb)
if __name__=='__main__':main()
