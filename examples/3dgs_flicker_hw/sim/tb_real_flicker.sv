`timescale 1ns/1ps
module tb_real_flicker;
 reg gp=0,hp=0,rst=0;always #5 gp=~gp;always #2.5 hp=~hp;
 reg [7:0] wa=0,ra=0;reg[31:0]wd=0;reg wvalid=0,rvalid=0;
 wire wready,rready;wire[31:0]rd;wire rdvalid;reg rdready=1;
 wire[31:0]araddr,awaddr;wire arvalid,awvalid,rready_mem,owned;
 wire[511:0]awdata;reg[511:0]rdata=0;reg rv=0;
 reg[511:0]mem[0:4095];integer ticks=0,delay=0;reg pending=0;reg[31:0]read_addr;
 wire arready=!pending&&!rv&&(ticks%3!=0);
 // A downstream arbiter may wait for VALID before asserting READY. The source
 // must offer a buffered valid independently, otherwise this legal sink locks.
 reg awready=0;
 integer write_seen=0,read_seen=0,overlaps=0;
 flicker_top dut(.gp_clk(gp),.gp_rst_n(rst),.hp_clk(hp),.hp_rst_n(rst),
  .wa(wa),.ra(ra),.wd(wd),.wvalid(wvalid),.rvalid(rvalid),.wready(wready),.rready(rready),.rd(rd),.rdvalid(rdvalid),.rdready(rdready),
  .araddr(araddr),.arvalid(arvalid),.arready(arready),.rdata(rdata),.rvalid_mem(rv),.rready_mem(rready_mem),
  .awaddr(awaddr),.awdata(awdata),.awvalid(awvalid),.awready(awready),.owned(owned));
 always @(posedge hp)begin
  ticks<=ticks+1;
  awready<=awvalid&&(ticks%5!=0);
  if(rv&&rready_mem)rv<=0;
  if(arvalid&&arready)begin pending<=1;read_addr<=araddr;delay<=2+ticks%7;read_seen<=read_seen+1;end
  if(pending)begin
   if(delay==0)begin rv<=1;rdata<=mem[read_addr>>6];pending<=0;end
   else delay<=delay-1;
  end
  if(awvalid&&awready)begin
   if(awaddr[5:0]!=0||awaddr>=262144)$fatal(1,"FAIL: write range");
   mem[awaddr>>6]<=awdata;write_seen<=write_seen+1;
  end
 end
 task write_reg(input[7:0]a,input[31:0]v);
  begin @(negedge gp);wa=a;wd=v;wvalid=1;@(posedge gp);while(!wready)@(posedge gp);@(negedge gp);wvalid=0;end
 endtask
 task read_reg(input[7:0]a,output[31:0]v);
  begin @(negedge gp);ra=a;rvalid=1;@(posedge gp);while(!rready)@(posedge gp);@(negedge gp);rvalid=0;
   while(!rdvalid)@(negedge gp);v=rd;@(negedge gp);end
 endtask
 task submit(input[31:0]ip,op,id);
  begin
   write_reg('h10,ip);write_reg('h14,2);write_reg('h18,op);write_reg('h1c,64);
   write_reg('h20,1);write_reg('h24,id);write_reg('h28,0);write_reg('h2c,1);
  end
 endtask
 reg[31:0]s,id,value;integer i,job,guard;reg[511:0]header,q;

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
