`timescale 1ns/1ps
module tb_flicker;
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
 reg sink_started=0,holding_output=0;
 reg [31:0] held_address;reg [511:0] held_data;
 integer sink_wait=0,output_full_cycles=0;
 integer write_seen=0,read_seen=0,overlaps=0;
 flicker_top dut(.gp_clk(gp),.gp_rst_n(rst),.hp_clk(hp),.hp_rst_n(rst),
  .wa(wa),.ra(ra),.wd(wd),.wvalid(wvalid),.rvalid(rvalid),.wready(wready),.rready(rready),.rd(rd),.rdvalid(rdvalid),.rdready(rdready),
  .araddr(araddr),.arvalid(arvalid),.arready(arready),.rdata(rdata),.rvalid_mem(rv),.rready_mem(rready_mem),
  .awaddr(awaddr),.awdata(awdata),.awvalid(awvalid),.awready(awready),.owned(owned));
 always @(posedge hp)begin
  ticks<=ticks+1;
  if(awvalid&&!sink_started)begin sink_started<=1;sink_wait<=100;end
  else if(sink_wait>0)sink_wait<=sink_wait-1;
  awready<=awvalid&&sink_started&&(sink_wait==0)&&(ticks%5!=0);
  if(dut.dma.output_level==4)output_full_cycles<=output_full_cycles+1;
  if(holding_output&&(!awvalid||awaddr!==held_address||awdata!==held_data))$fatal(1,"FAIL: output changed while stalled");
  holding_output<=awvalid&&!awready;held_address<=awaddr;held_data<=awdata;
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
 initial begin #10000000;$fatal(1,"FAIL: integration timeout");end
 initial begin
  for(i=0;i<4096;i=i+1)mem[i]={16{32'hdeadbeef}};
  header=0;header[31:0]=1;header[68:64]=16;header[73:69]=16;
  q=0;q[47:32]=16'h3c00;q[79:64]=16'h3c00;q[95:80]=16'h3800;
  q[111:96]=16'h3400;q[127:112]=16'h3800;q[143:128]=16'h3a00;
  mem['h1000>>6]=header;mem['h1040>>6]=q;
  repeat(8)@(negedge gp);rst=1;
  read_reg(0,value);if(value!==32'h464c4b30)$fatal(1,"FAIL: capability");
  submit('h1000,'h4000,1);
  // CPU prepares a separate input bank while the first job is active.
  repeat(40)@(negedge gp);
  if(!owned)$fatal(1,"FAIL: no active job during upload");
  q[111:96]=16'h3800;mem['h8000>>6]=header;mem['h8040>>6]=q;
  submit('h8000,'hc000,2);
  for(job=1;job<=2;job=job+1)begin
   s=0;guard=0;
   while(!(s&4)&&guard<10000)begin read_reg(8,s);guard=guard+1;end
   if(!(s&4)||(s&8))$fatal(1,"FAIL: completion/status %h",s);
   read_reg('h30,id);if(id!=job)$fatal(1,"FAIL: job order");
   read_reg('h38,value);if(value!=2)$fatal(1,"FAIL: exact read count %d",value);
   read_reg('h3c,value);if(value!=64)$fatal(1,"FAIL: exact write count");
   read_reg('h48,value);if(value!=0)$fatal(1,"FAIL: hardware protocol");
   for(i=0;i<64;i=i+1)begin
    if(mem[((job==1?'h4000:'hc000)>>6)+i][127:96]!==i*4)$fatal(1,"FAIL: pixel ordering");
   end
   if(mem[(job==1?'h4000:'hc000)>>6][15:0]!==(job==1?16'h3000:16'h3400))$fatal(1,"FAIL: result or bank overwrite");
   if(mem[((job==1?'h4000:'hc000)>>6)+64]!=={16{32'hdeadbeef}})$fatal(1,"FAIL: extra DMA output");
   if(job==1)begin
    repeat(100)@(negedge gp);
    if(mem['hc000>>6]!=={16{32'hdeadbeef}})$fatal(1,"FAIL: queued job ran before completion ACK");
   end
   write_reg('h2c,2);
  end
  if(write_seen!=128||read_seen!=6)$fatal(1,"FAIL: physical DMA count reads=%d writes=%d",read_seen,write_seen);
  if(output_full_cycles==0)$fatal(1,"FAIL: full output FIFO backpressure not exercised");
  repeat(20)@(negedge gp);
  write_reg('h10,'h1001);write_reg('h2c,1);read_reg(8,s);
  if(!(s&8)||owned)$fatal(1,"FAIL: unaligned descriptor not rejected");
  write_reg('h2c,4);read_reg(8,s);if(s&8)$fatal(1,"FAIL: reject clear");
  if(write_seen!=128||read_seen!=6)$fatal(1,"FAIL: rejected request touched DMA");
  $display("PASS: full HLS + DMA + CDC, two queued jobs, 512 pixels, random DDR stalls, exact counts, fence and guards");
  $finish;
 end
endmodule
