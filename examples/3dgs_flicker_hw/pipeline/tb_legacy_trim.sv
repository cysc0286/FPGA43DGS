`timescale 1ns/1ps
// Differential vendor-path regression; both wrappers receive identical requests.
module tb_legacy_trim;
 reg clk=0,ra_clk=0,rst_n=0,ra_rst_n=0;
 always #4 clk=~clk;
 always #5 ra_clk=~ra_clk;
 reg [31:0] wa=0,wd=0,ra=0;
 reg wvalid=0,rvalid=0,rdready=0;
 wire [1:0] wready,rready,rdvalid,arvalid,awvalid,memready,reset_reg;
 wire [31:0] rd[2],araddr[2],awaddr[2];
 wire [63:0] awdata[2];
 integer cycles=0,head=0,tail=0,writes=0,reads=0;
 reg [63:0] queue[1024];
 wire memvalid=head!=tail;
 wire [63:0] memdata=queue[tail];
 wire arready=cycles%3!=0,awready=cycles%5!=0;
 reg compare_reads=1;
 `define PORTS(I) \
 .clk(clk),.ra_clk(ra_clk),.rst_n(rst_n),.ra_rst_n(ra_rst_n), \
 .itf_ra_awaddr(wa),.itf_ra_awdata(wd),.itf_ra_awvalid(wvalid),.itf_ra_awready(wready[I]), \
 .itf_ra_araddr(ra),.itf_ra_arvalid(rvalid),.itf_ra_arready(rready[I]), \
 .itf_ra_rdata(rd[I]),.itf_ra_rvalid(rdvalid[I]),.itf_ra_rready(rdready), \
 .itf_araddr(araddr[I]),.itf_arvalid(arvalid[I]),.itf_arready(arready), \
 .itf_awaddr(awaddr[I]),.itf_awdata(awdata[I]),.itf_awvalid(awvalid[I]),.itf_awready(awready), \
 .itf_rdata(memdata),.itf_rvalid(memvalid),.itf_rready(memready[I]),.reset_reg(reset_reg[I])
 legacy_adder_top dut(`PORTS(0));
 legacy_reference reference(`PORTS(1));
 `undef PORTS
 always @(posedge clk) begin
  cycles<=cycles+1;
  if(rst_n) begin
   if({arvalid[0],awvalid[0],memready[0],reset_reg[0]}!=={arvalid[1],awvalid[1],memready[1],reset_reg[1]})$fatal(1,"DMA control differs from retained vendor reference");
   if(arvalid[0]&&araddr[0]!==araddr[1])$fatal(1,"read address differs");
   if(awvalid[0]&&{awaddr[0],awdata[0]}!=={awaddr[1],awdata[1]})$fatal(1,"write payload differs");
   if(arvalid[0]&&arready) begin
    if(araddr[0]!==32'd4096+reads*64)$fatal(1,"unexpected vendor read address");
    queue[head]<=64'h12340000+reads;head<=head+1;reads<=reads+1;
   end
   if(memvalid&&memready[0])tail<=tail+1;
   if(awvalid[0]&&awready) begin
    if(awaddr[0]!==32'd16384+writes*64||awdata[0]!==64'h12340001+writes)$fatal(1,"vendor arithmetic/address mismatch");
    writes<=writes+1;
   end
  end else begin head<=0;tail<=0;writes<=0;reads<=0;end
 end
 task wr(input [31:0] address,input [31:0] data);
  begin @(negedge ra_clk);wa=address;wd=data;wvalid=1;#1;
   if(wready!==2'b11)$fatal(1,"write handshake");
   @(negedge ra_clk);wvalid=0;
  end
 endtask
 task rdreg(input [31:0] address,output [31:0] data);
  integer n;
  begin
   @(negedge ra_clk);ra=address;rvalid=1;n=0;#1;
   while(rready[0]!==1&&n<100)begin @(negedge ra_clk);n=n+1;end
   if(rready[0]!==1)$fatal(1,"read request timeout");
   @(negedge ra_clk);rvalid=0;n=0;
   while(rdvalid[0]!==1&&n<100)begin @(negedge ra_clk);n=n+1;end
   if(rdvalid[0]!==1)$fatal(1,"read response timeout");
   data=rd[0];
   if(compare_reads&&(rdvalid[1]!==1||rd[0]!==rd[1]))$fatal(1,"vendor register differs");
   repeat(3)begin @(negedge ra_clk);if(rdvalid[0]!==1||rd[0]!==data)$fatal(1,"stalled read changed");end
   rdready=1;@(negedge ra_clk);rdready=0;
  end
 endtask
 integer trial,n,polls;reg [31:0] value;
 initial begin #2000000;$fatal(1,"watchdog");end
 initial begin
  for(trial=0;trial<3;trial=trial+1)begin
   rst_n=0;ra_rst_n=0;repeat(6)@(negedge ra_clk);
   rst_n=1;ra_rst_n=1;repeat(6)@(negedge ra_clk);
   rdreg('hc0,value);if(value!==32'h20230628)$fatal(1,"vendor version missing");
   wr('h3fc,32'h76543210+trial);rdreg('h3fc,value);
   if(value!==32'h76543210+trial)$fatal(1,"debug register mismatch");
   case(trial)0:n=16;1:n=64;2:n=256;endcase
   wr(4,(n<<16)|4096);wr(8,(n<<16)|16384);wr(0,1);
   polls=0;
   while(writes<n+1&&polls<3000)begin @(negedge clk);polls=polls+1;end
   if(writes!=n+1||reads!=n+1)$fatal(1,"legacy count+1 behavior changed");
   repeat(12)@(negedge ra_clk);rdreg('h80,value);if(value!=1)$fatal(1,"completion missing");
   $display("PASS: vendor n=%0d reads=%0d writes=%0d exact arithmetic and cycle-identical DMA under stalls",n,reads,writes);
  end
  compare_reads=0;
  rdreg('h9c,value);if(value!==0)$fatal(1,"removed GSC1 capability advertised");
  rdreg('h100,value);if(value!==32'hffffffff)$fatal(1,"removed GSB1 capability advertised");
  $display("PASS: removed capabilities absent; retained vendor path and read backpressure verified");
  $finish;
 end
endmodule
