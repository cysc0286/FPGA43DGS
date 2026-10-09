`timescale 1ns/1ps
// 512-bit native PL-DDR transport. One outstanding read, a bounded elastic
// prefetch queue, independent writes. Addresses/counts are 32-bit, byte based.
// Completion requires HLS done, exact output count AND a read-after-write fence.
module flicker_dma # (parameter LGDEPTH=5)(
 input wire clk,rst_n,start,
 input wire [31:0] input_base,input_words,output_base,output_words,
 output reg busy,done,error,
 output reg kernel_start,input wire kernel_done,
 output wire [511:0] kernel_input,output wire kernel_empty_n,input wire kernel_read,
 input wire [511:0] kernel_output,input wire kernel_write,output wire kernel_full_n,
 output wire [31:0] araddr,output wire arvalid,input wire arready,
 input wire [511:0] rdata,input wire rvalid,output wire rready,
 output wire [31:0] awaddr,output wire [511:0] awdata,output wire awvalid,input wire awready,
 output reg [31:0] cycles,read_count,write_count,prefetch_overlap_cycles,input_starve_cycles
);
 localparam DEPTH=1<<LGDEPTH;
 reg [511:0] queue[0:DEPTH-1];
 reg [LGDEPTH-1:0] wp,rp;reg [LGDEPTH:0] level;
 // HLS FIFO WRITE can depend combinationally on FULL_N. The vendor DDR
 // arbiter can likewise derive READY from VALID: isolate them with storage.
 reg [511:0] output_queue[0:3];
 reg [1:0] output_wp,output_rp;reg [2:0] output_level;
 reg [31:0] produced;
 reg [31:0] ib,ob,ni,no,requested,consumed;
 reg outstanding,kdone,fencing;
 wire push=rvalid&&rready&&!fencing;
 wire pop=kernel_read&&kernel_empty_n;
 wire write_fire=awvalid&&awready;
 wire produce=kernel_write&&kernel_full_n;
 assign kernel_input=queue[rp];
 assign kernel_empty_n=busy&&!fencing&&(level!=0);
 assign kernel_full_n=busy&&!fencing&&(produced<no)&&(output_level<4);
 assign araddr=fencing?(ob+((no-1)<<6)):(ib+(requested<<6));
 assign arvalid=busy&&!outstanding&&(fencing||(requested<ni&&level<DEPTH));
 assign rready=busy&&outstanding&&(fencing||level<DEPTH);
 assign awaddr=ob+(write_count<<6);
 assign awdata=output_queue[output_rp];
 assign awvalid=busy&&!fencing&&(write_count<no)&&(output_level!=0);
 // Memory contents have no reset; validity is owned by resettable pointers.
 always @(posedge clk) begin
  if(rst_n && push)queue[wp]<=rdata;
  if(rst_n && produce)output_queue[output_wp]<=kernel_output;
 end
 always @(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
   busy<=0;done<=0;error<=0;kernel_start<=0;wp<=0;rp<=0;level<=0;
   ib<=0;ob<=0;ni<=0;no<=0;requested<=0;consumed<=0;outstanding<=0;kdone<=0;fencing<=0;
   output_wp<=0;output_rp<=0;output_level<=0;produced<=0;
   cycles<=0;read_count<=0;write_count<=0;prefetch_overlap_cycles<=0;input_starve_cycles<=0;
  end else begin
   kernel_start<=0;done<=0;
   if(start&&!busy)begin
    error<=0;wp<=0;rp<=0;level<=0;requested<=0;consumed<=0;outstanding<=0;kdone<=0;fencing<=0;
    output_wp<=0;output_rp<=0;output_level<=0;produced<=0;
    cycles<=0;read_count<=0;write_count<=0;prefetch_overlap_cycles<=0;input_starve_cycles<=0;
    ib<=input_base;ob<=output_base;ni<=input_words;no<=output_words;
    if(input_base[5:0]!=0||output_base[5:0]!=0||input_words==0||output_words==0)begin error<=1;done<=1;end
    else begin busy<=1;kernel_start<=1;end
   end else if(busy) begin
    cycles<=cycles+1;
    if(kernel_done)kdone<=1;
    if(kernel_read&&!kernel_empty_n)input_starve_cycles<=input_starve_cycles+1;
    // Actual read responses while the kernel is working on earlier input.
    if(push&&consumed!=0&&!kdone&&!kernel_read)prefetch_overlap_cycles<=prefetch_overlap_cycles+1;
    if(arvalid&&arready)begin outstanding<=1;if(!fencing)requested<=requested+1;end
    if(rvalid&&rready)begin
     outstanding<=0;
     if(fencing)begin busy<=0;done<=1;fencing<=0;end
     else begin wp<=wp+1;read_count<=read_count+1;end
    end
    if(pop)begin rp<=rp+1;consumed<=consumed+1;end
    case({push,pop})
     2'b10:level<=level+1;
     2'b01:level<=level-1;
     default:level<=level;
    endcase
    if(produce)begin output_wp<=output_wp+1;produced<=produced+1;end
    if(write_fire)begin write_count<=write_count+1;output_rp<=output_rp+1;end
    case({produce,write_fire})
     2'b10:output_level<=output_level+1;
     2'b01:output_level<=output_level-1;
     default:output_level<=output_level;
    endcase
    if(kdone&&write_count==no&&read_count==ni&&consumed==ni&&!outstanding&&!fencing)fencing<=1;
    // A kernel ending without consuming all declared input is a protocol error.
    if(kdone&&consumed!=ni)begin error<=1;end
   end
  end
 end
endmodule
