`timescale 1ns/1ps
// One active descriptor + one queued descriptor. Bundled-data toggle CDC:
// source holds payload until acknowledgement; destination snapshots before ack.
// Completion payload stays stable until software acknowledges its event.
module flicker_regs(
 input wire gp_clk,gp_rst_n,hp_clk,hp_rst_n,
 input wire [7:0] wa,ra,input wire [31:0] wd,input wire wvalid,rvalid,
 output wire wready,rready,output reg [31:0] rd,output reg rdvalid,input wire rdready,
 output reg start,output reg [31:0] ib,ni,ob,no,tiles,mode,
 input wire busy,done,error,input wire [31:0] cycles,reads,writes,overlap,starve,
 output wire owned
);
 reg [31:0] shadow[0:6];reg req,ack,clear_req;reg rejected;
 (* ASYNC_REG="TRUE" *) reg ack_g1,ack_g2,comp_g1,comp_g2,busy_g1,busy_g2,pend_g1,pend_g2;
 (* ASYNC_REG="TRUE" *) reg req_h1,req_h2,clear_h1,clear_h2;
 reg completed,pending,active;reg [31:0] job,queued[0:6],result[0:7];
 assign wready=1;assign rready=!rdvalid||rdready;
 assign owned=active||pending;
 integer i;
 always @(posedge gp_clk or negedge gp_rst_n)begin
  if(!gp_rst_n)begin
   req<=0;clear_req<=0;rejected<=0;rd<=0;rdvalid<=0;
   ack_g1<=0;ack_g2<=0;comp_g1<=0;comp_g2<=0;busy_g1<=0;busy_g2<=0;pend_g1<=0;pend_g2<=0;
   for(i=0;i<7;i=i+1)shadow[i]<=0;
  end else begin
   ack_g1<=ack;ack_g2<=ack_g1;comp_g1<=completed;comp_g2<=comp_g1;
   busy_g1<=active;busy_g2<=busy_g1;pend_g1<=pending;pend_g2<=pend_g1;
   if(rdready)rdvalid<=0;
   if(wvalid)begin
    if(wa>=8'h10&&wa<=8'h28&&wa[1:0]==0)begin
     if(req==ack_g2)shadow[(wa-8'h10)>>2]<=wd;else rejected<=1;
    end
    if(wa==8'h2c)begin
     if(wd[2])rejected<=0;
     if(wd[1]&&comp_g2!=clear_req)clear_req<=comp_g2;
     if(wd[0])begin
      if(req!=ack_g2||shadow[0][5:0]!=0||shadow[2][5:0]!=0||shadow[1]==0||shadow[4]==0||
         shadow[3]!=(shadow[4]<<6)||shadow[6]>5||shadow[5]==0)rejected<=1;
      else req<=~req;
     end
    end
   end
   if(rvalid&&rready)begin
    rdvalid<=1;
    case(ra)
     8'h00:rd<=32'h464c4b31;
     8'h04:rd<=32'h00020000;
     8'h08:rd<={27'd0,(req!=ack_g2),rejected,(comp_g2!=clear_req),pend_g2,busy_g2};
     8'h30:rd<=result[0];8'h34:rd<=result[1];8'h38:rd<=result[2];8'h3c:rd<=result[3];
     8'h40:rd<=result[4];8'h44:rd<=result[5];8'h48:rd<=result[6];
     default:rd<=32'hffffffff;
    endcase
   end
  end
 end
 integer k;
 always @(posedge hp_clk or negedge hp_rst_n)begin
  if(!hp_rst_n)begin
   req_h1<=0;req_h2<=0;clear_h1<=0;clear_h2<=0;ack<=0;completed<=0;pending<=0;active<=0;start<=0;
   ib<=0;ni<=0;ob<=0;no<=0;tiles<=0;mode<=0;job<=0;
   for(k=0;k<7;k=k+1)queued[k]<=0;
   for(k=0;k<8;k=k+1)result[k]<=0;
  end else begin
   req_h1<=req;req_h2<=req_h1;clear_h1<=clear_req;clear_h2<=clear_h1;start<=0;
   if(req_h2!=ack&&!pending)begin
    for(k=0;k<7;k=k+1)queued[k]<=shadow[k];pending<=1;ack<=req_h2;
   end
   if(pending&&!active&&!busy&&completed==clear_h2)begin
    ib<=queued[0];ni<=queued[1];ob<=queued[2];no<=queued[3];tiles<=queued[4];job<=queued[5];mode<=queued[6];
    pending<=0;active<=1;start<=1;
   end
   if(done&&active)begin
    result[0]<=job;result[1]<=cycles;result[2]<=reads;result[3]<=writes;result[4]<=overlap;result[5]<=starve;
    result[6]<={31'b0,error};completed<=~completed;active<=0;
   end
  end
 end
endmodule
