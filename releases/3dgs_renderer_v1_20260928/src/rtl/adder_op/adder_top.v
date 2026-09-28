// 30TAI native user interface adapter. Legacy register/DMA ABI is retained.
module adder_top(
 input [31:0] itf_ra_awaddr,itf_ra_awdata,input itf_ra_awvalid,output itf_ra_awready,
 input [31:0] itf_ra_araddr,input itf_ra_arvalid,output itf_ra_arready,
 output [31:0] itf_ra_rdata,output itf_ra_rvalid,input itf_ra_rready,
 output [31:0] itf_awaddr,output [511:0]itf_awdata,output itf_awvalid,input itf_awready,
 output [31:0] itf_araddr,output itf_arvalid,input itf_arready,
 input [511:0] itf_rdata,input itf_rvalid,output itf_rready,
 output reset_reg,input clk,ra_clk,rst_n,ra_rst_n
);
 wire ws=itf_ra_awaddr[17:8]==10'h002,rs=itf_ra_araddr[17:8]==10'h002;
 wire lwready,lrready,lrvalid,fwready,frready,frvalid;wire[31:0]lrdata,frdata;
 wire free_read=!lrvalid&&!frvalid;
 assign itf_ra_awready=ws?fwready:lwready;
 assign itf_ra_arready=free_read&&(rs?frready:lrready);
 assign itf_ra_rvalid=lrvalid||frvalid;
 assign itf_ra_rdata=frvalid?frdata:lrdata;
 wire owned,larv,lawv,lrr,farv,fawv,frr;
 wire[31:0]lara,lawa,fara,fawa;wire[63:0]lawd;wire[511:0]fawd;
 assign itf_awaddr=owned?fawa:lawa;assign itf_awdata=owned?fawd:{448'd0,lawd};
 assign itf_awvalid=owned?fawv:lawv;assign itf_araddr=owned?fara:lara;
 assign itf_arvalid=owned?farv:larv;assign itf_rready=owned?frr:lrr;
 legacy_adder_top legacy(
  .itf_ra_awaddr(itf_ra_awaddr),.itf_ra_awdata(itf_ra_awdata),.itf_ra_awvalid(itf_ra_awvalid&&!ws),.itf_ra_awready(lwready),
  .itf_ra_araddr(itf_ra_araddr),.itf_ra_arvalid(itf_ra_arvalid&&!rs&&free_read),.itf_ra_arready(lrready),
  .itf_ra_rdata(lrdata),.itf_ra_rvalid(lrvalid),.itf_ra_rready(itf_ra_rready&&lrvalid),
  .itf_awaddr(lawa),.itf_awdata(lawd),.itf_awvalid(lawv),.itf_awready(itf_awready&&!owned),
  .itf_araddr(lara),.itf_arvalid(larv),.itf_arready(itf_arready&&!owned),
  .itf_rdata(itf_rdata[63:0]),.itf_rvalid(itf_rvalid&&!owned),.itf_rready(lrr),
  .reset_reg(reset_reg),.clk(clk),.ra_clk(ra_clk),.rst_n(rst_n),.ra_rst_n(ra_rst_n));
 flicker_top U_flicker(.gp_clk(ra_clk),.gp_rst_n(ra_rst_n),.hp_clk(clk),.hp_rst_n(rst_n),
  .wa(itf_ra_awaddr[7:0]),.ra(itf_ra_araddr[7:0]),.wd(itf_ra_awdata),
  .wvalid(itf_ra_awvalid&&ws),.rvalid(itf_ra_arvalid&&rs&&free_read),
  .wready(fwready),.rready(frready),.rd(frdata),.rdvalid(frvalid),.rdready(itf_ra_rready&&frvalid),
  .araddr(fara),.arvalid(farv),.arready(itf_arready&&owned),.rdata(itf_rdata),.rvalid_mem(itf_rvalid&&owned),.rready_mem(frr),
  .awaddr(fawa),.awdata(fawd),.awvalid(fawv),.awready(itf_awready&&owned),.owned(owned));
endmodule
