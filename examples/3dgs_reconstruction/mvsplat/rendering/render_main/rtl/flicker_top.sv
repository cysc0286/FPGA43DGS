module flicker_top(
 input wire gp_clk,gp_rst_n,hp_clk,hp_rst_n,
 input wire [7:0] wa,ra,input wire [31:0] wd,input wire wvalid,rvalid,
 output wire wready,rready,output wire [31:0] rd,output wire rdvalid,input wire rdready,
 output wire [31:0] araddr,output wire arvalid,input wire arready,
 input wire [511:0] rdata,input wire rvalid_mem,output wire rready_mem,
 output wire [31:0] awaddr,output wire [511:0] awdata,output wire awvalid,input wire awready,
 output wire owned
);
 wire start,busy,done,error,kstart,kdone;
 wire [31:0] ib,ni,ob,no,tiles,mode,cycles,reads,writes,overlap,starve;
 wire [511:0] ki,ko;wire empty_n,kr,kw,full_n;
 flicker_regs regs(.gp_clk(gp_clk),.gp_rst_n(gp_rst_n),.hp_clk(hp_clk),.hp_rst_n(hp_rst_n),
  .wa(wa),.ra(ra),.wd(wd),.wvalid(wvalid),.rvalid(rvalid),.wready(wready),.rready(rready),.rd(rd),.rdvalid(rdvalid),.rdready(rdready),
  .start(start),.ib(ib),.ni(ni),.ob(ob),.no(no),.tiles(tiles),.mode(mode),.busy(busy),.done(done),.error(error),
  .cycles(cycles),.reads(reads),.writes(writes),.overlap(overlap),.starve(starve),.owned(owned));
 flicker_dma dma(.clk(hp_clk),.rst_n(hp_rst_n),.start(start),.input_base(ib),.input_words(ni),.output_base(ob),.output_words(no),
  .busy(busy),.done(done),.error(error),.kernel_start(kstart),.kernel_done(kdone),
  .kernel_input(ki),.kernel_empty_n(empty_n),.kernel_read(kr),.kernel_output(ko),.kernel_write(kw),.kernel_full_n(full_n),
  .araddr(araddr),.arvalid(arvalid),.arready(arready),.rdata(rdata),.rvalid(rvalid_mem),.rready(rready_mem),
  .awaddr(awaddr),.awdata(awdata),.awvalid(awvalid),.awready(awready),.cycles(cycles),.read_count(reads),.write_count(writes),
  .prefetch_overlap_cycles(overlap),.input_starve_cycles(starve));
 flicker_render_pipeline render(.ap_clk(hp_clk),.ap_rst(!hp_rst_n),.ap_start(kstart),.ap_done(kdone),.ap_idle(),.ap_ready(),
  .input_V_V_dout(ki),.input_V_V_empty_n(empty_n),.input_V_V_read(kr),
  .output_V_V_din(ko),.output_V_V_full_n(full_n),.output_V_V_write(kw),.tiles(tiles),.mode(mode));
endmodule
