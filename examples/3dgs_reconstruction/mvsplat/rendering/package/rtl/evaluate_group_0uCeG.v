// ==============================================================
// File generated on Wed Sep 30 21:31:47 +0800 2026
// Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2018.3 (64-bit)
// SW Build 2405991 on Thu Dec  6 23:38:27 MST 2018
// IP Build 2404404 on Fri Dec  7 01:43:56 MST 2018
// Copyright 1986-2018 Xilinx, Inc. All Rights Reserved.
// ==============================================================
`timescale 1 ns / 1 ps
(* rom_style = "block" *) module evaluate_group_0uCeG_rom (
addr0, ce0, q0, clk);

parameter DWIDTH = 14;
parameter AWIDTH = 14;
parameter MEM_SIZE = 15066;

input[AWIDTH-1:0] addr0;
input ce0;
output reg[DWIDTH-1:0] q0;
input clk;

(* ram_style = "block" *)reg [DWIDTH-1:0] ram[0:MEM_SIZE-1];

initial begin
    $readmemh("D:/ADProjects/FPGA43DGS/platform/flicker_groupshared_fpga/rtl/flicker/evaluate_group_0uCeG_rom.dat", ram);
end



always @(posedge clk)  
begin 
    if (ce0) 
    begin
        q0 <= ram[addr0];
    end
end



endmodule

`timescale 1 ns / 1 ps
module evaluate_group_0uCeG(
    reset,
    clk,
    address0,
    ce0,
    q0);

parameter DataWidth = 32'd14;
parameter AddressRange = 32'd15066;
parameter AddressWidth = 32'd14;
input reset;
input clk;
input[AddressWidth - 1:0] address0;
input ce0;
output[DataWidth - 1:0] q0;



evaluate_group_0uCeG_rom evaluate_group_0uCeG_rom_U(
    .clk( clk ),
    .addr0( address0 ),
    .ce0( ce0 ),
    .q0( q0 ));

endmodule

