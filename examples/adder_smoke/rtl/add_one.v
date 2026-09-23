`timescale 1ns / 1ps

// Registered unsigned incrementer. Outputs update on the rising clock edge.
// Overflow wraps modulo 2^32. in_valid=0 creates an invalid output cycle.
module add_one (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire [31:0] in_data,
    output reg         out_valid,
    output reg  [31:0] out_data
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_data  <= 32'b0;
        end else begin
            out_valid <= in_valid;
            if (in_valid)
                out_data <= in_data + 32'd1;
        end
    end
endmodule
