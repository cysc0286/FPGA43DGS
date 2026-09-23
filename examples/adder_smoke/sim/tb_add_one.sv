`timescale 1ns / 1ps

module tb_add_one;
    reg clk = 0;
    reg rst_n = 0;
    reg in_valid = 0;
    reg [31:0] in_data = 0;
    wire out_valid;
    wire [31:0] out_data;
    integer checks = 0;
    integer i;
    reg [31:0] stimulus = 32'h13579bdf;
    reg [31:0] held_data;
    reg [63:0] reference_sum;

    add_one dut (.*);
    always #5 clk = ~clk;

    task automatic check_cycle(
        input bit valid,
        input reg [31:0] value,
        input reg [31:0] expected
    );
        @(negedge clk);
        in_valid = valid;
        in_data = value;
        @(posedge clk);
        #1;
        if (out_valid !== valid || out_data !== expected)
            $fatal(1, "FAIL: valid=%b data=%h, expected valid=%b data=%h",
                   out_valid, out_data, valid, expected);
        checks = checks + 1;
    endtask

    initial begin
        #1;
        @(posedge clk);
        #1;
        if (out_valid !== 0 || out_data !== 0)
            $fatal(1, "FAIL: initial reset");
        checks = checks + 1;
        @(negedge clk);
        rst_n = 1;

        check_cycle(1, 32'd0,         32'd1);
        check_cycle(1, 32'd15,        32'd16);
        check_cycle(1, 32'h7fffffff,  32'h80000000);
        check_cycle(1, 32'hffffffff,  32'd0);
        check_cycle(0, 32'h12345678,  32'd0);

        // Deterministic data and bubbles, with an independent wide reference.
        for (i = 0; i < 256; i = i + 1) begin
            stimulus = stimulus * 32'd1664525 + 32'd1013904223;
            reference_sum = {32'd0, stimulus} + 64'd1;
            check_cycle(1, stimulus, reference_sum[31:0]);
            if ((i % 7) == 0) begin
                held_data = out_data;
                check_cycle(0, ~stimulus, held_data);
            end
        end

        // Reset with a valid output pending, between clock edges.
        check_cycle(1, 32'd99, 32'd100);
        #1 rst_n = 0;
        #1;
        if (out_valid !== 0 || out_data !== 0)
            $fatal(1, "FAIL: asynchronous reset");
        checks = checks + 1;
        @(negedge clk);
        in_valid = 0;
        rst_n = 1;
        check_cycle(0, 32'hffffffff, 32'd0);
        check_cycle(1, 32'd41, 32'd42);

        $display("PASS: add_one (%0d checks)", checks);
        $finish;
    end

    initial begin
        #10000;
        $fatal(1, "FAIL: simulation timeout");
    end
endmodule
