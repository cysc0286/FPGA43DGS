`timescale 1ns/1ps
module tb_camera_servo_pwm;
    reg clk = 0;
    always #50 clk = ~clk; // 10 MHz: exercises the 10-cycle microsecond divider.
    reg rst_n = 0;
    reg [5:0] awaddr = 0, araddr = 0;
    reg [31:0] awdata = 0;
    reg awvalid = 0, arvalid = 0, rready = 1;
    wire awready, arready, rvalid, pan, tilt;
    wire [31:0] rdata;
    reg [31:0] value;
    integer checks = 0;
    time started;
    camera_servo_pwm #(.CLOCK_HZ(10000000), .WATCHDOG_US(100000)) dut (
        .i_clk(clk), .i_rst_n(rst_n), .i_awaddr(awaddr), .i_awdata(awdata),
        .i_awvalid(awvalid), .o_awready(awready), .i_araddr(araddr),
        .i_arvalid(arvalid), .o_arready(arready), .o_rdata(rdata),
        .o_rvalid(rvalid), .i_rready(rready), .o_pan_pwm(pan), .o_tilt_pwm(tilt)
    );
    task check;
        input condition;
        input [511:0] message;
        begin
            checks = checks + 1;
            if (!condition) begin
                $display("FAIL: %0s at %0t", message, $time);
                $finish;
            end
        end
    endtask
    task write_reg;
        input [5:0] address;
        input [31:0] data;
        begin
            @(negedge clk); awaddr = address; awdata = data; awvalid = 1;
            @(posedge clk); #1; check(awready, "write accepted");
            @(negedge clk); awvalid = 0;
        end
    endtask
    task read_reg;
        input [5:0] address;
        output [31:0] data;
        begin
            @(negedge clk); araddr = address; arvalid = 1;
            @(posedge clk); #1; check(rvalid, "read response valid"); data = rdata;
            @(negedge clk); arvalid = 0;
        end
    endtask
    task pan_width;
        input integer width;
        time t;
        begin
            @(posedge pan); t = $time;
            @(negedge pan); check($time - t == width * 1000, "pan pulse width");
        end
    endtask
    task tilt_width;
        input integer width;
        time t;
        begin
            @(posedge tilt); t = $time;
            @(negedge tilt); check($time - t == width * 1000, "tilt pulse width");
        end
    endtask
    initial begin
        #1000; check(!pan && !tilt, "reset outputs disabled");
        @(negedge clk); rst_n = 1;
        read_reg(6'h00, value); check(value == 32'h53525631, "module identity");
        read_reg(6'h04, value); check(value == 3, "complete-pulse-stop ABI");
        read_reg(6'h34, value); check(value == {16'd2500,16'd500}, "pan profile");
        read_reg(6'h38, value); check(value == {16'd2000,16'd1000}, "pitch profile");
        read_reg(6'h08, value); check(value == 10000000, "configured clock");
        read_reg(6'h10, value); check(value == 0, "reset enable mask");
        write_reg(6'h14, 1450); write_reg(6'h18, 1550);
        write_reg(6'h1c, 1); write_reg(6'h10, 3);
        pan_width(1450); tilt_width(1550);
        write_reg(6'h1c, 1);
        // Shadow writes during an active pulse must not alter that pulse.
        @(posedge pan); started = $time;
        #100000; write_reg(6'h14, 1600);
        @(negedge pan); check($time - started == 1450000, "mid-pulse update held");
        pan_width(1600);
        write_reg(6'h1c, 1);
        // Disable immediately, then restart only at a complete frame.
        @(posedge pan); #100000; write_reg(6'h10, 0);
        #1; check(!pan && !tilt, "immediate disable");
        write_reg(6'h10, 3); #1000; check(!pan && !tilt, "no partial re-enable");
        pan_width(1600);
        // A full 32-bit request is clamped before truncation.
        write_reg(6'h14, 0); write_reg(6'h18, 32'hffffffff);
        write_reg(6'h1c, 1);
        pan_width(500); tilt_width(2000);
        read_reg(6'h18, value); check(value == 2000, "tilt high clamp readback");
        write_reg(6'h18, 0); write_reg(6'h1c, 1);
        tilt_width(1000);
        read_reg(6'h18, value); check(value == 1000, "tilt low clamp readback");
        write_reg(6'h1c, 1);
        @(posedge pan); started = $time;
        @(posedge pan); check($time - started == 20000000, "50 Hz period");
        // Check read response ownership when the consumer is stalled.
        @(negedge clk); rready = 0; araddr = 0; arvalid = 1;
        @(posedge clk); #1; check(rvalid && rdata == 32'h53525631, "read started");
        @(negedge clk); arvalid = 0; araddr = 6'h08;
        repeat (3) begin
            @(posedge clk); #1;
            check(rvalid && !arready && rdata == 32'h53525631, "read backpressure");
        end
        @(negedge clk); rready = 1;
        #110000000; check(!pan && !tilt, "heartbeat timeout outputs disabled");
        read_reg(6'h20, value); check(value[8], "watchdog status");
        write_reg(6'h1c, 1); pan_width(500);
        @(negedge clk); rst_n = 0; #1;
        check(!pan && !tilt, "asynchronous reset during operation");
        $display("PASS: camera_servo_pwm %0d checks; 50Hz, dual pulse, clamp, frame update, watchdog, reset, read backpressure", checks);
        $finish;
    end
    initial begin
        #450000000;
        $display("FAIL: global simulation timeout");
        $finish;
    end
endmodule
