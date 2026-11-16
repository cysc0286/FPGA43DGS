`timescale 1ns/1ps
module tb_camera_servo_bus;
    reg clk=0; always #50 clk=~clk;
    reg rst_n=0;
    reg [31:0] awaddr=0, awdata=0, araddr=0;
    reg awvalid=0, arvalid=0, rready=0;
    wire awready, arready, rvalid, pan, tilt;
    wire [31:0] rdata, dwa, dwd, dra;
    wire dwvalid, drvalid, drready;
    reg dwready=1, daready=1, dvalid=0;
    reg [31:0] ddata=0;
    integer checks=0, legacy_writes=0;
    camera_servo_bus #(.CLOCK_HZ(10000000)) dut (
        .clk(clk),.rst_n(rst_n),.awaddr(awaddr),.awdata(awdata),
        .awvalid(awvalid),.awready(awready),.araddr(araddr),.arvalid(arvalid),
        .arready(arready),.rdata(rdata),.rvalid(rvalid),.rready(rready),
        .downstream_awaddr(dwa),.downstream_awdata(dwd),
        .downstream_awvalid(dwvalid),.downstream_awready(dwready),
        .downstream_araddr(dra),.downstream_arvalid(drvalid),
        .downstream_arready(daready),.downstream_rdata(ddata),
        .downstream_rvalid(dvalid),.downstream_rready(drready),
        .pan_pwm(pan),.tilt_pwm(tilt)
    );
    // Deliberately delayed downstream read response, with backpressure.
    integer delay_count=0;
    always @(posedge clk) begin
        if (!rst_n) begin dvalid<=0;delay_count<=0;legacy_writes<=0;end
        else begin
            if(dwvalid && dwready)legacy_writes<=legacy_writes+1;
            if(drvalid && daready)begin delay_count<=3;ddata<=dra ^ 32'h12345678;end
            if(delay_count>0)begin
                delay_count<=delay_count-1;
                if(delay_count==1)dvalid<=1;
            end
            if(dvalid && drready)dvalid<=0;
        end
    end
    task check;
        input condition;input [511:0] message;
        begin checks=checks+1;if(!condition)begin
            $display("FAIL: %0s",message);$finish;end end
    endtask
    task read_request;
        input [31:0] address;
        begin
            @(negedge clk);araddr=address;arvalid=1;
            @(posedge clk);check(arready,"read accepted");
            @(negedge clk);arvalid=0;
        end
    endtask
    initial begin
        #200;@(negedge clk);rst_n=1;
        // A servo write must not reach the legacy registers.
        @(negedge clk);awaddr=32'h400c0318;awdata=32'hffffffff;awvalid=1;
        @(posedge clk);check(awready && !dwvalid,"servo write isolated");
        @(negedge clk);awvalid=0;
        read_request(32'h400c0318);
        #1;check(rvalid && rdata==2000,"hardware pitch clamp through router");
        repeat(3)begin @(posedge clk);#1;check(!arready && rvalid && rdata==2000,"servo read held");end
        @(negedge clk);rready=1;@(posedge clk);@(negedge clk);rready=0;
        // The existing 0x200 renderer window passes through unchanged.
        @(negedge clk);awaddr=32'h400c022c;awdata=32'haabbccdd;awvalid=1;
        #1;check(dwvalid && dwa==awaddr && dwd==awdata,"renderer write forwarded");
        @(posedge clk);@(negedge clk);awvalid=0;
        read_request(32'h400c0200);
        repeat(2)begin @(posedge clk);#1;check(!arready,"read ownership retained during latency");end
        wait(rvalid);#1;check(rdata==(32'h400c0200 ^ 32'h12345678),"renderer response returned");
        // Change the address while a previous response is held: ownership is fixed.
        @(negedge clk);araddr=32'h400c0300;arvalid=1;
        repeat(3)begin @(posedge clk);#1;check(!arready && !drready && rvalid && rdata==(32'h400c0200 ^ 32'h12345678),"no cross-target response theft");end
        @(negedge clk);arvalid=0;rready=1;@(posedge clk);@(negedge clk);rready=0;
        read_request(32'h400c0300);#1;check(rvalid && rdata==32'h53525631,"servo identity after renderer transaction");
        check(legacy_writes==1,"servo did not alias legacy writes");
        check(!pan && !tilt,"tests never armed outputs");
        $display("PASS: camera_servo_bus %0d checks; routing, latency, ownership, pitch limits",checks);$finish;
    end
    initial begin #100000;$display("FAIL: router timeout");$finish;end
endmodule
