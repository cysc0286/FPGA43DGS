`timescale 1ns/1ps
module tb_camera_servo_stop;
    parameter integer CLOCK_HZ=10000000;
    reg clk=0;
    always #(500000000/CLOCK_HZ) clk=~clk;
    reg rst_n=0;
    reg [5:0] addr=0,araddr=0;
    reg [31:0] data=0;
    reg valid=0,arvalid=0;
    wire tilt,pan,rvalid;
    wire [31:0] rdata;
    time pulse_start,width;
    integer checks=0;
    camera_servo_pwm #(.CLOCK_HZ(CLOCK_HZ)) dut (
        .i_clk(clk),.i_rst_n(rst_n),.i_awaddr(addr),.i_awdata(data),.i_awvalid(valid),
        .o_awready(),.i_araddr(araddr),.i_arvalid(arvalid),.o_arready(),
        .o_rdata(rdata),.o_rvalid(rvalid),.i_rready(1'b1),.o_pan_pwm(pan),.o_tilt_pwm(tilt)
    );
    task check;
        input condition;
        input [511:0] message;
        begin
            checks=checks+1;
            if(!condition)$fatal(1,"FAIL: %0s at %0t",message,$time);
        end
    endtask
    task write_reg;
        input [5:0] a;
        input [31:0] d;
        begin
            @(negedge clk);addr=a;data=d;valid=1;
            @(negedge clk);valid=0;
        end
    endtask
    task arm;
        begin
            write_reg(6'h14,2500);write_reg(6'h18,1250);
            write_reg(6'h1c,1);write_reg(6'h10,3);
        end
    endtask
    task normal_stop;
        input integer delay_ns;
        time pan_start,tilt_start,pan_width,tilt_width;
        begin
            arm();
            @(posedge pan);pan_start=$time;tilt_start=$time;
            fork
                begin #delay_ns;write_reg(6'h3c,1);end
                begin @(negedge pan);pan_width=$time-pan_start;end
                begin @(negedge tilt);tilt_width=$time-tilt_start;end
            join
            check(pan_width==2500000,"normal stop preserves full pan pulse");
            check(tilt_width==1250000,"normal stop preserves full tilt pulse");
            #21000000;
            check(!pan && !tilt && dut.r_requested_enable==0 && dut.r_active_enable==0,"normal stop acknowledged");
            check(!dut.r_stop_pending,"pending stop consumed");
        end
    endtask
    initial begin
        #1000;@(negedge clk);rst_n=1;
        @(negedge clk);araddr=6'h04;arvalid=1;
        @(posedge clk);#1;check(rvalid && rdata==3,"graceful stop ABI3");
        @(negedge clk);arvalid=0;
        normal_stop(100000);
        normal_stop(1240000);
        normal_stop(2000000);
        normal_stop(19999000);
        // A stop requested exactly at a frame boundary may take one extra
        // frame; it must still output only full pulses before acknowledgement.
        arm();
        @(posedge pan);pulse_start=$time;
        #19999900;write_reg(6'h3c,1);
        #41000000;
        check(!pan && !tilt && !dut.r_requested_enable,"boundary-race stop acknowledged");
        arm();@(posedge tilt);pulse_start=$time;
        fork
            begin #250000;write_reg(6'h10,0);end
            begin @(negedge tilt);width=$time-pulse_start;end
        join
        check(width<1250000,"emergency disable remains immediate");
        check(!pan && !tilt,"emergency disables both axes");
        $display("PASS: camera_servo_stop %0d checks clock_hz=%0d; normal pulses intact, emergency immediate",checks,CLOCK_HZ);
        $finish;
    end
    initial begin #400000000;$fatal(1,"global timeout");end
endmodule
