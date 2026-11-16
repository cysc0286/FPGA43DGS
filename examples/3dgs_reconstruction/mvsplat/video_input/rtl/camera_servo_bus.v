`timescale 1ns/1ps
// Candidate native-register router. 0x300--0x33f is a NEW design allocation,
// not an address that is safe to access on the current frozen firmware.
// All other addresses pass to the existing rendering/NPU custom-op adapter.
module camera_servo_bus #(
    parameter integer CLOCK_HZ = 100000000
) (
    input wire clk, rst_n,
    input wire [31:0] awaddr, awdata,
    input wire awvalid, output wire awready,
    input wire [31:0] araddr, input wire arvalid, output wire arready,
    output wire [31:0] rdata, output wire rvalid, input wire rready,
    output wire [31:0] downstream_awaddr, downstream_awdata,
    output wire downstream_awvalid, input wire downstream_awready,
    output wire [31:0] downstream_araddr,
    output wire downstream_arvalid, input wire downstream_arready,
    input wire [31:0] downstream_rdata, input wire downstream_rvalid,
    output wire downstream_rready,
    output wire pan_pwm, tilt_pwm
);
    wire write_servo = awaddr[17:6] == 12'h00c;
    wire read_servo = araddr[17:6] == 12'h00c;
    wire servo_awready, servo_arready, servo_rvalid;
    wire [31:0] servo_rdata;
    reg read_pending, read_owner_servo;

    assign downstream_awaddr = awaddr;
    assign downstream_awdata = awdata;
    assign downstream_araddr = araddr;
    assign downstream_awvalid = rst_n && awvalid && !write_servo;
    assign awready = rst_n && (write_servo ? servo_awready : downstream_awready);
    assign downstream_arvalid = rst_n && !read_pending && arvalid && !read_servo;
    assign arready = rst_n && !read_pending &&
                     (read_servo ? servo_arready : downstream_arready);
    assign rvalid = rst_n && read_pending &&
                    (read_owner_servo ? servo_rvalid : downstream_rvalid);
    assign rdata = read_owner_servo ? servo_rdata : downstream_rdata;
    assign downstream_rready = rst_n && read_pending && !read_owner_servo && rready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_pending <= 1'b0;
            read_owner_servo <= 1'b0;
        end else begin
            if (arvalid && arready) begin
                read_pending <= 1'b1;
                read_owner_servo <= read_servo;
            end else if (rvalid && rready) begin
                read_pending <= 1'b0;
            end
        end
    end

    camera_servo_pwm #(.CLOCK_HZ(CLOCK_HZ)) pwm (
        .i_clk(clk), .i_rst_n(rst_n),
        .i_awaddr(awaddr[5:0]), .i_awdata(awdata),
        .i_awvalid(awvalid && write_servo), .o_awready(servo_awready),
        .i_araddr(araddr[5:0]), .i_arvalid(arvalid && read_servo && !read_pending),
        .o_arready(servo_arready), .o_rdata(servo_rdata), .o_rvalid(servo_rvalid),
        .i_rready(read_pending && read_owner_servo && rready),
        .o_pan_pwm(pan_pwm), .o_tilt_pwm(tilt_pwm)
    );
endmodule
