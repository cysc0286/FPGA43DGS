`timescale 1ns/1ps
// Two position servos on the board-side camera acquisition rig.
// Native combined address/data register interface; this is not AXI4-Lite.
// All register signals must share i_clk. GPIO package pins belong to the
// enclosing platform and must be verified against its board schematic.
module camera_servo_pwm #(
    parameter integer CLOCK_HZ = 100000000,
    parameter integer FRAME_US = 20000,
    parameter integer WATCHDOG_US = 500000,
    parameter integer PAN_MIN_PULSE_US = 500,
    parameter integer PAN_MAX_PULSE_US = 2500,
    // Electrical limits only. Installed camera neutral is not necessarily 1500us.
    parameter integer TILT_MIN_PULSE_US = 1000,
    parameter integer TILT_MAX_PULSE_US = 2000
) (
    input wire i_clk,
    input wire i_rst_n,
    input wire [5:0] i_awaddr,
    input wire [31:0] i_awdata,
    input wire i_awvalid,
    output wire o_awready,
    input wire [5:0] i_araddr,
    input wire i_arvalid,
    output wire o_arready,
    output wire [31:0] o_rdata,
    output wire o_rvalid,
    input wire i_rready,
    output wire o_pan_pwm,
    output wire o_tilt_pwm
);
    localparam integer CLOCKS_PER_US = CLOCK_HZ / 1000000;
    reg [31:0] r_divider;
    reg [31:0] r_phase_us;
    reg [31:0] r_watchdog_us;
    reg [31:0] r_frame_count;
    reg r_expired;
    reg r_stop_pending;
    reg [1:0] r_requested_enable;
    reg [1:0] r_active_enable;
    reg [15:0] r_pan_us;
    reg [15:0] r_tilt_us;
    reg [15:0] r_pan_active_us;
    reg [15:0] r_tilt_active_us;
    reg [31:0] r_read_data;
    reg r_read_valid;
    wire w_tick_us = r_divider == CLOCKS_PER_US - 1;

    function [15:0] clamp_pulse;
        input [31:0] requested;
        input [31:0] minimum;
        input [31:0] maximum;
        begin
            if (requested < minimum)
                clamp_pulse = minimum[15:0];
            else if (requested > maximum)
                clamp_pulse = maximum[15:0];
            else
                clamp_pulse = requested[15:0];
        end
    endfunction

    function [31:0] read_register;
        input [5:0] address;
        begin
            case (address)
                6'h00: read_register = 32'h53525631; // SRV1
                6'h04: read_register = 32'd3;
                6'h08: read_register = CLOCK_HZ;
                6'h0c: read_register = FRAME_US;
                6'h10: read_register = {30'd0, r_requested_enable};
                6'h14: read_register = {16'd0, r_pan_us};
                6'h18: read_register = {16'd0, r_tilt_us};
                6'h20: read_register = {23'd0, r_expired, 2'd0,
                                        r_requested_enable, 2'd0, r_active_enable};
                6'h24: read_register = r_frame_count;
                6'h28: read_register = {16'd0, r_pan_active_us};
                6'h2c: read_register = {16'd0, r_tilt_active_us};
                6'h30: read_register = WATCHDOG_US;
                6'h34: read_register = (PAN_MAX_PULSE_US << 16) | PAN_MIN_PULSE_US;
                6'h38: read_register = (TILT_MAX_PULSE_US << 16) | TILT_MIN_PULSE_US;
                6'h3c: read_register = {31'd0, r_stop_pending};
                default: read_register = 32'd0;
            endcase
        end
    endfunction

    assign o_awready = i_rst_n;
    assign o_arready = i_rst_n && (!r_read_valid || i_rready);
    assign o_rdata = r_read_data;
    assign o_rvalid = r_read_valid;
    // Disable and watchdog may truncate a pulse. Enabling can only take
    // effect at a frame boundary, preventing partial pulses on re-enable.
    assign o_pan_pwm = i_rst_n && !r_expired && r_requested_enable[0]
                       && r_active_enable[0] && r_phase_us < r_pan_active_us;
    assign o_tilt_pwm = i_rst_n && !r_expired && r_requested_enable[1]
                        && r_active_enable[1] && r_phase_us < r_tilt_active_us;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            r_divider <= 0;
            r_phase_us <= 0;
            r_watchdog_us <= 0;
            r_frame_count <= 0;
            r_expired <= 1'b1;
            r_stop_pending <= 1'b0;
            r_requested_enable <= 2'b00;
            r_active_enable <= 2'b00;
            r_pan_us <= 16'd1500;
            r_tilt_us <= 16'd1500;
            r_pan_active_us <= 16'd1500;
            r_tilt_active_us <= 16'd1500;
        end else begin
            if (w_tick_us) begin
                r_divider <= 0;
                if (r_phase_us == FRAME_US - 1) begin
                    r_phase_us <= 0;
                    r_frame_count <= r_frame_count + 1'b1;
                    r_pan_active_us <= r_pan_us;
                    r_tilt_active_us <= r_tilt_us;
                    r_active_enable <= (r_expired || r_stop_pending) ? 2'b00 : r_requested_enable;
                    if (r_stop_pending) r_requested_enable <= 2'b00;
                    r_stop_pending <= 1'b0;
                end else begin
                    r_phase_us <= r_phase_us + 1'b1;
                end
                // Once expired, outputs stay off until the host sends a
                // heartbeat. The next complete frame can then start.
                if (!r_expired) begin
                    if (r_watchdog_us >= WATCHDOG_US - 1) begin
                        r_expired <= 1'b1;
                        r_active_enable <= 2'b00;
                    end else begin
                        r_watchdog_us <= r_watchdog_us + 1'b1;
                    end
                end
            end else begin
                r_divider <= r_divider + 1'b1;
            end
            if (i_awvalid && o_awready) begin
                case (i_awaddr)
                    6'h10: begin
                        r_requested_enable <= i_awdata[1:0];
                        r_active_enable <= 2'b00;
                        r_stop_pending <= 1'b0;
                    end
                    6'h14: r_pan_us <= clamp_pulse(i_awdata, PAN_MIN_PULSE_US, PAN_MAX_PULSE_US);
                    6'h18: r_tilt_us <= clamp_pulse(i_awdata, TILT_MIN_PULSE_US, TILT_MAX_PULSE_US);
                    6'h1c: begin
                        r_watchdog_us <= 0;
                        r_expired <= 1'b0;
                    end
                    // Normal completion: finish the current pulse and disable
                    // at the next frame boundary. CONTROL=0 remains immediate.
                    6'h3c: if (i_awdata == 1) r_stop_pending <= 1'b1;
                    default: begin end
                endcase
            end
        end
    end

    // One outstanding read; data and VALID remain stable under backpressure.
    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            r_read_valid <= 1'b0;
            r_read_data <= 0;
        end else if (o_arready) begin
            r_read_valid <= i_arvalid;
            if (i_arvalid)
                r_read_data <= read_register(i_araddr);
        end
    end

    // synthesis translate_off
    initial begin
        if (CLOCK_HZ < 1000000 || CLOCK_HZ % 1000000 != 0 || FRAME_US < 3000
            || WATCHDOG_US < FRAME_US || PAN_MIN_PULSE_US < 1
            || PAN_MAX_PULSE_US >= FRAME_US || PAN_MAX_PULSE_US > 65535
            || PAN_MIN_PULSE_US > 1500 || PAN_MAX_PULSE_US < 1500
            || TILT_MIN_PULSE_US < 1000 || TILT_MAX_PULSE_US > 2000
            || TILT_MIN_PULSE_US > 1500 || TILT_MAX_PULSE_US < 1500) begin
            $display("FAIL: invalid camera servo timing parameters");
            $finish;
        end
    end
    // synthesis translate_on
endmodule
