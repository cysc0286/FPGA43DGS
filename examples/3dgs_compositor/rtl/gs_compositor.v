`timescale 1ns/1ps
// One shared pipelined 32x32 multiplier. All state changes commit atomically.
module gs_compositor (
    input wire clk, rst_n,
    input wire [31:0] alpha, red, green, blue, ordinal, opcode, request_seq,
    output reg [31:0] out_r, out_g, out_b, out_t, last,
    output reg [31:0] completed_seq, last_cycles, pixel_cycles,
    output wire [31:0] status
);
    localparam [31:0] ONE=32'd1073741824, CLAMP=32'd1063004406,
        MIN_ALPHA=32'd4210753, MIN_T=32'd107375, MAX_RGB=32'd268435456;
    localparam IDLE=0, CHECK=1, MUL=2, ROUND=3, APPLY=4, COMMIT=5;
    reg [2:0] state;
    reg busy, error, terminated, finished;
    reg [31:0] a, r, g, b, ord, op, seq, seen;
    reg [31:0] mul_a, mul_b, rounded, weight, next_t;
    reg [31:0] add_r, add_g, add_b, cycles;
    reg [63:0] product;
    reg [2:0] which;
    reg do_blend, do_bg;
    assign status={28'd0,finished,terminated,error,busy};
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state<=IDLE; busy<=0; error<=0; terminated<=0; finished<=0;
            out_r<=0; out_g<=0; out_b<=0; out_t<=ONE; last<=0; seen<=0;
            completed_seq<=0; last_cycles<=0; pixel_cycles<=0; cycles<=0;
            a<=0;r<=0;g<=0;b<=0;ord<=0;op<=0;seq<=0;
            mul_a<=0;mul_b<=0;product<=0;rounded<=0;weight<=0;next_t<=0;
            add_r<=0;add_g<=0;add_b<=0;which<=0;do_blend<=0;do_bg<=0;
        end else begin
            if (busy) cycles<=cycles+1'b1;
            case (state)
            IDLE: if (request_seq != completed_seq) begin
                a<=alpha; r<=red; g<=green; b<=blue; ord<=ordinal;
                op<=opcode;seq<=request_seq;busy<=1;error<=0;cycles<=1;
                do_blend<=0;do_bg<=0;state<=CHECK;
            end
            CHECK: begin
                if (op==0) begin
                    out_r<=0;out_g<=0;out_b<=0;out_t<=ONE;last<=0;seen<=0;
                    terminated<=0;finished<=0;pixel_cycles<=0;state<=COMMIT;
                end else if ((op!=1 && op!=2) || finished ||
                             r>MAX_RGB || g>MAX_RGB || b>MAX_RGB ||
                             (op==1 && (a>ONE || ord==0 || ord<=seen))) begin
                    error<=1;state<=COMMIT;
                end else if (op==1) begin
                    seen<=ord;
                    if (terminated || a<MIN_ALPHA) state<=COMMIT;
                    else begin
                        mul_a<=out_t;mul_b<=(a>CLAMP)?CLAMP:a;
                        which<=0;state<=MUL;
                    end
                end else begin
                    do_bg<=1;weight<=out_t;next_t<=out_t;
                    mul_a<=r;mul_b<=out_t;which<=1;state<=MUL;
                end
            end
            MUL: begin product<=mul_a*mul_b;state<=ROUND;end
            ROUND: begin rounded<=(product+64'd536870912)>>30;state<=APPLY;end
            APPLY: begin
                case (which)
                0: begin
                    weight<=rounded;next_t<=out_t-rounded;
                    if (out_t-rounded<MIN_T) begin terminated<=1;state<=COMMIT;end
                    else begin
                        do_blend<=1;mul_a<=r;mul_b<=rounded;which<=1;state<=MUL;
                    end
                end
                1: begin add_r<=rounded;mul_a<=g;mul_b<=weight;which<=2;state<=MUL;end
                2: begin add_g<=rounded;mul_a<=b;mul_b<=weight;which<=3;state<=MUL;end
                3: begin add_b<=rounded;state<=COMMIT;end
                default: begin error<=1;state<=COMMIT;end
                endcase
            end
            COMMIT: begin
                if (do_blend || do_bg) begin
                    out_r<=out_r+add_r;out_g<=out_g+add_g;out_b<=out_b+add_b;
                    out_t<=next_t;
                    if (do_blend) last<=ord;
                    if (do_bg) finished<=1;
                end
                last_cycles<=cycles+1'b1;pixel_cycles<=pixel_cycles+cycles+1'b1;
                completed_seq<=seq;busy<=0;state<=IDLE;
            end
            default: state<=IDLE;
            endcase
        end
    end
endmodule
