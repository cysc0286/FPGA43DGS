`timescale 1ns/1ps
// Single-clock register-command ALU. Software commits operands with request_seq.
// Only one command may be outstanding. Sequence zero is reserved after reset.
module basic_alu (
    input wire clk,
    input wire rst_n,
    input wire [31:0] operand_a,
    input wire [31:0] operand_b,
    input wire [31:0] opcode,
    input wire [31:0] request_seq,
    output reg [63:0] result,
    output reg [31:0] completed_seq,
    output wire [31:0] status
);
    localparam IDLE=2'd0, SIMPLE=2'd1, MULTIPLY=2'd2, SIGN_RESULT=2'd3;
    reg [1:0] state;
    reg error;
    reg [31:0] a, b, operation, active_seq;
    reg [63:0] accumulator, multiplicand;
    reg [31:0] multiplier;
    reg [5:0] iteration;
    reg negative_product;
    wire [31:0] magnitude_a = operand_a[31] ? (~operand_a + 32'd1) : operand_a;
    wire [31:0] magnitude_b = operand_b[31] ? (~operand_b + 32'd1) : operand_b;
    wire [63:0] next_product = accumulator + (multiplier[0] ? multiplicand : 64'd0);
    assign status = {30'd0, error, (state != IDLE)};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            error <= 1'b0;
            result <= 64'd0;
            completed_seq <= 32'd0;
            active_seq <= 32'd0;
            a <= 32'd0;
            b <= 32'd0;
            operation <= 32'd0;
            accumulator <= 64'd0;
            multiplicand <= 64'd0;
            multiplier <= 32'd0;
            iteration <= 6'd0;
            negative_product <= 1'b0;
        end else begin
            case (state)
                IDLE: if (request_seq != completed_seq) begin
                    a <= operand_a;
                    b <= operand_b;
                    operation <= opcode;
                    active_seq <= request_seq;
                    error <= 1'b0;
                    if ((opcode == 32'd2) || (opcode == 32'd3)) begin
                        accumulator <= 64'd0;
                        multiplicand <= {32'd0, (opcode == 32'd3 ? magnitude_a : operand_a)};
                        multiplier <= opcode == 32'd3 ? magnitude_b : operand_b;
                        negative_product <= (opcode == 32'd3) && (operand_a[31] ^ operand_b[31]);
                        iteration <= 6'd0;
                        state <= MULTIPLY;
                    end else state <= SIMPLE;
                end
                SIMPLE: begin
                    // Except products, results contain a 32-bit bit pattern in
                    // the low word; the high word is always zero.
                    case (operation)
                        32'd0: result <= {32'd0, (a + b)};
                        32'd1: result <= {32'd0, (a - b)};
                        32'd4: result <= {32'd0, (a & b)};
                        32'd5: result <= {32'd0, (a | b)};
                        32'd6: result <= {32'd0, (a ^ b)};
                        32'd7: result <= {32'd0, (a << b[4:0])};
                        32'd8: result <= {32'd0, (a >> b[4:0])};
                        32'd9: result <= {32'd0, ($signed(a) >>> b[4:0])};
                        32'd10: result <= {32'd0, ($signed(a) < $signed(b) ? a : b)};
                        32'd11: result <= {32'd0, ($signed(a) > $signed(b) ? a : b)};
                        default: begin result <= 64'd0; error <= 1'b1; end
                    endcase
                    completed_seq <= active_seq;
                    state <= IDLE;
                end
                MULTIPLY: begin
                    accumulator <= next_product;
                    multiplicand <= multiplicand << 1;
                    multiplier <= multiplier >> 1;
                    if (iteration == 6'd31) begin
                        if (negative_product) state <= SIGN_RESULT;
                        else begin
                            result <= next_product;
                            completed_seq <= active_seq;
                            state <= IDLE;
                        end
                    end else iteration <= iteration + 1'b1;
                end
                SIGN_RESULT: begin
                    result <= ~accumulator + 64'd1;
                    completed_seq <= active_seq;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
