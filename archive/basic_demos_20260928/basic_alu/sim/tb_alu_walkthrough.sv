`timescale 1ns/1ps
// A compact waveform showing latency from the ALU acceptance edge E0.
module tb_alu_walkthrough;
    reg clk=0;
    always #5 clk=~clk;
    reg rst_n=0;
    reg [31:0] operand_a=0, operand_b=0, opcode=0, request_seq=0;
    wire [63:0] result;
    wire [31:0] completed_seq, status;
    basic_alu dut(.*);
    integer measured_cycles;
    task automatic demonstrate(input [31:0] op, a, b, seq,
                                input [63:0] expected, input integer expected_cycles);
        begin
            @(negedge clk);
            operand_a=a; operand_b=b; opcode=op;
            repeat(2) @(negedge clk);
            request_seq=seq;
            @(posedge clk); #1;
            if (!status[0]) $fatal(1,"FAIL: request not accepted");
            measured_cycles=0;
            while (completed_seq !== seq && measured_cycles < 40) begin
                @(posedge clk); #1; measured_cycles=measured_cycles+1;
            end
            if (result !== expected || status !== 0 || measured_cycles != expected_cycles)
                $fatal(1,"FAIL: op=%0d result=%h cycles=%0d",op,result,measured_cycles);
            $display("DEMO op=%0d a=%h b=%h result=%h acceptance_to_done_cycles=%0d",op,a,b,result,measured_cycles);
            repeat(5) @(negedge clk);
        end
    endtask
    initial begin
        repeat(3) @(negedge clk); rst_n=1;
        demonstrate(0,17,5,1,64'd22,1);
        demonstrate(1,17,5,2,64'd12,1);
        demonstrate(2,7,9,3,64'd63,32);
        demonstrate(3,32'hfffffff9,9,4,64'hffffffffffffffc1,33);
        demonstrate(9,32'hfffffff0,2,5,64'h00000000fffffffc,1);
        $display("PASS: ALU waveform walkthrough, all five results and cycle counts verified");
        $finish;
    end
    initial begin #5000; $fatal(1,"FAIL: walkthrough watchdog"); end
endmodule
