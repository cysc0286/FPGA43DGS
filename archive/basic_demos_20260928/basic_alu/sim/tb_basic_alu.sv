`timescale 1ns/1ps
module tb_basic_alu;
    reg clk=0;
    always #5 clk=~clk;
    reg rst_n=0;
    reg [31:0] operand_a=0, operand_b=0, opcode=0, request_seq=0;
    wire [63:0] result;
    wire [31:0] completed_seq, status;
    basic_alu dut(.*);
    integer fd, scanned, count=0, cycles;
    reg [31:0] op, va, vb, sequence_id=0;
    reg [63:0] golden;
    reg err;
    reg [1023:0] vector_path;
    initial begin
        #10000000;
        $fatal(1, "FAIL: simulation watchdog expired");
    end
    initial begin
        if (!$value$plusargs("VECTORS=%s", vector_path)) vector_path="vectors.txt";
        fd=$fopen(vector_path,"r");
        if (!fd) $fatal(1,"FAIL: cannot open vectors");
        repeat(3) @(negedge clk);
        rst_n=1;
        @(negedge clk);
        if (status !== 0 || completed_seq !== 0 || result !== 0)
            $fatal(1,"FAIL: reset state");
        while (!$feof(fd)) begin
            scanned=$fscanf(fd,"%h %h %h %h %h\n",op,va,vb,golden,err);
            if (scanned != 5) $fatal(1,"FAIL: malformed vector %0d",count);
            // Operands alone must not trigger a request.
            operand_a=va; operand_b=vb; opcode=op;
            repeat(2) @(negedge clk);
            if (completed_seq !== sequence_id || status[0] !== 0)
                $fatal(1,"FAIL: request ran without sequence commit");
            sequence_id=sequence_id+1;
            request_seq=sequence_id;
            @(negedge clk);
            if (status[0] !== 1) $fatal(1,"FAIL: missing busy assertion");
            // After acceptance, inputs may change; the accepted command is latched.
            operand_a=~va; operand_b=~vb; opcode=32'hffffffff;
            cycles=0;
            while (completed_seq !== sequence_id && cycles < 45) begin
                @(negedge clk); cycles=cycles+1;
            end
            if (completed_seq !== sequence_id || status[0] !== 0)
                $fatal(1,"FAIL: completion timeout vector %0d",count);
            if (result !== golden || status[1] !== err)
                $fatal(1,"FAIL: vector=%0d op=%h a=%h b=%h got=%h expected=%h status=%h",
                       count,op,va,vb,result,golden,status);
            repeat(2) @(negedge clk);
            if (result !== golden || completed_seq !== sequence_id || status[0] !== 0)
                $fatal(1,"FAIL: duplicate request or unstable result");
            count=count+1;
        end
        $fclose(fd);
        if (count != 2800) $fatal(1,"FAIL: incomplete vector corpus: %0d",count);
        // Reset must abort an in-flight multiplication and clear the protocol.
        operand_a=32'hffffffff; operand_b=32'hffffffff; opcode=2;
        request_seq=sequence_id+1;
        repeat(5) @(negedge clk);
        rst_n=0; request_seq=0;
        @(negedge clk);
        if (status !== 0 || completed_seq !== 0 || result !== 0)
            $fatal(1,"FAIL: reset during multiplication");
        rst_n=1;
        repeat(3) @(negedge clk);
        operand_a=17; operand_b=5; opcode=1; request_seq=1;
        repeat(4) @(negedge clk);
        if (result !== 12 || completed_seq !== 1 || status !== 0)
            $fatal(1,"FAIL: operation after reset");
        $display("PASS: basic_alu (%0d golden vectors, protocol and reset checks)",count);
        $finish;
    end
endmodule
