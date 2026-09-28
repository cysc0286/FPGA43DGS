`timescale 1ns/1ps
// Exercise the actual patched vendor top and address decoder, not a mock bus.
module tb_register_alu;
    reg clk=0, ra_clk=0, rst_n=0, ra_rst_n=0;
    always #4 clk=~clk;
    always #5 ra_clk=~ra_clk;
    reg [31:0] itf_ra_awaddr=0, itf_ra_awdata=0, itf_ra_araddr=0;
    reg itf_ra_awvalid=0, itf_ra_arvalid=0, itf_ra_rready=0;
    wire itf_ra_awready, itf_ra_arready, itf_ra_rvalid;
    wire [31:0] itf_ra_rdata;
    wire [31:0] itf_awaddr, itf_araddr;
    wire [63:0] itf_awdata;
    wire itf_awvalid, itf_arvalid, itf_rready, reset_reg;
    reg itf_awready=1, itf_arready=1, itf_rvalid=0;
    reg [63:0] itf_rdata=0;
    adder_top dut(.*);

    task automatic write_reg(input [31:0] address, input [31:0] value);
        integer wait_cycles;
        begin
            @(negedge ra_clk);
            itf_ra_awaddr=address; itf_ra_awdata=value; itf_ra_awvalid=1;
            wait_cycles=0;
            while (itf_ra_awready !== 1 && wait_cycles < 100) begin
                @(negedge ra_clk); wait_cycles=wait_cycles+1;
            end
            if (itf_ra_awready !== 1) $fatal(1,"FAIL: write handshake timeout");
            @(negedge ra_clk); itf_ra_awvalid=0;
        end
    endtask

    task automatic read_reg(input [31:0] address, output [31:0] value);
        integer wait_cycles;
        begin
            @(negedge ra_clk);
            itf_ra_araddr=address; itf_ra_arvalid=1;
            wait_cycles=0;
            while (itf_ra_arready !== 1 && wait_cycles < 100) begin
                @(negedge ra_clk); wait_cycles=wait_cycles+1;
            end
            if (itf_ra_arready !== 1) $fatal(1,"FAIL: read address timeout");
            @(negedge ra_clk); itf_ra_arvalid=0;
            wait_cycles=0;
            while (itf_ra_rvalid !== 1 && wait_cycles < 100) begin
                @(negedge ra_clk); wait_cycles=wait_cycles+1;
            end
            if (itf_ra_rvalid !== 1) $fatal(1,"FAIL: read response timeout");
            value=itf_ra_rdata;
            // Apply read-response backpressure and check stable valid/data.
            repeat(2) begin
                @(negedge ra_clk);
                if (itf_ra_rvalid !== 1 || itf_ra_rdata !== value)
                    $fatal(1,"FAIL: unstable stalled read response");
            end
            itf_ra_rready=1;
            @(negedge ra_clk); itf_ra_rready=0;
        end
    endtask

    integer fd, scanned, count=0, polls;
    reg [31:0] op, a, b, sequence_id=0, value, low_word, high_word, status;
    reg [63:0] expected;
    reg error;
    initial begin #10000000; $fatal(1,"FAIL: integration watchdog"); end
    initial begin
        repeat(5) @(negedge ra_clk);
        rst_n=1; ra_rst_n=1;
        repeat(5) @(negedge ra_clk);
        read_reg('h94,value);
        if (value !== 'h414c5531) $fatal(1,"FAIL: ALU signature wiring");
        read_reg('hc0,value);
        if (value !== 'h20230628) $fatal(1,"FAIL: original adder version changed");
        read_reg('h8c,value);
        if (value !== 0) $fatal(1,"FAIL: completion reset");
        fd=$fopen("vectors.txt","r");
        if (!fd) $fatal(1,"FAIL: cannot open vectors");
        while (!$feof(fd)) begin
            scanned=$fscanf(fd,"%h %h %h %h %h\n",op,a,b,expected,error);
            if (scanned != 5) $fatal(1,"FAIL: malformed vector");
            write_reg('h0c,a); write_reg('h10,b); write_reg('h14,op);
            read_reg('h8c,value);
            if (value !== sequence_id) $fatal(1,"FAIL: unexpected request before commit");
            sequence_id=sequence_id+1;
            write_reg('h20,sequence_id);
            read_reg('h8c,value); polls=0;
            while (value !== sequence_id && polls < 20) begin
                read_reg('h8c,value); polls=polls+1;
            end
            if (value !== sequence_id) $fatal(1,"FAIL: completion timeout at %0d",count);
            read_reg('h84,low_word); read_reg('h88,high_word); read_reg('h90,status);
            if ({high_word,low_word} !== expected || status !== {30'd0,error,1'b0})
                $fatal(1,"FAIL: vector=%0d op=%h a=%h b=%h got=%h expected=%h status=%h",
                       count,op,a,b,{high_word,low_word},expected,status);
            if (itf_arvalid !== 0 || itf_awvalid !== 0)
                $fatal(1,"FAIL: ALU command started unrelated DMA");
            count=count+1;
        end
        $fclose(fd);
        if (count != 2800) $fatal(1,"FAIL: incomplete corpus");
        $display("PASS: register_alu (2800 golden vectors through actual vendor top, read backpressure, no DMA trigger)");
        $finish;
    end
endmodule
