`timescale 1ns/1ps
// Exercise the actual patched vendor top and address decoder, not a mock bus.
module tb_register_gs;
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

    integer fd,scanned,count=0,polls;
    reg [31:0] op,a,r,g,b,ord,er,eg,eb,et,el,es;
    reg [31:0] seq=0,value,ar,ag,ab,at,al,ast;
    initial begin #200000000; $fatal(1,"FAIL: integration watchdog");end
    initial begin
        repeat(5) @(negedge ra_clk);rst_n=1;ra_rst_n=1;
        repeat(5) @(negedge ra_clk);
        read_reg('h9c,value);if(value!=='h47534331)$fatal(1,"FAIL: GSC1 signature");
        read_reg('hc0,value);if(value!=='h20230628)$fatal(1,"FAIL: baseline version");
        fd=$fopen("vectors.txt","r");if(!fd)$fatal(1,"FAIL: vectors");
        while(!$feof(fd))begin
            scanned=$fscanf(fd,"%h %h %h %h %h %h %h %h %h %h %h %h\n",op,a,r,g,b,ord,er,eg,eb,et,el,es);
            if(scanned!=12)$fatal(1,"FAIL: malformed vector");
            write_reg('h0c,a);write_reg('h10,r);write_reg('h14,g);write_reg('h20,b);
            write_reg('h24,ord);write_reg('h28,op);
            read_reg('h94,value);if(value!==seq)$fatal(1,"FAIL: execution before commit");
            seq=seq+1;write_reg('h2c,seq);polls=0;
            read_reg('h94,value);
            while(value!==seq && polls<30)begin read_reg('h94,value);polls=polls+1;end
            if(value!==seq)$fatal(1,"FAIL: timeout");
            read_reg('h84,ar);read_reg('h88,ag);read_reg('h8c,ab);read_reg('h90,at);
            read_reg('ha0,al);read_reg('h98,ast);
            if({ar,ag,ab,at,al,ast}!=={er,eg,eb,et,el,es})$fatal(1,"FAIL: vector %0d",count);
            if(itf_arvalid!==0 || itf_awvalid!==0)$fatal(1,"FAIL: unrelated DMA triggered");
            count=count+1;
        end
        if(count!=84496)$fatal(1,"FAIL: count %0d",count);
        $display("PASS: register_gs 84496 exact vectors, actual vendor decoder, read backpressure, no DMA trigger");
        $finish;
    end
endmodule
