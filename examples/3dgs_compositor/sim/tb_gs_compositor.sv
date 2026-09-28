`timescale 1ns/1ps
module tb_gs_compositor;
    reg clk=0,rst_n=0;
    always #5 clk=~clk;
    reg [31:0] alpha=0,red=0,green=0,blue=0,ordinal=0,opcode=0,request_seq=0;
    wire [31:0] out_r,out_g,out_b,out_t,last,completed_seq,last_cycles,pixel_cycles,status;
    gs_compositor dut(.*);
    integer fd, scanned, count=0, elapsed, output_fd;
    reg [31:0] er,eg,eb,et,el,es,prev_done,prev_cycles;
    initial begin #100000000; $fatal(1,"FAIL: watchdog");end
    initial begin
        repeat(4) @(negedge clk);rst_n=1;
        @(negedge clk);
        if (completed_seq!==0 || out_t!==32'd1073741824 || status!==0)
            $fatal(1,"FAIL: reset");
        fd=$fopen("vectors.txt","r");output_fd=$fopen("actual.txt","w");
        if(!fd || !output_fd)$fatal(1,"FAIL: files");
        while (!$feof(fd)) begin
            scanned=$fscanf(fd,"%h %h %h %h %h %h %h %h %h %h %h %h\n",
                opcode,alpha,red,green,blue,ordinal,er,eg,eb,et,el,es);
            if(scanned!=12)$fatal(1,"FAIL: vector format");
            // Payload changes without commit must not start execution.
            prev_done=completed_seq;prev_cycles=pixel_cycles;
            repeat(2) @(negedge clk);
            if(completed_seq!==prev_done || status[0]!==0 || pixel_cycles!==prev_cycles)
                $fatal(1,"FAIL: uncommitted execution");
            request_seq=request_seq+1;elapsed=0;
            while (completed_seq!==request_seq && elapsed<30) begin
                @(negedge clk);elapsed=elapsed+1;
                // Once captured, mutation of operand staging must not alter transaction.
                if (elapsed==1) begin alpha=0;red=0;green=0;blue=0;ordinal=0;opcode=255;end
            end
            if(completed_seq!==request_seq)$fatal(1,"FAIL: command timeout");
            if({out_r,out_g,out_b,out_t,last,status}!=={er,eg,eb,et,el,es})
                $fatal(1,"FAIL: vector %0d got %h %h %h %h %h %h expected %h %h %h %h %h %h",
                    count,out_r,out_g,out_b,out_t,last,status,er,eg,eb,et,el,es);
            if(last_cycles!==elapsed)$fatal(1,"FAIL: cycle counter %0d %0d",last_cycles,elapsed);
            $fdisplay(output_fd,"%08h %08h %08h %08h %08h %08h %08h %08h",
                out_r,out_g,out_b,out_t,last,status,last_cycles,pixel_cycles);
            count=count+1;
        end
        if(count!=84496)$fatal(1,"FAIL: incomplete corpus %0d",count);
        $fclose(fd);$fclose(output_fd);
        // Reset cancels an in-flight multiply and permits a new sequence epoch.
        @(negedge clk);opcode=0;request_seq=request_seq+1;
        repeat(4) @(negedge clk);
        opcode=1;alpha=32'd536870912;red=32'd16777216;ordinal=1;request_seq=request_seq+1;
        repeat(4) @(negedge clk);
        rst_n=0;request_seq=0;
        repeat(3) @(negedge clk);rst_n=1;
        repeat(20) @(negedge clk);
        if(completed_seq!==0 || out_r!==0 || out_t!==32'd1073741824 || status!==0)
            $fatal(1,"FAIL: in-flight reset leaked output");
        $display("PASS: gs_compositor 84496 exact vectors, atomic capture, repeated sequence, in-flight reset, cycle counts");
        $finish;
    end
endmodule
