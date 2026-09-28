set config [current_wave_config]
if {[string length $config] == 0} {create_wave_config}
foreach name {clk rst_n opcode operand_a operand_b request_seq status completed_seq result} {
    add_wave /tb_alu_walkthrough/$name
}
foreach name {state iteration accumulator multiplicand multiplier negative_product} {
    add_wave /tb_alu_walkthrough/dut/$name
}
log_wave -r /tb_alu_walkthrough/*
run all
save_wave_config walkthrough.wcfg
puts "WAVEFORM_READY=PASS"
