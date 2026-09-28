# Read-only checkpoint view for timing inspection alongside the project window.
set here [file dirname [file normalize [info script]]]
set repo [file normalize [file join $here .. .. ..]]
set output [file join $here .. build timing_view]
file mkdir $output
set status [open [file join $output status.txt] w]
if {[catch {
    open_checkpoint [file join $repo platform basic_alu_fpga fpai_demo_vivado.runs impl_1 ai7030_edif_top_routed.dcp]
    puts $status "CHECKPOINT_OPENED=PASS"; flush $status
    report_timing_summary -name Board_Timing_Summary
    set cells [get_cells -hier -filter {NAME =~ *U_basic_alu*}]
    report_timing -through [get_pins -of_objects $cells -filter {DIRECTION == OUT}] -delay_type max -max_paths 10 -name ALU_Setup_100MHz -file [file join $output alu_setup.rpt]
    set registers [get_cells -hier -filter {NAME =~ *U_basic_alu* && IS_SEQUENTIAL}]
    report_timing -to $registers -delay_type min -max_paths 10 -name ALU_Hold_100MHz -file [file join $output alu_hold.rpt]
    report_timing -delay_type max -slack_lesser_than 0 -max_paths 20 -name Board_Setup_Violations
    select_objects [get_cells U_adder_top/U_basic_alu]
    puts $status "TIMING_GUI_READY=PASS"
} message options]} {
    puts $status "TIMING_GUI_ERROR=$message"
    if {[dict exists $options -errorinfo]} {puts $status [dict get $options -errorinfo]}
}
close $status
