# Open the implemented board project and named timing reports.
# No synthesis, programming, or RTL edits are performed by this script.
set here [file dirname [file normalize [info script]]]
set repo [file normalize [file join $here .. .. ..]]
set project [file join $repo platform basic_alu_fpga]
set status [open [file join $here .. build review_open_status.txt] w]
proc review_note {text} {global status; puts $text; puts $status $text; flush $status}
if {[catch {
    open_project [file join $project fpai_demo_vivado.xpr]
    review_note "PROJECT_OPENED=$project"
    open_run impl_1
    review_note "IMPLEMENTED_DESIGN_OPENED=impl_1"
    report_timing_summary -name Board_Timing_Summary
    set cells [get_cells -hier -filter {NAME =~ *U_basic_alu*}]
    report_timing -through [get_pins -of_objects $cells -filter {DIRECTION == OUT}] -delay_type max -max_paths 10 -name ALU_Setup_100MHz
    set registers [get_cells -hier -filter {NAME =~ *U_basic_alu* && IS_SEQUENTIAL}]
    report_timing -to $registers -delay_type min -max_paths 10 -name ALU_Hold_100MHz
    report_timing -delay_type max -slack_lesser_than 0 -max_paths 20 -name Board_Setup_Violations
    review_note "GUI_REVIEW_READY=PASS"
} message options]} {
    review_note "GUI_REVIEW_ERROR=$message"
    if {[dict exists $options -errorinfo]} {review_note [dict get $options -errorinfo]}
}
close $status
