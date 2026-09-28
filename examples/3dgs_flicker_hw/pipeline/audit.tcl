# Read-only audit of the completed routed design; never changes constraints.
set here [file dirname [file normalize [info script]]]
set repo [file normalize [file join $here .. .. ..]]
set report_dir [file normalize $::env(FLK_REPORT_DIR)]
set status_file [open [file join $report_dir status.txt] r]
set status [read $status_file];close $status_file
if {![regexp -line {^PROJECT=([^\r\n]+)} $status all project_dir]} {error "Missing platform provenance"}
set checkpoint [file join $project_dir fpai_demo_vivado.runs impl_1 ai7030_edif_top_routed.dcp]
if {[info exists ::env(FLK_AUDIT_CHECKPOINT)]} {set checkpoint [file normalize $::env(FLK_AUDIT_CHECKPOINT)]}
open_checkpoint $checkpoint
report_timing_summary -file [file join $report_dir audited_timing_summary.rpt]
report_clocks -file [file join $report_dir clocks.rpt]
set report [open [file join $report_dir renderer_clock_and_paths.txt] w]
foreach unit {render dma regs} {
 set registers [get_cells -hier -filter "NAME =~ *U_flicker/$unit/* && IS_SEQUENTIAL == 1"]
 set pins [get_pins -of_objects $registers -filter {REF_PIN_NAME == C}]
 set clocks [get_clocks -of_objects $pins]
 puts $report "UNIT=$unit REGISTER_CLOCK_PINS=[llength $pins] CLOCKS=$clocks"
 if {[llength $clocks]==0} {error "No audited clock for $unit"}
 foreach clock $clocks {puts $report "CLOCK=$clock PERIOD_NS=[get_property PERIOD $clock]"}
}
set cells [get_cells -hier -filter {NAME =~ *U_flicker*}]
set pins [get_pins -of_objects $cells -filter {DIRECTION == OUT}]
foreach type {max min} {
 set paths [get_timing_paths -through $pins -delay_type $type -max_paths 20]
 puts $report "DELAY_TYPE=$type PATH_COUNT=[llength $paths]"
 foreach path $paths {puts $report "SLACK=[get_property SLACK $path] START=[get_property STARTPOINT_PIN $path] END=[get_property ENDPOINT_PIN $path]"}
 report_timing -through $pins -delay_type $type -max_paths 20 -file [file join $report_dir renderer_${type}_paths.rpt]
}
check_timing -verbose -file [file join $report_dir check_timing.rpt]
close $report
close_design
exit
