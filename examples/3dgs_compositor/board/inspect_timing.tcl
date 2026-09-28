# Additional bounded inspection of the already routed design; does not rebuild.
set here [file dirname [file normalize [info script]]]
set repo [file normalize [file join $here .. .. ..]]
if {$argc != 1} {error "Pass the output report directory as the only Tcl argument"}
set output [file normalize [lindex $argv 0]]
file mkdir $output
open_checkpoint [file join $repo platform gs_compositor_fpga fpai_demo_vivado.runs impl_1 ai7030_edif_top_routed.dcp]
report_timing -delay_type max -slack_lesser_than 0 -max_paths 20 -file [file join $output setup_violations.rpt]
set registers [get_cells -hier -filter {NAME =~ *U_gs_compositor* && IS_SEQUENTIAL}]
if {[llength $registers] == 0} {error "No compositor registers"}
report_timing -to $registers -delay_type min_max -max_paths 10 -file [file join $output compositor_input_timing.rpt]
report_timing -from $registers -delay_type min_max -max_paths 10 -file [file join $output compositor_output_timing.rpt]
report_bus_skew -file [file join $output bus_skew.rpt]
close_design
exit 0
