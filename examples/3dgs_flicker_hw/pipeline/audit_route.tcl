# Additional read-only route/CDC-bus checks, separate from setup/hold audit.
set report_dir [file normalize $::env(FLK_REPORT_DIR)]
set f [open [file join $report_dir status.txt] r]
set status [read $f];close $f
if {![regexp -line {^PROJECT=([^\r\n]+)} $status all project_dir]} {error "Missing platform provenance"}
open_checkpoint [file join $project_dir fpai_demo_vivado.runs impl_1 ai7030_edif_top_routed.dcp]
report_bus_skew -file [file join $report_dir bus_skew.rpt]
report_route_status -file [file join $report_dir route_status.rpt]
report_utilization -file [file join $report_dir utilization_flat.rpt]
close_design
exit
