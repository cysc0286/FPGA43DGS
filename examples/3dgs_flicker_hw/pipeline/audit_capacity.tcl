if {$argc != 2} {error "Usage: audit_capacity.tcl <optimized checkpoint> <new evidence directory>"}
set evidence [file normalize [lindex $argv 1]]
file mkdir $evidence
open_checkpoint [file normalize [lindex $argv 0]]
report_utilization -hierarchical -file [file join $evidence utilization_hierarchical.rpt]
report_control_sets -verbose -file [file join $evidence control_sets.rpt]
close_design
exit
