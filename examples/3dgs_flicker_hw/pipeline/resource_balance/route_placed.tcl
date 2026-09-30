# Route an independently placed candidate without changing its source platform.
# Uses the production route directive and restores only existing vendor checks.
if {$argc != 3} {error "Usage: route_placed.tcl placed.dcp new_output_directory platform"}
set source [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
set ::env(JFM_PATH) [file normalize [lindex $argv 2]]
if {![file exists $source] || [file exists $out]} {error "Missing source or output already exists"}
file mkdir $out
set f [open [file join $out provenance.txt] w]
puts $f "source=$source\nplatform=$::env(JFM_PATH)\nversion=[version -short]\nroute_directive=NoTimingRelaxation\nboard_programmed=false"
close $f
set_param general.maxThreads 2
open_checkpoint $source
if {[get_property PART [current_project]] ne "xc7z030ffg676-2"} {error "Wrong device"}
if {[get_property PERIOD [get_clocks clk_pll_i]] != 5.0} {error "Shared 200 MHz clock changed"}
source [file join [file dirname [info script]] .. .. physical_opt restore_vendor_checks.tcl]
phys_opt_design -directive AggressiveExplore
write_checkpoint [file join $out before_route.dcp]
route_design -directive NoTimingRelaxation
write_checkpoint [file join $out routed.dcp]
report_route_status -file [file join $out route.rpt]
report_timing_summary -delay_type min_max -file [file join $out timing.rpt]
report_utilization -file [file join $out utilization.rpt]
set pins [get_pins -hier -filter {NAME =~ U_adder_top/U_flicker/* && DIRECTION == OUT}]
foreach delay {max min} {
    report_timing -delay_type $delay -through $pins -max_paths 20 -file [file join $out render_${delay}.rpt]
}
set f [open [file join $out complete.txt] w]
puts $f "Routing completed; inspect reports before packaging. No bitstream or board changes."
close $f
close_design
exit
