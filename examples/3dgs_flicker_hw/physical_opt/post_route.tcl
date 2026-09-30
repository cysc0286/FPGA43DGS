# Isolated physical optimization experiment; never changes the source project,
# constraints, firmware, or live board clock. Vivado 2018.3 / 30TAI proxy netlist.
if {$argc < 2 || $argc > 3} {error "Usage: post_route.tcl routed.dcp new_output_directory ?extra_setup_margin_ns?"}
set source [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
set margin 0.0
if {$argc == 3} {set margin [lindex $argv 2]}
if {[file exists $out]} {error "Output already exists; preserve prior evidence"}
file mkdir $out
set f [open [file join $out provenance.txt] w]
puts $f "source=$source\nversion=[version -short]\nexperimental_setup_margin_ns=$margin\nboard_programmed=false"
close $f
set_param general.maxThreads 4
proc snapshot {folder} {
    file mkdir $folder
    report_timing_summary -delay_type min_max -report_unconstrained -file [file join $folder timing.rpt]
    report_route_status -file [file join $folder route.rpt]
    report_utilization -file [file join $folder utilization.rpt]
    report_clock_interaction -file [file join $folder clock_interaction.rpt]
    set pins [get_pins -hier -filter {NAME =~ U_adder_top/U_flicker/* && DIRECTION == OUT}]
    foreach delay {max min} {
        report_timing -delay_type $delay -through $pins -max_paths 20 -file [file join $folder render_${delay}.rpt]
    }
    # The 200 MHz clock is shared with vendor DDR logic. Report the entire
    # domain, not just the renderer, before discussing a clock increase.
    report_timing -from [get_clocks clk_pll_i] -to [get_clocks clk_pll_i] -max_paths 20 -file [file join $folder domain_200mhz.rpt]
}
open_checkpoint $source
write_xdc [file join $out original_constraints.xdc]
snapshot [file join $out before]
if {$margin > 0} {
    set f [open [file join $out original_constraints.xdc] r]
    set original_xdc [read $f];close $f
    if {[regexp {set_clock_uncertainty} $original_xdc]} {
        error "Existing user clock uncertainty: review before tightening or resetting"
    }
    # Exploration only: stronger setup margin on the complete shared DDR domain.
    # This is not a PLL change and is never packaged as board firmware.
    set_clock_uncertainty -setup $margin [get_clocks clk_pll_i]
    snapshot [file join $out tightened_before]
}
set started [clock seconds]
if {[catch {phys_opt_design -directive AggressiveExplore} message options]} {
    set f [open [file join $out failed.txt] w]; puts $f $message; close $f
    return -options $options $message
}
snapshot [file join $out after]
if {$margin > 0} {
    # Record native-budget timing separately. The supplied source XDC has no
    # explicit setup uncertainty on this clock (verified before this experiment).
    set_clock_uncertainty -setup 0.0 [get_clocks clk_pll_i]
    snapshot [file join $out restored_budget]
}
write_checkpoint [file join $out physical_opt.dcp]
set f [open [file join $out complete.txt] w]
puts $f "physical_optimization_seconds=[expr {[clock seconds]-$started}]\nNo bitstream generated or installed."
close $f
close_design
exit
