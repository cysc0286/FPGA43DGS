# Independent full-placement control for a changed netlist whose incremental
# placement is expensive. Read the post-opt checkpoint BEFORE incremental reuse.
# No project properties, source checkpoint or board state are modified.
if {$argc != 3} {error "Usage: full_place.tcl post_opt.dcp new_output_directory original_vendor_opt_pre.tcl"}
set source [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
if {[file exists $out]} {error "Preserve previous placement evidence"}
file mkdir $out
set f [open [file join $out provenance.txt] w]
puts $f "source=$source\nversion=[version -short]\nincremental=false\nboard_programmed=false"
close $f
set_param general.maxThreads 4
open_checkpoint $source
# Session DRC settings are not restored by open_checkpoint. Replay only the
# four pre-existing vendor proxy-device checks from the supplied original hook;
# reject an absent declaration rather than silently suppressing a new check.
set vendor [file normalize [lindex $argv 2]]
set f [open $vendor r]; set hook [read $f]; close $f
foreach rule {REQP-44 REQP-46 REQP-52 REQP-56} {
    set pattern [format {set_property IS_ENABLED 0 \[get_drc_checks +%s\]} $rule]
    if {![regexp $pattern $hook]} {error "Missing inherited vendor check setting: $rule"}
    set_property IS_ENABLED false [get_drc_checks $rule]
}
set f [open [file join $out inherited_vendor_checks.txt] w]
puts $f "hook=$vendor\nREQP-44 REQP-46 REQP-52 REQP-56: original proxy-device exclusions restored; no new waiver."
close $f
set started [clock seconds]
place_design -directive ExtraNetDelay_high
write_checkpoint [file join $out full_placed.dcp]
report_utilization -file [file join $out utilization.rpt]
report_timing_summary -delay_type min_max -file [file join $out timing.rpt]
set f [open [file join $out complete.txt] w]
puts $f "placement_seconds=[expr {[clock seconds]-$started}]\nNo routing or bitstream generated."
close $f
close_design
exit
