# Read-only physical resource audit for the FMQL 30TAI proxy design.
# Run with Vivado 2018.3: inspect.tcl CHECKPOINT NEW_REPORT_DIRECTORY.
# No source, placement constraint, clock or checkpoint is modified.
if {$argc != 2} {error "Usage: inspect.tcl checkpoint.dcp new_report_directory"}
set source [file normalize [lindex $argv 0]]
set out [file normalize [lindex $argv 1]]
if {![file exists $source] || [file exists $out]} {error "Missing source or report directory already exists"}
file mkdir $out
set_param general.maxThreads 2
open_checkpoint $source
if {[get_property PART [current_design]] ne "xc7z030ffg676-2"} {error "Unexpected proxy device"}
report_utilization -file [file join $out utilization.rpt]
report_utilization -hierarchical -file [file join $out hierarchy.rpt]
report_control_sets -verbose -file [file join $out control_sets.rpt]
report_timing_summary -delay_type min_max -file [file join $out timing.rpt]
set f [open [file join $out regions.txt] w]
puts $f "source=$source\nversion=[version -short]\nread_only=true"
foreach block [get_pblocks -quiet] {
    puts $f "PBLOCK=$block"
    foreach prop {GRID_RANGES EXCLUDE_PLACEMENT CONTAIN_ROUTING IS_SOFT} {
        if {[lsearch -exact [list_property $block] $prop] >= 0} {
            puts $f "$prop=[get_property $prop $block]"
        }
    }
}
puts $f "LOC_FIXED=[llength [get_cells -hier -filter {IS_PRIMITIVE == 1 && IS_LOC_FIXED == 1}]]"
puts $f "CLOCKS=[get_clocks]"
close $f
set f [open [file join $out complete.txt] w]
puts $f "read_only=true\ncheckpoint=$source\nNo bitstream or board changes."
close $f
close_design
exit
