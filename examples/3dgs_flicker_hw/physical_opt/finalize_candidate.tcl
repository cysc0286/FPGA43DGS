# Package an independently optimized routed checkpoint through the original
# vendor bitstream hooks. Preserve the original implementation before calling.
if {$argc != 4} {error "Usage: finalize_candidate.tcl PLATFORM OPTIMIZED_DCP NEW_REPORT ORIGINAL_FROZEN_MANIFEST"}
set platform [file normalize [lindex $argv 0]]
set candidate [file normalize [lindex $argv 1]]
set report_dir [file normalize [lindex $argv 2]]
set original_manifest [file normalize [lindex $argv 3]]
set helper [file dirname [file normalize [info script]]]
if {![file exists $candidate] || ![file exists $original_manifest]} {error "Missing candidate or preserved original"}
if {[file exists $report_dir]} {error "Report exists; do not overwrite evidence"}
file mkdir $report_dir
set status [open [file join $report_dir status.txt] w]
proc note {line} {global status; puts $line; puts $status $line; flush $status}
note "PROJECT=$platform"
note "REPORT_DIR=$report_dir"
note "POST_ROUTE_CANDIDATE=$candidate"
note "ORIGINAL_FROZEN_MANIFEST=$original_manifest"
set ::env(JFM_PATH) $platform
set_param general.maxThreads 4
if {[catch {
    open_checkpoint $candidate
    source [file join $helper restore_vendor_checks.tcl]
    if {[get_property PART [current_project]] ne "xc7z030ffg676-2"} {error "Unexpected part"}
    if {[get_property PERIOD [get_clocks clk_pll_i]] != 5.0} {error "Shared clock changed"}
    foreach delay {max min} {
        set path [get_timing_paths -delay_type $delay -max_paths 1]
        if {[llength $path] != 1 || [get_property SLACK $path] < 0} {error "Unresolved $delay timing"}
        note "NATIVE_${delay}_SLACK_NS=[get_property SLACK $path]"
    }
    foreach {port pin} {FMC_HPC_GBTCLK0_M2C_C_P R6 FMC_HPC_GBTCLK0_M2C_C_N R5} {
        if {[get_property PACKAGE_PIN [get_ports $port]] ne $pin} {error "Lite pin mismatch"}
    }
    report_route_status -file [file join $report_dir route_status.rpt]
    report_timing_summary -file [file join $report_dir timing_summary.rpt]
    report_utilization -hierarchical -file [file join $report_dir utilization.rpt]
    report_drc -file [file join $report_dir drc.rpt]
    report_cdc -file [file join $report_dir cdc.rpt]
    cd [file join $platform fpai_demo_vivado.runs impl_1]
    # The preserved original manifest covers all replaced implementation files.
    write_checkpoint -force ai7030_edif_top_routed.dcp
    source [file join $platform ip_patch process_control write_bitstream_pre.tcl]
    write_bitstream -force ai7030_edif_top.bit
    source [file join $platform ip_patch process_control write_bitstream_post.tcl]
    if {![file exists ai7030_edif_top_disable_icap.bit]} {error "Missing vendor-patched bitstream"}
    note "BITSTREAM_GENERATION=PASS"
    set f [open [file join $report_dir timing_summary.rpt] r]; set timing [read $f]; close $f
    if {[string first "All user specified timing constraints are met." $timing] >= 0} {
        note "TIMING_ACCEPTANCE=PASS_UNDER_CURRENT_CONSTRAINTS"
    } else {
        note "TIMING_ACCEPTANCE=FAIL_REQUIRES_REPORT_REVIEW"
    }
} message options]} {
    note "BUILD_FAILED=$message"
    close $status
    exit 1
}
close $status
close_design
exit 0
