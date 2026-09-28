set script_dir [file dirname [file normalize [info script]]]
set repo [file normalize [file join $script_dir .. .. ..]]
set project_dir [file join $repo platform flicker_cat_fpga]
if {[info exists ::env(FLK_PLATFORM_NAME)]} {set project_dir [file join $repo platform $::env(FLK_PLATFORM_NAME)]}
set stamp [clock format [clock seconds] -format %Y%m%d_%H%M%S]
set report_dir [file join $script_dir .. build board_cat_$stamp]
file mkdir $report_dir
set report [open [file join $report_dir status.txt] w]
proc note {s} {global report; puts $s; puts $report $s; flush $report}
if {[catch {
 open_project [file join $project_dir fpai_demo_vivado.xpr]
 if {[get_property PART [current_project]] ne "xc7z030ffg676-2"} {error "Wrong proxy part"}
 set ::env(JFM_PATH) $project_dir
 add_files -norecurse [file join $project_dir rtl adder_op legacy_adder_top.v]
 foreach pattern {*.v *.sv *.dat} {
  foreach f [glob -directory [file join $project_dir rtl flicker] $pattern] {
   if {[llength [get_files -quiet $f]]==0} {add_files -norecurse $f}
  }
 }
 foreach t [glob -directory [file join $project_dir rtl flicker] *_ip.tcl] {
  set ip [string range [file tail $t] 0 end-7]
  if {[llength [get_ips -quiet $ip]]==0} {source $t}
 }
 update_compile_order -fileset sources_1
 note "PROJECT=$project_dir"; note "REPORT_DIR=$report_dir"
 reset_run synth_1
 launch_runs impl_1 -to_step write_bitstream -jobs 2
 wait_on_run impl_1
 foreach r {synth_1 impl_1} {note "RUN=$r STATUS=[get_property STATUS [get_runs $r]] PROGRESS=[get_property PROGRESS [get_runs $r]]"}
 if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Implementation incomplete"}
 open_run impl_1
 foreach {port pin} {FMC_HPC_GBTCLK0_M2C_C_P R6 FMC_HPC_GBTCLK0_M2C_C_N R5} {
  if {[get_property PACKAGE_PIN [get_ports $port]] ne $pin} {error "Lite pin mismatch"}
 }
 set cells [get_cells -hier -filter {NAME =~ *U_flicker*}]
 if {[llength $cells]==0} {error "Renderer absent"}
 report_timing_summary -file [file join $report_dir timing_summary.rpt]
 report_timing -through [get_pins -of_objects $cells -filter {DIRECTION == OUT}] -max_paths 20 -file [file join $report_dir flicker_timing.rpt]
 report_utilization -hierarchical -file [file join $report_dir utilization.rpt]
 report_cdc -file [file join $report_dir cdc.rpt]
 report_drc -file [file join $report_dir drc.rpt]
 note "BITSTREAM_GENERATION=PASS"
 set f [open [file join $report_dir timing_summary.rpt] r];set timing [read $f];close $f
 if {[string first "All user specified timing constraints are met." $timing]>=0} {note "TIMING_ACCEPTANCE=PASS_UNDER_CURRENT_CONSTRAINTS"} else {note "TIMING_ACCEPTANCE=FAIL_REQUIRES_REPORT_REVIEW"}
 close_project
} err opts]} {
 note "BUILD_FAILED=$err"
 if {[dict exists $opts -errorinfo]} {note [dict get $opts -errorinfo]}
 catch {close_project};close $report;exit 1
}
close $report
exit 0
