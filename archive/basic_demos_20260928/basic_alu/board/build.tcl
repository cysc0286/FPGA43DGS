set script_dir [file dirname [file normalize [info script]]]
set repo [file normalize [file join $script_dir .. .. ..]]
set project_dir [file join $repo platform basic_alu_fpga]
set stamp [clock format [clock seconds] -format %Y%m%d_%H%M%S]
set report_dir [file join $script_dir .. build board_$stamp]
file mkdir $report_dir
set report [open [file join $report_dir status.txt] w]
proc note {s} {global report; puts $s; puts $report $s; flush $report}
if {[catch {
    if {[version -short] ne "2018.3"} {error "Requires Vivado 2018.3"}
    open_project [file join $project_dir fpai_demo_vivado.xpr]
    if {[get_property PART [current_project]] ne "xc7z030ffg676-2"} {error "Wrong part"}
    # Keep the already inspected vendor hooks in this isolated copy.
    set ::env(JFM_PATH) $project_dir
    set alu_file [file join $project_dir rtl adder_op basic_alu.v]
    if {[llength [get_files -quiet $alu_file]] == 0} {add_files -norecurse $alu_file}
    update_compile_order -fileset sources_1
    note "PROJECT=$project_dir"
    note "REPORT_DIR=$report_dir"
    note "STARTED=[clock format [clock seconds]]"
    reset_run synth_1
    launch_runs impl_1 -to_step write_bitstream -jobs 2
    wait_on_run impl_1
    foreach r {synth_1 impl_1} {note "RUN=$r STATUS=[get_property STATUS [get_runs $r]] PROGRESS=[get_property PROGRESS [get_runs $r]]"}
    set impl [get_runs impl_1]
    if {[get_property PROGRESS $impl] ne "100%" || ![string match {*Complete*} [get_property STATUS $impl]]} {error "Implementation incomplete"}
    foreach bit [glob -nocomplain -directory [get_property DIRECTORY $impl] *.bit] {note "BIT=$bit SIZE=[file size $bit]"}
    open_run impl_1
    foreach {port pin} {FMC_HPC_GBTCLK0_M2C_C_P R6 FMC_HPC_GBTCLK0_M2C_C_N R5} {
        set actual [get_property PACKAGE_PIN [get_ports $port]]
        note "LITE_PIN=$port EXPECTED=$pin ACTUAL=$actual"
        if {$actual ne $pin} {error "Wrong Lite pin"}
    }
    set alu_cells [get_cells -hier -filter {NAME =~ *U_basic_alu*}]
    if {[llength $alu_cells] == 0} {error "ALU missing from implemented design"}
    note "ALU_CELLS=[llength $alu_cells]"
    report_timing_summary -file [file join $report_dir timing_summary.rpt]
    report_timing -through [get_pins -of_objects $alu_cells -filter {DIRECTION == OUT}] -max_paths 10 -file [file join $report_dir alu_timing.rpt]
    report_utilization -hierarchical -file [file join $report_dir utilization.rpt]
    report_drc -file [file join $report_dir drc.rpt]
    set f [open [file join $report_dir timing_summary.rpt] r]; set timing [read $f]; close $f
    note "BITSTREAM_GENERATION=PASS"
    if {[string first "Timing constraints are not met." $timing] >= 0} {
        note "TIMING_ACCEPTANCE=FAIL_REQUIRES_REPORT_REVIEW"
    } elseif {[string first "All user specified timing constraints are met." $timing] >= 0} {
        note "TIMING_ACCEPTANCE=PASS_UNDER_CURRENT_CONSTRAINTS"
    } else {error "No timing verdict found"}
    close_project
} err opts]} {
    note "BUILD_FAILED=$err"
    if {[dict exists $opts -errorinfo]} {note [dict get $opts -errorinfo]}
    catch {close_project}; close $report; exit 1
}
close $report
exit 0
