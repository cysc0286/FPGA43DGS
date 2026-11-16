# Run only against a separate clone made for this candidate. Never program a board.
if {$argc != 1} {error "Usage: build_camera_servo.tcl <candidate-platform-directory>"}
set root [file normalize [lindex $argv 0]]
if {[file tail $root] ne "camera_servo_platform_j17_23_21"} {
    error "Refusing to edit any project other than the explicitly named servo candidate"
}
set out [file join $root servo_reports]
file mkdir $out
set log [open [file join $out status.txt] w]
proc note {text} {global log; puts $text; puts $log $text; flush $log}
if {[catch {
    if {[version -short] ne "2018.3"} {error "Vivado 2018.3 is required"}
    open_project [file join $root fpai_demo_vivado.xpr]
    if {[get_property PART [current_project]] ne "xc7z030ffg676-2"} {error "Unexpected reference part"}
    foreach f [get_files -all] {
        set name [file normalize [get_property NAME $f]]
        if {[string first "D:/ADProjects/FPGA43DGS/platform/flicker_groupshared_fpga/" $name] == 0} {
            error "Candidate retains a mutable reference into the frozen parent: $name"
        }
    }
    add_files -norecurse [list [file join $root rtl camera_servo_pwm.v] [file join $root rtl camera_servo_bus.v]]
    add_files -fileset constrs_1 -norecurse [list [file join $root constrs_1 camera_servo_j17_23_21.xdc]]
    update_compile_order -fileset sources_1
    note "CLOCK_SOURCE=clk_wiz_0.clk_out1 -> gp_axi_aclk -> ps_clk_out -> gp_clk; requested 100 MHz"
    source [file join $root ip_patch run.tcl]
    reset_run synth_1
    launch_runs impl_1 -to_step write_bitstream -jobs 2
    wait_on_run impl_1
    foreach r {synth_1 impl_1} {note "RUN=$r STATUS=[get_property STATUS [get_runs $r]] PROGRESS=[get_property PROGRESS [get_runs $r]]"}
    if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Implementation did not complete"}
    open_run impl_1
    foreach {port pin} {camera_pan_pwm AE13 camera_tilt_pwm W13} {
        set obj [get_ports $port]
        if {[get_property PACKAGE_PIN $obj] ne $pin || [get_property IOSTANDARD $obj] ne "LVCMOS33"} {
            error "Servo pin assignment or voltage standard is wrong"
        }
        note "PIN=$port BALL=$pin IOSTANDARD=[get_property IOSTANDARD $obj]"
    }
    set servo_regs [get_cells -hier -filter {NAME =~ *U_camera_servo_bus* && IS_SEQUENTIAL}]
    if {[llength $servo_regs] == 0} {error "No servo registers found after implementation"}
    note "SERVO_REGISTERS=[llength $servo_regs]"
    set servo_clocks [get_clocks -of_objects [get_pins -of_objects $servo_regs -filter {REF_PIN_NAME == C}]]
    if {[llength $servo_clocks] == 0} {error "No timing clock reaches the servo registers"}
    foreach clk $servo_clocks {
        note "SERVO_CLOCK=$clk PERIOD=[get_property PERIOD $clk]"
        if {abs([get_property PERIOD $clk] - 10.0) > 0.001} {error "Servo clock is not 100 MHz"}
    }
    report_timing_summary -file [file join $out timing_summary.rpt]
    report_timing -to $servo_regs -max_paths 20 -file [file join $out servo_setup.rpt]
    report_timing -delay_type min -to $servo_regs -max_paths 20 -file [file join $out servo_hold.rpt]
    report_utilization -file [file join $out utilization.rpt]
    report_io -file [file join $out io.rpt]
    report_drc -file [file join $out drc.rpt]
    note "BITSTREAM_GENERATION=COMPLETE_REQUIRES_REVIEW_NOT_DEPLOYED"
    close_project
} message opts]} {
    note "BUILD_FAILED=$message"
    if {[dict exists $opts -errorinfo]} {note [dict get $opts -errorinfo]}
    catch {close_project}
    close $log
    exit 1
}
close $log
exit 0
