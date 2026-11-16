# Read-only review of the isolated candidate after routing; never program hardware.
if {$argc != 1} {error "Usage: review_camera_servo.tcl <candidate-platform-directory>"}
set root [file normalize [lindex $argv 0]]
if {[file tail $root] ne "camera_servo_platform_j17_23_21"} {
    error "Unexpected platform; refusing to review a different candidate"
}
set reports [file join $root servo_reports]
file mkdir $reports
set result [open [file join $reports review_status.txt] w]
proc record {message} {global result; puts $message; puts $result $message; flush $result}
if {[catch {
    set routed [file join $root fpai_demo_vivado.runs impl_1 ai7030_edif_top_routed.dcp]
    if {![file exists $routed]} {error "No routed checkpoint"}
    open_checkpoint $routed
    foreach {port ball} {camera_pan_pwm AE13 camera_tilt_pwm W13} {
        set p [get_ports $port]
        if {[get_property PACKAGE_PIN $p] ne $ball || [get_property IOSTANDARD $p] ne "LVCMOS33"} {
            error "Incorrect candidate I/O assignment"
        }
        record "PORT=$port BALL=$ball IOSTANDARD=LVCMOS33"
    }
    set regs [get_cells -hier -filter {NAME =~ *U_camera_servo_bus* && IS_SEQUENTIAL}]
    set logic [get_cells -hier -filter {NAME =~ *U_camera_servo_bus* && IS_PRIMITIVE && !IS_SEQUENTIAL}]
    if {![llength $regs]} {error "Servo registers were not found"}
    set clocks [lsort -unique [get_clocks -of_objects [get_pins -of_objects $regs -filter {REF_PIN_NAME == C}]]]
    if {![llength $clocks]} {error "Servo registers have no timing clock"}
    foreach clock $clocks {
        set period [get_property PERIOD $clock]
        record "CLOCK=$clock PERIOD_NS=$period"
        if {abs($period - 10.0) > 0.001} {error "Servo clock must be 100 MHz"}
    }
    # Include combinational pass-through added to the original renderer bus,
    # not only paths terminating in the new servo counters.
    set through [get_pins -of_objects $logic -filter {DIRECTION == OUT}]
    set all_pass 1
    foreach delay {max min} {
        foreach scope {from to through} {
            switch $scope {
                from {set objects $regs}
                to {set objects $regs}
                through {set objects $through}
            }
            if {![llength $objects]} {error "No objects for timing scope $scope"}
            set paths [get_timing_paths -delay_type $delay -$scope $objects -max_paths 1]
            if {![llength $paths]} {error "No timed path in $delay/$scope scope"}
            set slack [get_property SLACK [lindex $paths 0]]
            record "NEW_LOGIC_${delay}_${scope}_SLACK_NS=$slack"
            if {$slack < 0.0} {set all_pass 0}
            report_timing -delay_type $delay -$scope $objects -max_paths 20 \
                -file [file join $reports new_logic_${delay}_${scope}.rpt]
        }
    }
    report_timing_summary -file [file join $reports reviewed_timing_summary.rpt]
    report_utilization -file [file join $reports reviewed_utilization.rpt]
    report_io -file [file join $reports reviewed_io.rpt]
    report_drc -file [file join $reports reviewed_drc.rpt]
    if {!$all_pass} {error "New servo or register-routing timing failed"}
    record "NEW_LOGIC_TIMING=PASS_UNDER_CURRENT_CONSTRAINTS"
    record "FULL_CHIP_SIGNOFF=REQUIRES_SEPARATE_REVIEW"
    record "BOARD_PROGRAMMED=NO"
    close_design
} message options]} {
    record "REVIEW_FAILED=$message"
    if {[dict exists $options -errorinfo]} {record [dict get $options -errorinfo]}
    close $result
    exit 1
}
close $result
exit 0
