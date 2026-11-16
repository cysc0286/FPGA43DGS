# Module-only synthesis; never loads a design onto hardware.
# Run with Vivado 2018.3 -mode batch -source <this-file> -tclargs <report-dir>.
if {$argc != 1} {
    error "Expected one absolute report directory"
}
set report_dir [file normalize [lindex $argv 0]]
file mkdir $report_dir
set source_dir [file dirname [file normalize [info script]]]
set_param general.maxThreads 4
read_verilog [list [file join $source_dir camera_servo_pwm.v]]
# The board manual names XC7Z030FFG676-2I as the compatible device reference.
# This is a reference OOC estimate, not a Fudan full-platform implementation.
synth_design -top camera_servo_pwm -part xc7z030ffg676-2 -mode out_of_context
create_clock -name servo_clock -period 10.000 [get_ports i_clk]
report_utilization -file [file join $report_dir utilization_reference_ooc.rpt]
report_timing_summary -file [file join $report_dir timing_reference_ooc.rpt]
write_checkpoint -force [file join $report_dir camera_servo_reference_ooc.dcp]
puts "CAMERA_SERVO_REFERENCE_OOC_COMPLETE"
