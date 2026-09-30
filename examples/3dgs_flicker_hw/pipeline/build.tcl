set here [file dirname [file normalize [info script]]]
cd [file normalize [file join $here .. build]]
set project hls_pipeline_workset
if {[info exists ::env(FLK_HLS_PROJECT)]} {set project $::env(FLK_HLS_PROJECT)}
open_project -reset $project
set_top flicker_render_pipeline
set flags {-std=c++0x}
if {[info exists ::env(FLK_EXACT_EXP_ROM)] && $::env(FLK_EXACT_EXP_ROM) in {1 2}} {
 append flags " -DFLK_EXACT_EXP_ROM=$::env(FLK_EXACT_EXP_ROM)"
}
if {[info exists ::env(FLK_STATE_PORTS)]} {
 if {$::env(FLK_STATE_PORTS) ni {1 2}} {error "FLK_STATE_PORTS must be 1 or 2"}
 append flags " -DFLK_STATE_PORTS=$::env(FLK_STATE_PORTS)"
}
if {[info exists ::env(FLK_GROUP_SUBTILES)]} {
 if {$::env(FLK_GROUP_SUBTILES) ni {0 1 2 3}} {error "FLK_GROUP_SUBTILES must be 0, 1, 2 or 3"}
 append flags " -DFLK_GROUP_SUBTILES=$::env(FLK_GROUP_SUBTILES)"
}
if {[info exists ::env(FLK_GROUP_TRIM_RANGE)]} {
 if {$::env(FLK_GROUP_TRIM_RANGE) ni {0 1}} {error "FLK_GROUP_TRIM_RANGE must be 0 or 1"}
 append flags " -DFLK_GROUP_TRIM_RANGE=$::env(FLK_GROUP_TRIM_RANGE)"
}
if {[info exists ::env(FLK_LANE_FIFO_STORAGE)]} {
 if {$::env(FLK_LANE_FIFO_STORAGE) ni {0 1 2}} {error "FLK_LANE_FIFO_STORAGE must be 0, 1 or 2"}
 append flags " -DFLK_LANE_FIFO_STORAGE=$::env(FLK_LANE_FIFO_STORAGE)"
}
if {[info exists ::env(FLK_GROUP_SHARED_ATTR)]} {
 if {$::env(FLK_GROUP_SHARED_ATTR) ni {0 1}} {error "FLK_GROUP_SHARED_ATTR must be 0 or 1"}
 append flags " -DFLK_GROUP_SHARED_ATTR=$::env(FLK_GROUP_SHARED_ATTR)"
}
add_files [file join $here pipeline.cpp] -cflags $flags
add_files [file join $here framed_ctu.cpp] -cflags {-std=c++0x}
add_files -tb [file join $here .. hls renderer.cpp] -cflags {-std=c++0x}
add_files -tb [file join $here tb.cpp] -cflags {-std=c++0x}
open_solution -reset solution1
set_part {xc7z030ffg676-2}
create_clock -period 5 -name default
csim_design -clean -O
csynth_design
foreach f [glob -nocomplain $project/solution1/syn/report/evaluate_*csynth.xml] {
 set fd [open $f r];set report [read $fd];close $fd
 foreach {all ii} [regexp -all -inline {<PipelineII>([0-9]+)</PipelineII>} $report] {
  if {$ii != 1} {error "Mini-tile pipeline II=$ii failed in $f; preserve reports, skip expensive cosim"}
 }
}
cosim_design -rtl verilog -tool xsim
exit
