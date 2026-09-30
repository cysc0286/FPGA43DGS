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
add_files [file join $here pipeline.cpp] -cflags $flags
add_files [file join $here framed_ctu.cpp] -cflags {-std=c++0x}
add_files -tb [file join $here .. hls renderer.cpp] -cflags {-std=c++0x}
add_files -tb [file join $here tb.cpp] -cflags {-std=c++0x}
open_solution -reset solution1
set_part {xc7z030ffg676-2}
create_clock -period 5 -name default
csim_design -clean -O
csynth_design
foreach f [glob $project/solution1/syn/report/evaluate_mini*csynth.xml] {
 set fd [open $f r];set report [read $fd];close $fd
 foreach {all ii} [regexp -all -inline {<PipelineII>([0-9]+)</PipelineII>} $report] {
  if {$ii != 1} {error "Mini-tile pipeline II=$ii failed in $f; preserve reports, skip expensive cosim"}
 }
}
cosim_design -rtl verilog -tool xsim
exit
