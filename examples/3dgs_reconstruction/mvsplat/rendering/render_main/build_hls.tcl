# Isolated rebuild; never changes the installed firmware.
set here [file dirname [file normalize [info script]]]
set src [file join $here fpga pipeline]
set stage csim
if {[info exists ::env(RENDER_BUILD_STAGE)]} {set stage $::env(RENDER_BUILD_STAGE)}
if {$stage ni {csim synth cosim}} {error "stage must be csim, synth or cosim"}
file mkdir [file join $here build]
cd [file join $here build]
set project [format "%s_%s" $stage [clock seconds]]
if {[file exists $project]} {error "Refusing to overwrite existing build"}
file mkdir [file dirname $project]
open_project $project
set_top flicker_render_pipeline
set flags {-std=c++0x -DFLK_GROUP_SUBTILES=3 -DFLK_GROUP_TRIM_RANGE=1 -DFLK_LANE_FIFO_STORAGE=1 -DFLK_GROUP_SHARED_ATTR=1 -DFLK_EXACT_EXP_ROM=2 -DFLK_STREAM_LANE=0 -DFLK_STREAM_CORES=4}
add_files [file join $src pipeline.cpp] -cflags $flags
add_files [file join $src framed_ctu.cpp] -cflags $flags
add_files -tb [file join $here fpga hls renderer.cpp] -cflags {-std=c++0x}
add_files -tb [file join $src tb.cpp] -cflags $flags
open_solution solution1
set_part {xc7z030ffg676-2}
create_clock -period 5 -name default
csim_design -clean -O
if {$stage ne "csim"} {csynth_design}
if {$stage eq "cosim"} {cosim_design -rtl verilog -tool xsim}
puts "RENDER_PACKAGE_PASS stage=$stage project=$project"
exit
