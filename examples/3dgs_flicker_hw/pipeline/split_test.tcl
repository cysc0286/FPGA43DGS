set here [file dirname [file normalize [info script]]]
cd [file normalize [file join $here .. build]]
open_project -reset hls_split_differential
set_top flicker_split_probe
add_files [file join $here pipeline.cpp] -cflags {-std=c++0x -DFLK_SPLIT_TEST}
add_files [file join $here framed_ctu.cpp] -cflags {-std=c++0x}
add_files -tb [file join $here split_tb.cpp] -cflags {-std=c++0x}
open_solution -reset solution1
set_part {xc7z030ffg676-2}
create_clock -period 5 -name default
csim_design -clean -O
csynth_design
cosim_design -rtl verilog -tool xsim
exit
