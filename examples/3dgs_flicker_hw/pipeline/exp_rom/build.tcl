set here [file dirname [file normalize [info script]]]
set build [file normalize [file join $here .. .. build]]
cd $build
open_project -reset exp_generate_20260930
set_top exp_probe
add_files [file join $here probe.cpp] -cflags {-std=c++0x}
add_files -tb [file join $here generate.cpp] -cflags {-std=c++0x}
open_solution -reset solution1
set_part {xc7z030ffg676-2}
create_clock -period 5 -name default
csim_design -clean -O -argv [file join $here table.hpp]
close_project
foreach candidate {baseline rom} {
 open_project -reset exp_probe_20260930_$candidate
 set_top exp_probe
 set flags {-std=c++0x}
 if {$candidate eq "rom"} {append flags " -DFLK_EXACT_EXP_ROM"}
 add_files [file join $here probe.cpp] -cflags $flags
 add_files -tb [file join $here tb.cpp] -cflags {-std=c++0x}
 open_solution -reset solution1
 set_part {xc7z030ffg676-2}
 create_clock -period 5 -name default
 csim_design -clean -O
 csynth_design
 cosim_design -rtl verilog -tool xsim
 close_project
}
exit
