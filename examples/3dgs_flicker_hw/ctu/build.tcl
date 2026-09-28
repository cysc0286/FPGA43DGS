set here [file dirname [file normalize [info script]]]
cd [file normalize [file join $here .. build]]
open_project -reset hls_ctu
set_top flicker_ctu
add_files [file join $here ctu.cpp] -cflags {-std=c++0x}
add_files -tb [file join $here tb.cpp] -cflags {-std=c++0x}
open_solution -reset solution1
set_part {xc7z030ffg676-2}
create_clock -period 5 -name default
csim_design -clean -O
csynth_design
cosim_design -rtl verilog -tool xsim
# Consume generated RTL directly, as for the rendering baseline. Vivado 2018.3
# IP catalog packaging uses a date-derived revision that overflows in 2026.
# Do not change the machine clock or suppress an actual RTL/cosim failure.
exit
