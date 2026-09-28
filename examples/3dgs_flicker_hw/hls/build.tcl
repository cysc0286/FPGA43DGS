set here [file dirname [file normalize [info script]]]
cd [file normalize [file join $here .. build]]
open_project -reset hls_renderer
set_top flicker_render
add_files [file join $here renderer.cpp] -cflags "-std=c++0x"
add_files -tb [file join $here tb.cpp] -cflags "-std=c++0x"
open_solution -reset solution1
set_part {xc7z030ffg676-2}
create_clock -period 5 -name default
config_compile -no_signed_zeros
csim_design
csynth_design
cosim_design -rtl verilog
exit
