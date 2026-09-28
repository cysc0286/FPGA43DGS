set here [file dirname [file normalize [info script]]]
cd [file normalize [file join $here .. build]]
open_project hls_ctu
open_solution solution1
set ref [file normalize [file join $here .. evidence ctu_real]]
csim_design -clean -O -argv "[file join $ref input.flk] [file join $ref expected]"
cosim_design -rtl verilog -tool xsim -argv "[file join $ref input.flk] [file join $ref expected]"
exit
