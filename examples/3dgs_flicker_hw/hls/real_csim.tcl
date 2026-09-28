set here [file dirname [file normalize [info script]]]
cd [file normalize [file join $here .. build]]
open_project hls_renderer
open_solution solution1
csim_design -clean -O -argv "$::env(FLK_INPUT) $::env(FLK_OUTPUT) $::env(FLK_TILES)"
exit
