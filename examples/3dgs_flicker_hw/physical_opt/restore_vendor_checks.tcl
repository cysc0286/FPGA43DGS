# A partial project run starts a new Vivado session. DRC enable settings from
# the original vendor opt-pre hook are session state, not checkpoint state.
# Restore only settings already present in that hook. No new timing exception.
if {![info exists ::env(JFM_PATH)]} {error "JFM_PATH is required"}
set vendor_hook [file normalize [file join $::env(JFM_PATH) ip_patch process_control implementation_opt_pre.tcl]]
set f [open $vendor_hook r]; set vendor_text [read $f]; close $f
foreach rule {REQP-44 REQP-46 REQP-52 REQP-56} {
    set pattern [format {set_property IS_ENABLED 0 \[get_drc_checks +%s\]} $rule]
    if {![regexp $pattern $vendor_text]} {error "Not an inherited vendor setting: $rule"}
    set_property IS_ENABLED false [get_drc_checks $rule]
    puts "RESTORED_ORIGINAL_VENDOR_CHECK=$rule HOOK=$vendor_hook"
}
