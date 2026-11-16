# User-selected replacement for the ambiguous J17 3/5 wiring.
# JFMQL30TAI_LITE.pdf SHA256
# cdec19cef2b65fb09340c89bcc0fd8d98fdaaa79421b210112940de488b5e1b6
# PDF page 16 (drawing sheet 17): J17 pin23=HR_IO13_1P, pin21=HR_IO13_2P.
# PDF page 9: those nets connect to actual BANK 12, VCCO=VCC3V3.
# Confirmed in the compatible XC7Z030FFG676 package database.
# Does not by itself qualify the whole design or authorize loading a bitstream.
set_property PACKAGE_PIN AE13 [get_ports camera_pan_pwm]
set_property PACKAGE_PIN W13 [get_ports camera_tilt_pwm]
set_property IOSTANDARD LVCMOS33 [get_ports {camera_pan_pwm camera_tilt_pwm}]
set_property DRIVE 4 [get_ports {camera_pan_pwm camera_tilt_pwm}]
set_property SLEW SLOW [get_ports {camera_pan_pwm camera_tilt_pwm}]
