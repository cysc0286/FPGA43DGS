$ErrorActionPreference='Stop'
$env:RDI_PLATFORM='win64'
$env:PROCESSOR_ARCHITECTURE='AMD64'
$stamp=Get-Date -Format yyyyMMddTHHmmss
Push-Location (Join-Path $PSScriptRoot '..\build')
try {
 & 'D:\Xilinx\Vivado\2018.3\bin\vivado_hls.bat' -f (Join-Path $PSScriptRoot 'build.tcl') -l "pipeline_$stamp.log"
 if($LASTEXITCODE){throw 'HLS pipeline failed: inspect log'}
 # Vivado HLS 2018.3 can return zero after a Tcl error. Require the actual
 # C/RTL report rather than treating process exit alone as acceptance.
 $project='hls_pipeline_workset'
 if($env:FLK_HLS_PROJECT){$project=$env:FLK_HLS_PROJECT}
 $report=Join-Path $project 'solution1/sim/report/flicker_render_pipeline_cosim.rpt'
 if(!(Test-Path $report) -or (Get-Content $report -Raw) -notmatch '\|\s*Verilog\|\s*Pass\|'){
  throw 'HLS pipeline did not pass C/RTL co-simulation; inspect reports and log'
 }
} finally {Pop-Location}
