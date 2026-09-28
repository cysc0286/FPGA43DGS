$ErrorActionPreference='Stop'
$env:RDI_PLATFORM='win64'
$env:PROCESSOR_ARCHITECTURE='AMD64'
$stamp=Get-Date -Format yyyyMMddTHHmmss
Push-Location (Join-Path $PSScriptRoot '..\build')
try {
 & 'D:\Xilinx\Vivado\2018.3\bin\vivado_hls.bat' -f (Join-Path $PSScriptRoot 'build.tcl') -l "pipeline_$stamp.log"
 if($LASTEXITCODE){throw 'HLS pipeline failed: inspect log'}
} finally {Pop-Location}
