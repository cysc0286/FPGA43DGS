$ErrorActionPreference='Stop'
$env:RDI_PLATFORM='win64'
$env:PROCESSOR_ARCHITECTURE='AMD64'
$stamp=Get-Date -Format yyyyMMdd_HHmmss
Push-Location (Join-Path $PSScriptRoot '..\build')
try {
 & 'D:\Xilinx\Vivado\2018.3\bin\vivado_hls.bat' -f (Join-Path $PSScriptRoot 'build.tcl') *> "ctu_$stamp.log"
 if($LASTEXITCODE -or (Get-Content "ctu_$stamp.log" -Raw) -match '(?m)^ERROR:') {throw "CTU build failed: ctu_$stamp.log"}
 Write-Output "CTU_BUILD_LOG=ctu_$stamp.log"
} finally {Pop-Location}
