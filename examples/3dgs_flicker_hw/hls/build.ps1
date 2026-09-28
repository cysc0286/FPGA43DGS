param([string]$VivadoBin='D:\Xilinx\Vivado\2018.3\bin')
$ErrorActionPreference='Stop'
$env:RDI_PLATFORM='win64'
$env:PROCESSOR_ARCHITECTURE='AMD64'
$work=Join-Path $PSScriptRoot '..\build'
New-Item -ItemType Directory -Force -Path $work | Out-Null
Push-Location $work
try {
 & (Join-Path $VivadoBin 'vivado_hls.bat') -f (Join-Path $PSScriptRoot 'build.tcl') | Tee-Object -FilePath hls_run.txt
 if($LASTEXITCODE){throw 'HLS build failed'}
 if((Get-Content hls_run.txt -Raw) -match '(?m)^ERROR:'){throw 'HLS reported errors'}
} finally {Pop-Location}
