param(
 [ValidateSet('csim','synth','cosim')][string]$Stage='csim',
 [string]$Hls='vivado_hls.bat'
)
$ErrorActionPreference='Stop'
$previous=$env:RENDER_BUILD_STAGE
try {
 $env:RENDER_BUILD_STAGE=$Stage
 $logDir=Join-Path $PSScriptRoot 'build'
 New-Item -ItemType Directory -Path $logDir -Force | Out-Null
 $log=Join-Path $logDir ('driver_'+[Guid]::NewGuid().ToString('N')+'.log')
 & $Hls -f (Join-Path $PSScriptRoot 'build_hls.tcl') -l $log
 if($LASTEXITCODE -or -not (Test-Path -LiteralPath $log) -or (Get-Content -LiteralPath $log -Raw) -notmatch 'RENDER_PACKAGE_PASS'){throw "HLS $Stage failed; inspect $log"}
} finally {$env:RENDER_BUILD_STAGE=$previous}
