[CmdletBinding()]
param([string]$VivadoBin = 'D:\Xilinx\Vivado\2018.3\bin')
$ErrorActionPreference = 'Stop'
$repo = Split-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) -Parent
$project = Join-Path $repo 'platform\gs_compositor_fpga'
if (-not (Test-Path -LiteralPath (Join-Path $project 'gs_compositor_manifest.json'))) { throw 'Run prepare_project.py first.' }
$env:JFM_PATH = $project
$env:FMSH_PROCISE_PATH = [Environment]::GetEnvironmentVariable('FMSH_PROCISE_PATH','Machine')
if (-not $env:PROCESSOR_ARCHITECTURE) { $env:PROCESSOR_ARCHITECTURE = 'AMD64' }
if (-not $env:NUMBER_OF_PROCESSORS) { $env:NUMBER_OF_PROCESSORS = [string][Environment]::ProcessorCount }
$output = Join-Path (Split-Path $PSScriptRoot -Parent) 'build'
New-Item -ItemType Directory -Force -Path $output | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
Push-Location -LiteralPath $output
try {
    & (Join-Path $VivadoBin 'vivado.bat') -mode batch -source (Join-Path $PSScriptRoot 'build.tcl') -log "build_$stamp.log" -journal "build_$stamp.jou"
    if ($LASTEXITCODE -ne 0) { throw "Vivado failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
