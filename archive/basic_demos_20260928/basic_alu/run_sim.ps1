param([string]$VivadoBin='D:\Xilinx\Vivado\2018.3\bin')
$ErrorActionPreference='Stop'
if (-not $env:PROCESSOR_ARCHITECTURE) {$env:PROCESSOR_ARCHITECTURE='AMD64'}
if (-not $env:NUMBER_OF_PROCESSORS) {$env:NUMBER_OF_PROCESSORS=[string][Environment]::ProcessorCount}
$dir=Join-Path $PSScriptRoot ('build\sim_'+(Get-Date -Format yyyyMMdd_HHmmss))
New-Item -ItemType Directory -Path $dir -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'data\vectors.txt') -Destination (Join-Path $dir 'vectors.txt')
Push-Location $dir
try {
    & "$VivadoBin\xvlog.bat" --sv (Join-Path $PSScriptRoot 'rtl\basic_alu.v') (Join-Path $PSScriptRoot 'sim\tb_basic_alu.sv')
    if ($LASTEXITCODE) {throw 'xvlog failed'}
    & "$VivadoBin\xelab.bat" work.tb_basic_alu -s basic_alu_sim
    if ($LASTEXITCODE) {throw 'xelab failed'}
    & "$VivadoBin\xsim.bat" basic_alu_sim -runall -log simulation.log
    if ($LASTEXITCODE) {throw 'xsim failed'}
    $log=Get-Content simulation.log -Raw
    if ($log -notmatch 'PASS: basic_alu' -or $log -match '(?im)^\s*(FAIL:|FATAL:|ERROR:)') {throw 'Simulation assertions failed'}
    Write-Output "SIMULATION_PASSED=$dir"
} finally {Pop-Location}
