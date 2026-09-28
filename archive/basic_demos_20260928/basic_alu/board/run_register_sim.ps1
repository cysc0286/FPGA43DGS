param([string]$VivadoBin='D:\Xilinx\Vivado\2018.3\bin')
$ErrorActionPreference='Stop'
$example=Split-Path $PSScriptRoot -Parent
$repo=Split-Path (Split-Path $example -Parent) -Parent
$rtl=Join-Path $repo 'platform\basic_alu_fpga\rtl\adder_op'
if (-not (Test-Path -LiteralPath (Join-Path $rtl 'basic_alu.v'))) {throw 'Run prepare_project.py first'}
if (-not $env:PROCESSOR_ARCHITECTURE) {$env:PROCESSOR_ARCHITECTURE='AMD64'}
if (-not $env:NUMBER_OF_PROCESSORS) {$env:NUMBER_OF_PROCESSORS=[string][Environment]::ProcessorCount}
$dir=Join-Path $example ('build\register_sim_'+(Get-Date -Format yyyyMMdd_HHmmss))
New-Item -ItemType Directory -Path $dir -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $example 'data\vectors.txt') -Destination (Join-Path $dir 'vectors.txt')
$files=@('basic_alu.v','adder_top.v','reg_ctrl.v','dma.v','adder.v','pulse_cross.v','AVR\avr_rs.v','AVR\avr_frs.v','AVR\avr_brs.v') | ForEach-Object {Join-Path $rtl $_}
Push-Location -LiteralPath $dir
try {
    & "$VivadoBin\xvlog.bat" --sv @files (Join-Path $example 'sim\tb_register_alu.sv')
    if ($LASTEXITCODE) {throw 'xvlog failed'}
    & "$VivadoBin\xelab.bat" work.tb_register_alu -s register_alu_sim
    if ($LASTEXITCODE) {throw 'xelab failed'}
    & "$VivadoBin\xsim.bat" register_alu_sim -runall -log simulation.log
    if ($LASTEXITCODE) {throw 'xsim failed'}
    $log=Get-Content simulation.log -Raw
    if ($log -notmatch 'PASS: register_alu' -or $log -match '(?im)^\s*(FAIL:|FATAL:|ERROR:)') {throw 'Register simulation assertions failed'}
    Write-Output "REGISTER_SIMULATION_PASSED=$dir"
} finally {Pop-Location}
