param([string]$VivadoBin='D:\Xilinx\Vivado\2018.3\bin',[switch]$Registers)
$ErrorActionPreference='Stop'
if (-not $env:PROCESSOR_ARCHITECTURE) {$env:PROCESSOR_ARCHITECTURE='AMD64'}
if (-not $env:NUMBER_OF_PROCESSORS) {$env:NUMBER_OF_PROCESSORS=[string][Environment]::ProcessorCount}
$dir=Join-Path $PSScriptRoot ('build\sim_'+$(if($Registers){'register_'}else{'core_'})+(Get-Date -Format yyyyMMdd_HHmmss))
New-Item -ItemType Directory -Path $dir -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'data\vectors.txt') -Destination (Join-Path $dir 'vectors.txt')
$top='tb_gs_compositor'
$files=@((Join-Path $PSScriptRoot 'rtl\gs_compositor.v'))
if($Registers){
    $repo=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
    $rtl=Join-Path $repo 'platform\gs_compositor_fpga\rtl\adder_op'
    $top='tb_register_gs'
    $files=@('gs_compositor.v','adder_top.v','reg_ctrl.v','dma.v','adder.v','pulse_cross.v','AVR\avr_rs.v','AVR\avr_frs.v','AVR\avr_brs.v') | ForEach-Object {Join-Path $rtl $_}
}
Push-Location $dir
try {
    & "$VivadoBin\xvlog.bat" --sv @files (Join-Path $PSScriptRoot "sim\$top.sv")
    if($LASTEXITCODE){throw 'xvlog failed'}
    & "$VivadoBin\xelab.bat" "work.$top" -s gs_sim
    if($LASTEXITCODE){throw 'xelab failed'}
    & "$VivadoBin\xsim.bat" gs_sim -runall -log simulation.log
    if($LASTEXITCODE){throw 'xsim failed'}
    $log=Get-Content simulation.log -Raw
    if($log -notmatch 'PASS: .*84496' -or $log -match '(?im)^\s*(FAIL:|FATAL:|ERROR:)'){throw 'Simulation assertions failed'}
    Write-Output "SIMULATION_PASSED=$dir"
}finally{Pop-Location}
