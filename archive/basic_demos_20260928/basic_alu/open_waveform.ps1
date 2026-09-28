[CmdletBinding()]
param([string]$VivadoBin='D:\Xilinx\Vivado\2018.3\bin')
$ErrorActionPreference='Stop'
if (-not $env:PROCESSOR_ARCHITECTURE) {$env:PROCESSOR_ARCHITECTURE='AMD64'}
if (-not $env:NUMBER_OF_PROCESSORS) {$env:NUMBER_OF_PROCESSORS=[string][Environment]::ProcessorCount}
$dir=Join-Path $PSScriptRoot ('build\waveform_'+(Get-Date -Format yyyyMMdd_HHmmss))
New-Item -ItemType Directory -Force -Path $dir | Out-Null
Push-Location -LiteralPath $dir
try {
    & "$VivadoBin\xvlog.bat" --sv (Join-Path $PSScriptRoot 'rtl\basic_alu.v') (Join-Path $PSScriptRoot 'sim\tb_alu_walkthrough.sv')
    if ($LASTEXITCODE) {throw 'xvlog failed'}
    & "$VivadoBin\xelab.bat" work.tb_alu_walkthrough -debug typical -s alu_walkthrough
    if ($LASTEXITCODE) {throw 'xelab failed'}
    $script=(Join-Path $PSScriptRoot 'sim\walkthrough.tcl').Replace('\','/')
    & "$VivadoBin\xsim.bat" alu_walkthrough -tclbatch $script -wdb walkthrough.wdb -log walkthrough.log
    if ($LASTEXITCODE) {throw 'xsim failed'}
    $text=Get-Content -LiteralPath 'walkthrough.log' -Raw
    if ($text -notmatch 'PASS: ALU waveform walkthrough' -or $text -notmatch 'WAVEFORM_READY=PASS' -or $text -match '(?im)^\s*(FAIL:|FATAL:|ERROR:)') {throw 'Waveform validation failed'}
    $args=@('walkthrough.wdb','-gui','-view','walkthrough.wcfg','-log','waveform_gui.log')
    $process=Start-Process -FilePath "$VivadoBin\xsim.bat" -ArgumentList $args -WorkingDirectory $dir -WindowStyle Normal -PassThru
    Write-Output "WAVEFORM_GUI_LAUNCHED_PID=$($process.Id) DIRECTORY=$dir"
} finally {Pop-Location}
