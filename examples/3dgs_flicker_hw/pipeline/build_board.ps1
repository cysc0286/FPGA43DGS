param([string]$VivadoBin='D:\Xilinx\Vivado\2018.3\bin',[string]$PlatformName='flicker_cat_fpga')
$ErrorActionPreference='Stop'
$repo=Split-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) -Parent
$env:FLK_PLATFORM_NAME=$PlatformName
$env:JFM_PATH=Join-Path $repo "platform\$PlatformName"
$env:FMSH_PROCISE_PATH=[Environment]::GetEnvironmentVariable('FMSH_PROCISE_PATH','Machine')
$env:PROCESSOR_ARCHITECTURE='AMD64'
$env:RDI_PLATFORM='win64'
$work=Join-Path $PSScriptRoot '..\build'
$stamp=Get-Date -Format yyyyMMdd_HHmmss
Push-Location $work
try{
 & (Join-Path $VivadoBin 'vivado.bat') -mode batch -source (Join-Path $PSScriptRoot 'build_board.tcl') -log "platform_cat_$stamp.log" -journal "platform_cat_$stamp.jou"
 if($LASTEXITCODE){throw 'Platform implementation failed'}
}finally{Pop-Location}
