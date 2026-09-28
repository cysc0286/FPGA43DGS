[CmdletBinding()]
param([string]$VivadoBin='D:\Xilinx\Vivado\2018.3\bin')
$ErrorActionPreference='Stop'
if (-not $env:PROCESSOR_ARCHITECTURE) {$env:PROCESSOR_ARCHITECTURE='AMD64'}
if (-not $env:NUMBER_OF_PROCESSORS) {$env:NUMBER_OF_PROCESSORS=[string][Environment]::ProcessorCount}
$output=Join-Path (Split-Path $PSScriptRoot -Parent) 'build'
New-Item -ItemType Directory -Force -Path $output | Out-Null
$tcl=(Join-Path $PSScriptRoot 'open_review.tcl').Replace('\','/')
$stamp=Get-Date -Format yyyyMMdd_HHmmss
# A visible window is intentional: the user requested an interactive project view.
$arguments=@('-mode','gui','-source',('"'+$tcl+'"'),'-log',"review_$stamp.log",'-journal',"review_$stamp.jou")
$process=Start-Process -FilePath (Join-Path $VivadoBin 'vivado.bat') -ArgumentList $arguments -WorkingDirectory $output -WindowStyle Normal -PassThru
Write-Output "VIVADO_REVIEW_LAUNCHED_PID=$($process.Id)"
