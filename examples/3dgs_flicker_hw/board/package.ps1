param([Parameter(Mandatory=$true)][string]$ReportDir)
$ErrorActionPreference='Stop'
$repo=Split-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) -Parent
$status=Get-Content -LiteralPath (Join-Path $ReportDir 'status.txt') -Raw
if($status -notmatch '(?m)^BITSTREAM_GENERATION=PASS\s*$'){throw 'Incomplete platform build'}
$boot=Join-Path (Resolve-Path -LiteralPath $ReportDir).Path 'boot'
if(Test-Path -LiteralPath $boot){throw 'Package exists; inspect before replacing'}
New-Item -ItemType Directory -Path $boot|Out-Null
$source=Join-Path $repo 'platform\BOOT_Gen'
foreach($name in @('FSBL251210_ddr400_demo.out','bl31.elf','u-boot','create_boot.tcl','create_boot.ps1')){
 Copy-Item -LiteralPath (Join-Path $source $name) -Destination (Join-Path $boot $name)
}
$bif=(Get-Content -LiteralPath (Join-Path $source '7030ai_psin.bif') -Raw).Replace($source.Replace('\','/'),$boot.Replace('\','/'))
[IO.File]::WriteAllText((Join-Path $boot '7030ai_psin.bif'),$bif,[Text.UTF8Encoding]::new($false))
$bit=Join-Path $repo 'platform\flicker_fpga\fpai_demo_vivado.runs\impl_1\ai7030_edif_top_disable_icap.bit'
& (Join-Path $boot 'create_boot.ps1') -BitFile $bit
& D:\Tools\Python31210\python.exe (Join-Path $repo 'examples\3dgs_compositor\board\verify_boot.py') --boot (Join-Path $boot 'BOOT.bin') --bit $bit --output (Join-Path $boot 'comparison.json')
if($LASTEXITCODE){throw 'BOOT partition verification failed'}
Write-Output "VERIFIED_PACKAGE=$boot"
