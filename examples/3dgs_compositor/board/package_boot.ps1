param([Parameter(Mandatory=$true)][string]$ReportDir)
$ErrorActionPreference='Stop'
$example=Split-Path $PSScriptRoot -Parent
$repo=Split-Path (Split-Path $example -Parent) -Parent
$project=Join-Path $repo 'platform\gs_compositor_fpga'
$status=Get-Content -LiteralPath (Join-Path $ReportDir 'status.txt') -Raw
if($status -notmatch '(?m)^BITSTREAM_GENERATION=PASS\s*$' -or $status -match 'BUILD_FAILED='){throw 'No completed implementation'}
$manifest=Get-Content (Join-Path $project 'gs_compositor_manifest.json') -Raw | ConvertFrom-Json
$bit=Join-Path $project 'fpai_demo_vivado.runs\impl_1\ai7030_edif_top_disable_icap.bit'
foreach($rtl in @((Join-Path $example 'rtl\gs_compositor.v'),(Join-Path $project 'rtl\adder_op\gs_compositor.v'))){
    if((Get-FileHash -LiteralPath $rtl -Algorithm SHA256).Hash -ne $manifest.rtl_sha256){throw 'RTL drift'}
}
if((Get-FileHash -LiteralPath (Join-Path $project 'rtl\adder_op\adder_top.v')).Hash -ne $manifest.patched_top_sha256){throw 'Patched top drift'}
if(-not $status.Contains('BIT='+$bit.Replace('\','/')+' SIZE=')){throw 'Unexpected bitstream'}
if((Get-Item $bit).LastWriteTime -gt (Get-Item (Join-Path $ReportDir 'status.txt')).LastWriteTime){throw 'Stale report'}
$boot=Join-Path $ReportDir 'boot'
if(Test-Path $boot){throw 'Packaging directory exists; inspect previous result'}
New-Item -ItemType Directory -Path $boot | Out-Null
$source=Join-Path $repo 'platform\BOOT_Gen'
foreach($name in @('FSBL251210_ddr400_demo.out','bl31.elf','u-boot','create_boot.tcl','create_boot.ps1')){
    Copy-Item -LiteralPath (Join-Path $source $name) -Destination $boot
}
$bif=(Get-Content (Join-Path $source '7030ai_psin.bif') -Raw).Replace($source.Replace('\','/'),$boot.Replace('\','/'))
[IO.File]::WriteAllText((Join-Path $boot '7030ai_psin.bif'),$bif,[Text.UTF8Encoding]::new($false))
& (Join-Path $boot 'create_boot.ps1') -BitFile $bit
if(-not (Test-Path (Join-Path $boot 'BOOT.bin'))){throw 'BOOT absent'}
python (Join-Path $PSScriptRoot 'verify_boot.py') --boot (Join-Path $boot 'BOOT.bin') --bit $bit --output (Join-Path $boot 'comparison.json')
if($LASTEXITCODE){throw 'BOOT contents did not match verified inputs'}
Write-Output "PACKAGED_BOOT=$boot"
