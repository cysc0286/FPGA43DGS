[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$ReportDir,
    [string]$ProciseRoot='D:\FudanMicro\Procise',
    [string]$CableSerial='210251784595',
    [switch]$AllowTimingFailure
)
$ErrorActionPreference='Stop'
$example=Split-Path $PSScriptRoot -Parent
$repo=Split-Path (Split-Path $example -Parent) -Parent
$project=Join-Path $repo 'platform\gs_compositor_fpga'
$report=Get-Content -LiteralPath (Join-Path $ReportDir 'status.txt') -Raw
if ($report -notmatch '(?m)^BITSTREAM_GENERATION=PASS\s*$' -or $report -match '(?m)^BUILD_FAILED=') {
    throw 'No successful build report; do not load a copied or stale bitstream.'
}
if ($report -notmatch 'TIMING_ACCEPTANCE=PASS_UNDER_CURRENT_CONSTRAINTS' -and -not $AllowTimingFailure) {
    throw 'Timing did not pass. Inspect reports; prototype use requires -AllowTimingFailure.'
}
$manifest=Get-Content -LiteralPath (Join-Path $project 'gs_compositor_manifest.json') -Raw | ConvertFrom-Json
foreach ($rtl in @((Join-Path $example 'rtl\gs_compositor.v'), (Join-Path $project 'rtl\adder_op\gs_compositor.v'))) {
    if ((Get-FileHash -LiteralPath $rtl -Algorithm SHA256).Hash -ne $manifest.rtl_sha256) {throw 'RTL differs from prepared project; rebuild first.'}
}
$bit=Join-Path $project 'fpai_demo_vivado.runs\impl_1\ai7030_edif_top_disable_icap.bit'
$bitTcl=$bit.Replace('\','/')
if (-not $report.Contains("BIT=$bitTcl SIZE=")) {throw 'Report does not identify the expected bitstream.'}
if ((Get-Item -LiteralPath $bit).LastWriteTime -gt (Get-Item -LiteralPath (Join-Path $ReportDir 'status.txt')).LastWriteTime) {throw 'Bitstream is newer than this report.'}
if ($CableSerial -notmatch '^\d+$') {throw 'Unexpected cable serial format.'}
$env:TCL_LIBRARY=Join-Path $ProciseRoot 'tcl8.4'
$env:ICTIME_HOME=$ProciseRoot
$env:APP_DIR=$ProciseRoot
$env:FMSH_DB=Join-Path $ProciseRoot 'db'
$env:PATH="$ProciseRoot\dll;$ProciseRoot\bin;"+$env:PATH
$stamp=Get-Date -Format yyyyMMdd_HHmmss
$output=Join-Path $example 'build'
$log=Join-Path $output "program_$stamp.log"
$err=Join-Path $output "program_$stamp.err"
$tcl=Join-Path $output "program_$stamp.tcl"
$script=@"
puts {TEMPORARY_COMPOSITOR_PROGRAM_BEGIN}
if {[catch {init_chain -cable_type usb-jtag-smt2 -serial_number $CableSerial} message]} {
    puts "CHAIN_ERROR=`$message"
    exit 1
}
if {[catch {program_bit {$bitTcl} -part 0} message]} {
    puts "PROGRAM_ERROR=`$message"
    exit 2
}
puts "PROGRAM_RESULT=`$message"
puts {TEMPORARY_COMPOSITOR_PROGRAM_COMPLETE}
exit
"@
[IO.File]::WriteAllText($tcl,$script,[Text.UTF8Encoding]::new($false))
$hash=(Get-FileHash -LiteralPath $bit -Algorithm SHA256).Hash
Write-Output "TEST_BIT_SHA256=$hash"
$proc=Start-Process -FilePath (Join-Path $ProciseRoot 'bin\procise.exe') -ArgumentList ('"'+$tcl.Replace('\','/')+'"') -WorkingDirectory $output -WindowStyle Hidden -PassThru -RedirectStandardOutput $log -RedirectStandardError $err
if (-not $proc.WaitForExit(45000)) {throw "Programming still running (PID $($proc.Id)); inspect logs before retrying."}
$proc.Refresh()
$text=Get-Content -LiteralPath $log -Raw
Write-Output $text
if ($proc.ExitCode -ne 0 -or $text -notmatch 'TEMPORARY_COMPOSITOR_PROGRAM_COMPLETE') {throw "Programming did not complete: $log / $err"}
@{bit=$bit;sha256=$hash;report_dir=$ReportDir;program_log=$log;temporary_pl=$true;sd_boot_modified=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $output "program_$stamp.json") -Encoding utf8
Write-Output 'JTAG_COMMAND_COMPLETED; run the board numerical test to verify functionality.'
