param(
    [string]$VivadoBin = 'D:\Xilinx\Vivado\2018.3\bin'
)
$ErrorActionPreference = 'Stop'

foreach ($tool in @('xvlog', 'xelab', 'xsim')) {
    if (-not (Test-Path -LiteralPath (Join-Path $VivadoBin "$tool.bat"))) {
        throw "Missing $tool.bat; specify the Vivado bin directory with -VivadoBin."
    }
}
if (-not $env:PROCESSOR_ARCHITECTURE -and [Environment]::Is64BitOperatingSystem) {
    $env:PROCESSOR_ARCHITECTURE = 'AMD64'
}
if (-not $env:NUMBER_OF_PROCESSORS) {
    $env:NUMBER_OF_PROCESSORS = [string][Environment]::ProcessorCount
}

$runDir = Join-Path $PSScriptRoot ('build\' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
New-Item -ItemType Directory -Path $runDir -Force | Out-Null
$rtlFile = (Join-Path $PSScriptRoot 'rtl\add_one.v').Replace('\', '/')
$testbenchFile = (Join-Path $PSScriptRoot 'sim\tb_add_one.sv').Replace('\', '/')
$simScript = (Join-Path $PSScriptRoot 'sim\run.tcl').Replace('\', '/')

function Invoke-SimTool {
    param([string]$ToolName, [string[]]$ToolArguments)
    & (Join-Path $VivadoBin "$ToolName.bat") @ToolArguments | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "$ToolName failed with exit code $LASTEXITCODE. Logs: $runDir"
    }
}

Push-Location -LiteralPath $runDir
try {
    Invoke-SimTool -ToolName xvlog -ToolArguments @(
        '--sv', '-work', 'xil_defaultlib', $rtlFile, $testbenchFile
    )
    Invoke-SimTool -ToolName xelab -ToolArguments @(
        'xil_defaultlib.tb_add_one', '-s', 'add_one_sim'
    )
    Invoke-SimTool -ToolName xsim -ToolArguments @(
        'add_one_sim', '-tclbatch', $simScript,
        '-log', 'simulation.log'
    )
    $log = Get-Content -LiteralPath 'simulation.log' -Raw
    if ($log -notmatch 'PASS: add_one \(\d+ checks\)' -or $log -match '(?im)^\s*(FAIL:|FATAL:|ERROR:)') {
        throw "Simulation did not pass. Inspect $runDir\simulation.log"
    }
    Write-Host ($log -split "`r?`n" | Where-Object { $_ -match '^PASS: add_one' })
    Write-Host "Logs: $runDir"
} finally {
    Pop-Location
}
