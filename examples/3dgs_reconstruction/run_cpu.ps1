param(
    [string]$Video = "$PSScriptRoot\data\nyu_snippet_curl.mp4",
    [string]$RunName = ("cpu_" + (Get-Date -Format 'yyyyMMddTHHmmss')),
    [int]$Iterations = 1000,
    [int]$MaxGaussians = 20000,
    [int]$Frames = 30
)
$ErrorActionPreference = 'Stop'
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunName must be a directory name, not a path' }
$py = "$PSScriptRoot\.venv\Scripts\python.exe"
$exe = "$PSScriptRoot\vendor\OpenSplat\build_cpu\opensplat.exe"
if (!(Test-Path -LiteralPath $exe)) { throw 'Build CPU OpenSplat first; see SETUP.md' }
& $py "$PSScriptRoot\run_cpu.py" --video $Video --run "$PSScriptRoot\runs\$RunName" --opensplat $exe --iterations $Iterations --max-gaussians $MaxGaussians --frames $Frames
if ($LASTEXITCODE -ne 0) { throw "CPU pipeline failed ($LASTEXITCODE)" }
& $py "$PSScriptRoot\evaluate.py" --run "$PSScriptRoot\runs\$RunName"
if ($LASTEXITCODE -ne 0) { throw "Evaluation failed ($LASTEXITCODE)" }
