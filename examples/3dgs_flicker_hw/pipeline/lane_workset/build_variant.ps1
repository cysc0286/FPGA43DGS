param(
 [Parameter(Mandatory=$true)]
 [ValidateSet('control','dual-port','grouped-bram','grouped-registers')]
 [string]$Variant,
 [Parameter(Mandatory=$true)]
 [ValidatePattern('^[A-Za-z0-9_]+$')]
 [string]$Project
)
$ErrorActionPreference='Stop'
$root=(Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if(Test-Path (Join-Path $root "build\$Project")){throw 'Project already exists; use a new name'}
$names=@('FLK_HLS_PROJECT','FLK_EXACT_EXP_ROM','FLK_STATE_PORTS','FLK_GROUP_SUBTILES')
$saved=@{}
foreach($name in $names){$saved[$name]=[Environment]::GetEnvironmentVariable($name,'Process')}
try {
 $env:FLK_HLS_PROJECT=$Project
 $env:FLK_EXACT_EXP_ROM='2'
 $env:FLK_STATE_PORTS='1'
 $env:FLK_GROUP_SUBTILES='0'
 if($Variant -eq 'dual-port'){$env:FLK_STATE_PORTS='2'}
 if($Variant -eq 'grouped-bram'){$env:FLK_GROUP_SUBTILES='1'}
 if($Variant -eq 'grouped-registers'){$env:FLK_GROUP_SUBTILES='2'}
 & (Join-Path $PSScriptRoot '..\build.ps1')
} finally {
 foreach($name in $names){[Environment]::SetEnvironmentVariable($name,$saved[$name],'Process')}
}
