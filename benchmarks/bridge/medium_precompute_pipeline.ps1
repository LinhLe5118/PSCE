param([Parameter(Mandatory=$true)][string]$PidFile,
      [Parameter(Mandatory=$true)][string]$LogFile)

$ErrorActionPreference = 'Stop'
$rootDir = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $rootDir
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& 'D:\anaconda3\python.exe' -u -m benchmarks.bridge.opened_minigrid_medium_spawn `
    --max-groups 5 --pid-file $PidFile *>> $LogFile
if ($LASTEXITCODE -ne 0) { throw "Medium precompute launcher exited with $LASTEXITCODE" }
