param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('core', 'external')]
    [string]$Phase,
    [ValidateRange(1, 32)]
    [int]$Shards = 8,
    [string]$ShardIndices = ''
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$output = Join-Path $repo 'artifacts/oeh_confirmatory_unseen_120_2026-10-04'
if (-not (Test-Path -LiteralPath (Join-Path $output 'preoutcome_seal.json'))) {
    throw 'Forecasts and learner choices must be sealed before evaluation.'
}
$python = (Get-Command python).Source
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$logDir = Join-Path $repo 'tmp/oeh_confirmatory_shards'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$phaseName = if ($Phase -eq 'core') { 'evaluate-core' } else { 'evaluate-external' }
$planName = if ($Phase -eq 'core') { 'execution_plan.json' } else { 'external_execution_plan.json' }
$indices = if ($ShardIndices) { @(($ShardIndices -split ',') | ForEach-Object { [int]$_ } | Sort-Object -Unique) } else { @(0..($Shards - 1)) }
if ($indices[0] -ne 0 -or $indices[-1] -ge $Shards) {
    throw 'Selected shard indices must include zero and be smaller than Shards.'
}

function Start-Shard([int]$index) {
    $arguments = @('-m', 'scripts.analysis.run_oeh_confirmatory_20261004',
                   $phaseName, '--shard', "$index", '--shards', "$Shards")
    $stdout = Join-Path $logDir "$Phase-$index.stdout.log"
    $stderr = Join-Path $logDir "$Phase-$index.stderr.log"
    $process = Start-Process -FilePath $python -ArgumentList $arguments `
        -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr -PassThru
    Write-Host "$Phase shard $index started: PID $($process.Id)"
    return $process
}

$processes = @(Start-Shard 0)
$plan = Join-Path $output $planName
$deadline = (Get-Date).AddMinutes(2)
while (-not (Test-Path -LiteralPath $plan)) {
    if ($processes[0].HasExited) { throw "Shard 0 exited before the shared plan: $($processes[0].ExitCode)" }
    if ((Get-Date) -ge $deadline) { throw "Timed out waiting for $plan" }
    Start-Sleep -Seconds 1
}
for ($index = 1; $index -lt $Shards; $index++) {
    if ($indices -contains $index) { $processes += Start-Shard $index }
}
for ($offset = 0; $offset -lt $processes.Count; $offset++) {
    $index = $indices[$offset]
    $process = $processes[$offset]
    $process.WaitForExit()
    if ($null -ne $process.ExitCode -and $process.ExitCode -ne 0) {
        throw "$Phase shard PID $($process.Id) exited with code $($process.ExitCode); inspect $logDir"
    }
    $stdout = Join-Path $logDir "$Phase-$index.stdout.log"
    $expected = if ($Phase -eq 'core') { "Core shard $index/$Shards complete" } else { "External shard $index/$Shards complete" }
    if (-not (Select-String -LiteralPath $stdout -SimpleMatch $expected -Quiet)) {
        throw "$Phase shard $index did not write its completion marker; inspect $logDir"
    }
}
Write-Host "All $($indices.Count) selected $Phase shards completed."
