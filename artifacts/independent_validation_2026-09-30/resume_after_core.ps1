$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$pythonExe = 'D:\anaconda3\python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$statusPath = Join-Path $PSScriptRoot 'background_status.json'
function Write-Status([string]$phase, [string]$detail) {
    @{ phase = $phase; detail = $detail; at_utc = [DateTime]::UtcNow.ToString('o') } |
        ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
}
try {
    if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'choices_seal.json'))) {
        throw 'Frozen choices are missing.'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $repo 'artifacts/independent_difficulty_2026-09-30/result_manifest.json'))) {
        throw 'Core difficulty analysis has not completed.'
    }
    if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'execution_plan.json')) {
        throw 'Routing plan already exists; this pre-job recovery script must not resume a partial run.'
    }
    Write-Status 'evaluating' 'Frozen 21,000 matched method-seed jobs started after completed core analysis.'
    $run = Start-Process -FilePath $pythonExe -ArgumentList @('-m', 'scripts.analysis.run_unseen_validation', 'evaluate') -WorkingDirectory $repo -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot 'execution.log') -RedirectStandardError (Join-Path $PSScriptRoot 'execution.stderr.log')
    if ($run.ExitCode -ne 0) { throw "Evaluation exited $($run.ExitCode); see execution logs and failure_evaluate.json." }
    Write-Status 'analyzing' 'All routing jobs complete; fixed analysis started.'
    $analysis = Start-Process -FilePath $pythonExe -ArgumentList @('-m', 'scripts.analysis.run_unseen_validation', 'analyze') -WorkingDirectory $repo -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot 'analysis.log') -RedirectStandardError (Join-Path $PSScriptRoot 'analysis.stderr.log')
    if ($analysis.ExitCode -ne 0) { throw "Analysis exited $($analysis.ExitCode); see analysis logs." }
    Write-Status 'complete' 'Independent routing result_manifest.json is available.'
} catch {
    Write-Status 'failed' $_.Exception.Message
    $_ | Out-String | Add-Content -LiteralPath (Join-Path $PSScriptRoot 'recovery_error.log') -Encoding UTF8
    exit 1
}
