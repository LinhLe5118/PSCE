$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$pythonExe = 'D:\anaconda3\python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$statusPath = Join-Path $PSScriptRoot 'post_routing_status.json'
function Write-Status([string]$phase, [string]$detail) {
    @{ phase = $phase; detail = $detail; at_utc = [DateTime]::UtcNow.ToString('o') } |
        ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
}
function Run-Step([string]$phase, [string[]]$arguments, [string]$prefix) {
    Write-Status $phase ($arguments -join ' ')
    $process = Start-Process -FilePath $pythonExe -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot ($prefix + '.stdout.log')) -RedirectStandardError (Join-Path $PSScriptRoot ($prefix + '.stderr.log'))
    if ($process.ExitCode -ne 0) { throw "$phase exited $($process.ExitCode); inspect $prefix logs." }
}
try {
    Write-Status 'waiting_for_sealed_routing' 'Waiting for all 21,000 preserved job outcomes before analysis and E1 replication.'
    $deadline = [DateTime]::UtcNow.AddHours(48)
    $complete = Join-Path $PSScriptRoot 'execution_complete.json'
    $failure = Join-Path $PSScriptRoot 'failure_evaluate.json'
    $progress = Join-Path $PSScriptRoot 'execution/progress.json'
    while (-not (Test-Path -LiteralPath $complete)) {
        if (Test-Path -LiteralPath $failure) { throw 'Routing evaluator recorded a failure; no E1 follow-up was run.' }
        if ([DateTime]::UtcNow -gt $deadline) { throw 'Routing wait exceeded 48 hours; no E1 follow-up was run.' }
        if ((Test-Path -LiteralPath $progress) -and
                (([DateTime]::UtcNow - (Get-Item -LiteralPath $progress).LastWriteTimeUtc).TotalHours -gt 2)) {
            throw 'Routing progress stale for more than two hours; no E1 follow-up was run.'
        }
        Start-Sleep -Seconds 30
    }
    Run-Step 'routing_analyze' @('-m', 'scripts.analysis.run_unseen_validation', 'analyze') 'post_routing_analysis'
    if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'result_manifest.json'))) {
        throw 'Routing analysis returned without a result manifest.'
    }
    Run-Step 'e1_quality_prepare' @('-m', 'scripts.analysis.tabular_e1_quality_signals', 'prepare') 'e1_quality_prepare'
    Run-Step 'e1_quality_score' @('-m', 'scripts.analysis.tabular_e1_quality_signals', 'score') 'e1_quality_score'
    Write-Status 'complete' 'Routing result and E1 tabular quality-signal extraction/scoring manifests exist.'
} catch {
    Write-Status 'failed' $_.Exception.Message
    $_ | Out-String | Add-Content -LiteralPath (Join-Path $PSScriptRoot 'post_routing_error.log') -Encoding UTF8
    exit 1
}
