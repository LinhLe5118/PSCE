param()

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $repo
$python = 'D:\anaconda3\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

$base = Join-Path $repo 'artifacts\week8'
$fresh = Join-Path $base 'bridge_expanded_full_test_analysis_2026-10-08\manifest.json'
$opened = Join-Path $base 'bridge_opened_minigrid_full_analysis_exploratory_2026-10-09\manifest.json'
$freshStatus = Join-Path $base 'bridge_expanded_live_status_2026-10-08.json'
$openedStatus = Join-Path $base 'bridge_opened_minigrid_queue_status_2026-10-09.json'
$status = Join-Path $base 'difficulty_common_metric_watch_2026-10-09.json'
$log = Join-Path $base 'difficulty_common_metric_watch_2026-10-09.log'
$utf8 = New-Object System.Text.UTF8Encoding($false)

function Write-State([string]$phase, [string]$detail) {
    $state = [ordered]@{
        phase = $phase
        detail = $detail
        updated_local = (Get-Date).ToString('o')
        prospective_analysis = Test-Path -LiteralPath $fresh
        opened_minigrid_analysis = Test-Path -LiteralPath $opened
        final_common_metric = Test-Path -LiteralPath (Join-Path $base 'difficulty_common_metric_final_2026-10-09\manifest.json')
    }
    [IO.File]::WriteAllText($status, (($state | ConvertTo-Json) + [Environment]::NewLine), $utf8)
}

try {
    while (-not ((Test-Path -LiteralPath $fresh) -and (Test-Path -LiteralPath $opened))) {
        foreach ($file in @($freshStatus, $openedStatus)) {
            if (Test-Path -LiteralPath $file) {
                $phase = (Get-Content -LiteralPath $file -Raw | ConvertFrom-Json).phase
                if ($phase -eq 'NO_GO') { throw "Upstream pipeline reported NO_GO: $file" }
            }
        }
        Write-State 'WAITING_FOR_BOTH_ANALYSES' 'Strict common-metric check requires both remaining analysis manifests'
        Start-Sleep -Seconds 60
    }
    Write-State 'VERIFYING' 'Recomputing seven-class scores and checking full source chain'
    & $python -m scripts.analysis.verify_common_difficulty_metrics *>> $log
    if ($LASTEXITCODE -ne 0) { throw "Common-metric verifier exited with $LASTEXITCODE" }
    $final = Join-Path $base 'difficulty_common_metric_final_2026-10-09\manifest.json'
    if (-not (Test-Path -LiteralPath $final)) { throw 'Final common-metric manifest missing' }
    $manifest = Get-Content -LiteralPath $final -Raw | ConvertFrom-Json
    if ($manifest.status -ne 'ALL_PANELS_COMMON_METRIC_VERIFIED') {
        throw "Unexpected final verifier status: $($manifest.status)"
    }
    Write-State 'COMPLETE' 'All four panels verified; RESULTS.md and paired/summary CSVs are ready for manuscript review'
} catch {
    $message = $_.Exception.Message
    Add-Content -LiteralPath $log -Value "[$((Get-Date).ToString('o'))] NO_GO $message" -Encoding UTF8
    Write-State 'NO_GO' $message
    exit 1
}
