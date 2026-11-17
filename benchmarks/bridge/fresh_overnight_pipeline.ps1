param()

$ErrorActionPreference = 'Stop'
$rootDir = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $rootDir
$pythonExe = 'D:\anaconda3\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$week8 = Join-Path $rootDir 'artifacts\week8'
$progress = Join-Path $week8 'bridge_fresh_overnight_progress_2026-10-08.json'
$logFile = Join-Path $week8 'bridge_fresh_overnight_pipeline_2026-10-08.log'
$start = Get-Date
$deadline = $start.AddHours(72)

function Write-State([string]$phase, [string]$detail) {
    $oehbDir = Join-Path $week8 'bridge_fresh_full_oehb_preoutcome_2026-10-08\groups'
    $horizonDir = Join-Path $week8 'bridge_fresh_horizon_preoutcome_2026-10-08\groups'
    $oehbCount = @(Get-ChildItem -LiteralPath $oehbDir -Filter '*.json' -ErrorAction SilentlyContinue).Count
    $horizonCount = @(Get-ChildItem -LiteralPath $horizonDir -Filter '*.json' -ErrorAction SilentlyContinue).Count
    $state = [ordered]@{
        phase = $phase
        detail = $detail
        updated_local = (Get-Date).ToString('o')
        oehb_groups = $oehbCount
        horizon_groups = $horizonCount
        total_groups = 25
        labels_opened = (Test-Path -LiteralPath (Join-Path $week8 'bridge_fresh_test_labels_2026-10-08\manifest.json'))
        analysis_complete = (Test-Path -LiteralPath (Join-Path $week8 'bridge_fresh_full_test_analysis_2026-10-08\manifest.json'))
    }
    $serialized = ($state | ConvertTo-Json -Depth 4) + [Environment]::NewLine
    for ($attempt = 1; $attempt -le 10; $attempt++) {
        try {
            [System.IO.File]::WriteAllText($progress, $serialized, [System.Text.Encoding]::UTF8)
            return
        } catch [System.IO.IOException] {
            if ($attempt -eq 10) { throw }
            Start-Sleep -Milliseconds 250
        }
    }
}

function Run-Step([string]$module) {
    Add-Content -LiteralPath $logFile -Value "[$((Get-Date).ToString('o'))] START $module" -Encoding UTF8
    & $pythonExe -m $module *>> $logFile
    if ($LASTEXITCODE -ne 0) {
        throw "$module exited with $LASTEXITCODE"
    }
    Add-Content -LiteralPath $logFile -Value "[$((Get-Date).ToString('o'))] PASS $module" -Encoding UTF8
}

try {
    Write-State 'WAITING_PREOUTCOME' 'Waiting for OEH-B and OEH-H manifests; no labels'
    while ($true) {
        $oehbManifest = Join-Path $week8 'bridge_fresh_full_oehb_preoutcome_2026-10-08\manifest.json'
        $horizonManifest = Join-Path $week8 'bridge_fresh_horizon_preoutcome_2026-10-08\manifest.json'
        $featureManifest = Join-Path $week8 'bridge_fresh_pilot_features_preoutcome_2026-10-08\manifest.json'
        if ((Test-Path -LiteralPath $oehbManifest) -and (Test-Path -LiteralPath $horizonManifest) -and (Test-Path -LiteralPath $featureManifest)) { break }
        foreach ($failure in @(
            (Join-Path $week8 'bridge_fresh_full_oehb_preoutcome_2026-10-08\failure_report.json'),
            (Join-Path $week8 'bridge_fresh_horizon_preoutcome_2026-10-08\failure_report.json')
        )) {
            if (Test-Path -LiteralPath $failure) { throw "Preoutcome failure report: $failure" }
        }
        if ((Get-Date) -gt $deadline) { throw '72-hour resource cap exceeded before preoutcome seal' }
        Write-State 'WAITING_PREOUTCOME' 'OEH-B and OEH-H in progress; no labels'
        Start-Sleep -Seconds 30
    }
    Write-State 'FULL_FORECAST' 'All pilot descriptors sealed; fitting frozen full OEH meta-model'
    Run-Step 'benchmarks.bridge.fresh_preoutcome_full_forecast'
    while (-not ((Test-Path -LiteralPath (Join-Path $week8 'bridge_fresh_b2_forecast_preoutcome_2026-10-08\manifest.json')) -and
                  (Test-Path -LiteralPath (Join-Path $week8 'bridge_fresh_original_scores_preoutcome_2026-10-08\manifest.json')))) {
        if ((Get-Date) -gt $deadline) { throw '72-hour resource cap exceeded before all comparators sealed' }
        Write-State 'WAITING_COMPARATORS' 'Full OEH sealed; waiting for B2 and original scalar scores'
        Start-Sleep -Seconds 30
    }
    Write-State 'PRELABEL_SEAL' 'Auditing all 75+75 forecasts and 25 original-score rows'
    Run-Step 'benchmarks.bridge.fresh_prelabel_seal'
    Write-State 'LABELS' 'Prelabel seal passed; one independent label run begins'
    Add-Content -LiteralPath $logFile -Value "[$((Get-Date).ToString('o'))] START fresh_run_test_labels" -Encoding UTF8
    & $pythonExe -m benchmarks.bridge.fresh_run_test_labels --workers 2 *>> $logFile
    if ($LASTEXITCODE -ne 0) { throw "fresh_run_test_labels exited with $LASTEXITCODE" }
    Add-Content -LiteralPath $logFile -Value "[$((Get-Date).ToString('o'))] PASS fresh_run_test_labels" -Encoding UTF8
    Write-State 'SCORING' 'Learner labels sealed; scoring full OEH and original rank signals'
    Run-Step 'benchmarks.bridge.fresh_score_test'
    Write-State 'COMPLETE' 'Fresh full OEH BRIDGE analysis complete'
} catch {
    $message = $_.Exception.Message
    Add-Content -LiteralPath $logFile -Value "[$((Get-Date).ToString('o'))] NO_GO $message" -Encoding UTF8
    Write-State 'NO_GO' $message
    exit 1
}
