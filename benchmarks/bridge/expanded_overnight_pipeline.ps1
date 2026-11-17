param()

$ErrorActionPreference = 'Stop'
$rootDir = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $rootDir
$pythonExe = 'D:\anaconda3\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'

$week8 = Join-Path $rootDir 'artifacts\week8'
$panel = Join-Path $rootDir 'benchmarks\bridge\expanded_panel_2026-10-08'
$progress = Join-Path $week8 'bridge_expanded_pipeline_progress_2026-10-08.json'
$logFile = Join-Path $week8 'bridge_expanded_pipeline_2026-10-08.log'
$utf8 = New-Object System.Text.UTF8Encoding($false)

function Write-State([string]$phase, [string]$detail) {
    $oehbDir = Join-Path $week8 'bridge_expanded_full_oehb_preoutcome_2026-10-08\groups'
    $horizonDir = Join-Path $week8 'bridge_expanded_horizon_preoutcome_2026-10-08\groups'
    $featureDir = Join-Path $week8 'bridge_expanded_pilot_features_preoutcome_2026-10-08\groups'
    $state = [ordered]@{
        phase = $phase
        detail = $detail
        supervisor_pid = $PID
        updated_local = (Get-Date).ToString('o')
        oehb_groups = @(Get-ChildItem -LiteralPath $oehbDir -Filter '*.json' -ErrorAction SilentlyContinue).Count
        horizon_groups = @(Get-ChildItem -LiteralPath $horizonDir -Filter '*.json' -ErrorAction SilentlyContinue).Count
        feature_groups = @(Get-ChildItem -LiteralPath $featureDir -Filter '*.json' -ErrorAction SilentlyContinue).Count
        total_groups = 25
        prelabel_seal = Test-Path -LiteralPath (Join-Path $panel 'prelabel_seal.json')
        labels_opened = Test-Path -LiteralPath (Join-Path $week8 'bridge_expanded_test_labels_2026-10-08\manifest.json')
        analysis_complete = Test-Path -LiteralPath (Join-Path $week8 'bridge_expanded_full_test_analysis_2026-10-08\manifest.json')
    }
    [System.IO.File]::WriteAllText($progress, (($state | ConvertTo-Json -Depth 4) + [Environment]::NewLine), $utf8)
}

function Run-Step([string]$module, [string]$manifest, [string[]]$extraArgs = @()) {
    if (Test-Path -LiteralPath $manifest) {
        Add-Content -LiteralPath $logFile -Value "[$((Get-Date).ToString('o'))] SKIP sealed $module" -Encoding UTF8
        return
    }
    Write-State $module 'Running; checkpoints are retained after interruption'
    Add-Content -LiteralPath $logFile -Value "[$((Get-Date).ToString('o'))] START $module" -Encoding UTF8
    & $pythonExe -m $module @extraArgs *>> $logFile
    if ($LASTEXITCODE -ne 0) { throw "$module exited with $LASTEXITCODE" }
    if (-not (Test-Path -LiteralPath $manifest)) { throw "$module did not write expected manifest: $manifest" }
    Add-Content -LiteralPath $logFile -Value "[$((Get-Date).ToString('o'))] PASS $module" -Encoding UTF8
    Write-State $module 'Sealed; moving to next stage'
}

try {
    $pilotManifest = Join-Path $rootDir 'datasets\bridge_v1_1_1\offline_pilot_expanded_2026-10-08\manifest.json'
    $reference = Join-Path $panel 'reference_preflight.json'
    if (-not (Test-Path -LiteralPath $pilotManifest) -or -not (Test-Path -LiteralPath $reference)) {
        throw 'Frozen pilot or exact-reference preflight is missing'
    }
    $steps = @(
        @('benchmarks.bridge.expanded_preoutcome_features', 'bridge_expanded_pilot_features_preoutcome_2026-10-08\manifest.json'),
        @('benchmarks.bridge.expanded_preoutcome_b2_forecast', 'bridge_expanded_b2_forecast_preoutcome_2026-10-08\manifest.json'),
        @('benchmarks.bridge.expanded_published_survival_forecast', 'bridge_expanded_published_xgboost_cf_preoutcome_2026-10-08\manifest.json'),
        @('benchmarks.bridge.expanded_preoutcome_horizon', 'bridge_expanded_horizon_preoutcome_2026-10-08\manifest.json'),
        @('benchmarks.bridge.expanded_preoutcome_oehb', 'bridge_expanded_full_oehb_preoutcome_2026-10-08\manifest.json'),
        @('benchmarks.bridge.expanded_original_scores_preoutcome', 'bridge_expanded_original_scores_preoutcome_2026-10-08\manifest.json'),
        @('benchmarks.bridge.expanded_preoutcome_full_forecast', 'bridge_expanded_full_forecast_preoutcome_2026-10-08\manifest.json'),
        @('benchmarks.bridge.expanded_published_survival_full_forecast', 'bridge_expanded_published_xgboost_cf_full108_preoutcome_2026-10-08\manifest.json')
    )
    foreach ($step in $steps) {
        $extra = @()
        if ($step[0] -eq 'benchmarks.bridge.expanded_preoutcome_horizon') {
            $extra = @('--workers', '1')
        } elseif ($step[0] -eq 'benchmarks.bridge.expanded_preoutcome_oehb') {
            # Independent groups use separate task-derived seeds and atomic checkpoints.
            # Two workers fit within the measured memory headroom on this machine.
            $extra = @('--workers', '2')
        }
        Run-Step $step[0] (Join-Path $week8 $step[1]) $extra
    }
    Run-Step 'benchmarks.bridge.expanded_prelabel_seal' (Join-Path $panel 'prelabel_seal.json')
    Run-Step 'benchmarks.bridge.expanded_run_test_labels' (Join-Path $week8 'bridge_expanded_test_labels_2026-10-08\manifest.json') @('--workers', '2')
    Run-Step 'benchmarks.bridge.expanded_score_test' (Join-Path $week8 'bridge_expanded_full_test_analysis_2026-10-08\manifest.json')
    Write-State 'COMPLETE' 'Five new BRIDGE tasks scored with frozen forecasters and one independent learner-label pass'
} catch {
    $message = $_.Exception.Message
    Add-Content -LiteralPath $logFile -Value "[$((Get-Date).ToString('o'))] NO_GO $message" -Encoding UTF8
    Write-State 'NO_GO' $message
    exit 1
}
