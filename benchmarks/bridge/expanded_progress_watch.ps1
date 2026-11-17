param([int]$PipelinePid = 0)

$ErrorActionPreference = 'Stop'
$rootDir = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$week8 = Join-Path $rootDir 'artifacts\week8'
$panel = Join-Path $rootDir 'benchmarks\bridge\expanded_panel_2026-10-08'
$source = Join-Path $week8 'bridge_expanded_pipeline_progress_2026-10-08.json'
$target = Join-Path $week8 'bridge_expanded_live_status_2026-10-08.json'
$utf8 = New-Object System.Text.UTF8Encoding($false)

function Count-Files([string]$subdir) {
    return @(Get-ChildItem -LiteralPath (Join-Path $week8 $subdir) -Filter '*.json' -ErrorAction SilentlyContinue).Count
}

while ($true) {
    $phase = 'NOT_STARTED'
    $sourcePid = 0
    if (Test-Path -LiteralPath $source) {
        try {
            $progress = Get-Content -LiteralPath $source -Raw | ConvertFrom-Json
            $phase = $progress.phase
            if ($progress.supervisor_pid) { $sourcePid = [int]$progress.supervisor_pid }
        }
        catch { $phase = 'STATUS_READ_RETRY' }
    }
    $trackedPid = if ($PipelinePid -gt 0) { $PipelinePid } else { $sourcePid }
    $running = ($trackedPid -gt 0) -and ($null -ne (Get-Process -Id $trackedPid -ErrorAction SilentlyContinue))
    $state = [ordered]@{
        updated_local = (Get-Date).ToString('o')
        phase = $phase
        pipeline_running = $running
        feature_groups = Count-Files 'bridge_expanded_pilot_features_preoutcome_2026-10-08\groups'
        horizon_groups = Count-Files 'bridge_expanded_horizon_preoutcome_2026-10-08\groups'
        oehb_groups = Count-Files 'bridge_expanded_full_oehb_preoutcome_2026-10-08\groups'
        original_score_tasks = Count-Files 'bridge_expanded_original_scores_preoutcome_2026-10-08\tasks'
        label_groups = Count-Files 'bridge_expanded_test_labels_2026-10-08\groups'
        total_groups = 25
        total_original_tasks = 5
        prelabel_seal = Test-Path -LiteralPath (Join-Path $panel 'prelabel_seal.json')
        labels_manifest = Test-Path -LiteralPath (Join-Path $week8 'bridge_expanded_test_labels_2026-10-08\manifest.json')
        analysis_manifest = Test-Path -LiteralPath (Join-Path $week8 'bridge_expanded_full_test_analysis_2026-10-08\manifest.json')
    }
    [System.IO.File]::WriteAllText($target, (($state | ConvertTo-Json -Depth 4) + [Environment]::NewLine), $utf8)
    if ($phase -in @('COMPLETE', 'NO_GO') -or -not $running) { break }
    Start-Sleep -Seconds 30
}
