param([int]$IntervalSeconds = 30)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$artifactRoot = Join-Path $projectRoot 'artifacts\week8'
$statusPath = Join-Path $artifactRoot 'bridge_overnight_pipeline_2026-10-07.json'
$outputPath = Join-Path $artifactRoot 'bridge_live_progress_2026-10-07.json'

while ($true) {
    $horizonDir = Join-Path $artifactRoot 'bridge_horizon_preoutcome_2026-10-07'
    $forecastDir = Join-Path $artifactRoot 'bridge_reduced_forecast_preoutcome_2026-10-07'
    $labelDir = Join-Path $artifactRoot 'bridge_reduced_test_labels_2026-10-07'
    $analysisDir = Join-Path $artifactRoot 'bridge_reduced_test_analysis_2026-10-07'
    $pipeline = if (Test-Path -LiteralPath $statusPath) {
        Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    } else { $null }
    $labelFiles = @(Get-ChildItem -LiteralPath (Join-Path $labelDir 'groups') -Filter '*.json' -ErrorAction SilentlyContinue)
    $failedLabels = 0
    foreach ($file in $labelFiles) {
        try {
            $item = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
        } catch {
            # A worker may still be writing this checkpoint; count it next cycle.
            continue
        }
        if ($item.status -ne 'BRIDGE_TEST_LABEL_COMPLETE') { $failedLabels++ }
    }
    $progress = [ordered]@{
        updated_local = (Get-Date).ToString('o')
        pipeline_state = if ($pipeline) { $pipeline.state } else { 'UNKNOWN' }
        pipeline_step = if ($pipeline) { $pipeline.detail } else { 'No pipeline status file' }
        horizon_groups_complete = @(Get-ChildItem -LiteralPath (Join-Path $horizonDir 'groups') -Filter '*.json' -ErrorAction SilentlyContinue).Count
        horizon_groups_expected = 70
        horizon_sealed = Test-Path -LiteralPath (Join-Path $horizonDir 'manifest.json')
        reduced_forecast_sealed = Test-Path -LiteralPath (Join-Path $forecastDir 'manifest.json')
        label_groups_complete = $labelFiles.Count - $failedLabels
        label_groups_failed = $failedLabels
        label_groups_expected = 70
        labels_sealed = Test-Path -LiteralPath (Join-Path $labelDir 'manifest.json')
        analysis_complete = Test-Path -LiteralPath (Join-Path $analysisDir 'manifest.json')
        test_results_available = Test-Path -LiteralPath (Join-Path $analysisDir 'RESULTS.md')
    }
    $progress | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $outputPath -Encoding UTF8
    if ($progress.analysis_complete -or ($progress.pipeline_state -eq 'NO_GO')) { break }
    Start-Sleep -Seconds $IntervalSeconds
}
