param()

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$week8 = Join-Path $repo 'artifacts\week8'

function Read-State([string]$name) {
    $file = Join-Path $week8 $name
    if (Test-Path -LiteralPath $file) {
        return Get-Content -LiteralPath $file -Raw | ConvertFrom-Json
    }
    return $null
}

$prospective = Read-State 'bridge_expanded_live_status_2026-10-08.json'
$opened = Read-State 'bridge_opened_minigrid_queue_status_2026-10-09.json'
$common = Read-State 'difficulty_common_metric_watch_2026-10-09.json'
$manuscript = Read-State 'difficulty_manuscript_watch_2026-10-09.json'
$awake = Read-State 'bridge_keep_awake_status_2026-10-09.json'
$health = Read-State 'bridge_run_health_2026-10-09.json'
$parentHealth = Read-State 'bridge_oehb_parent_health_2026-10-10.json'
$continuousGuard = Read-State 'continuous_execution_guard_2026-10-10.json'
$orphan = Read-State 'bridge_orphan_oehb_recovery_2026-10-10.json'
$mediumChain = Read-State 'bridge_opened_minigrid_medium_chain_2026-10-10.json'
$mediumGuard = Read-State 'bridge_opened_minigrid_medium_guard_2026-10-10.json'
$openedGroups = @(Get-ChildItem -LiteralPath (Join-Path $week8 'bridge_opened_minigrid_full_oehb_exploratory_2026-10-09\groups') -Filter '*.json' -ErrorAction SilentlyContinue).Count
$horizonGroups = @(Get-ChildItem -LiteralPath (Join-Path $week8 'bridge_expanded_horizon_preoutcome_2026-10-08\groups') -Filter '*.json' -ErrorAction SilentlyContinue).Count
$oehbGroups = @(Get-ChildItem -LiteralPath (Join-Path $week8 'bridge_expanded_full_oehb_preoutcome_2026-10-08\groups') -Filter '*.json' -ErrorAction SilentlyContinue).Count
$labelGroups = @(Get-ChildItem -LiteralPath (Join-Path $week8 'bridge_expanded_test_labels_2026-10-08\groups') -Filter '*.json' -ErrorAction SilentlyContinue).Count
$internal = Test-Path -LiteralPath (Join-Path $repo 'artifacts\oeh_confirmatory_unseen_120_2026-10-04\result_manifest.json')
$atari = Test-Path -LiteralPath (Join-Path $week8 'bridge_fresh_full_test_analysis_2026-10-08\manifest.json')

Write-Output "Checked: $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss zzz'))"
Write-Output "Internal 120 MDP: $(if ($internal) {'COMPLETE'} else {'MISSING'})"
Write-Output "BRIDGE Atari 5 MDP: $(if ($atari) {'COMPLETE'} else {'MISSING'})"
if ($prospective) {
    $liveAge = ((Get-Date) - [datetime]$prospective.updated_local).TotalSeconds
    $running = if ($liveAge -gt 90) {'STALE_STATUS'} else {[string]$prospective.pipeline_running}
    Write-Output "BRIDGE new 5 MDP: $($prospective.phase); running=$running; pilot=$($prospective.feature_groups)/25; OEH-H=$horizonGroups/25; OEH-B=$oehbGroups/25; original=$($prospective.original_score_tasks)/5; labels=$labelGroups/25; seal=$($prospective.prelabel_seal); analysis=$($prospective.analysis_manifest)"
} else {
    Write-Output 'BRIDGE new 5 MDP: no live status'
}
Write-Output "BRIDGE opened MiniGrid 14 MDP: $openedGroups/70 OEH-B groups; queue=$(if ($opened) {$opened.phase} else {'NO_STATUS'}); analysis=$(if ($opened) {$opened.analysis} else {$false})"
if ($mediumChain) {
    $chainAge = ((Get-Date) - [datetime]$mediumChain.checked_local).TotalSeconds
    $chainPhase = if ($mediumChain.phase -like 'COMPLETE*') {$mediumChain.phase} elseif ($chainAge -gt 150) {'STALE_STATUS'} else {$mediumChain.phase}
    $guardPhase = if (-not $mediumGuard) {'NO_STATUS'} elseif ($mediumGuard.phase -like 'COMPLETE*') {$mediumGuard.phase} elseif (((Get-Date) - [datetime]$mediumGuard.checked_local).TotalSeconds -gt 150) {'STALE_STATUS'} else {$mediumGuard.phase}
    Write-Output "MiniGrid guarded precompute: chain=$chainPhase; guard=$guardPhase; child=$(if ($guardPhase -eq 'RUNNING') {$mediumGuard.child_pid} else {0})"
}
Write-Output "Common-metric audit: $(if ($common) {$common.phase} else {'NO_STATUS'})"
Write-Output "Paper/thesis update: $(if ($manuscript) {$manuscript.phase} else {'NO_STATUS'})"
Write-Output "Continuous guard: $(if (-not $continuousGuard) {'NO_STATUS'} elseif (((Get-Date) - [datetime]$continuousGuard.checked_local).TotalSeconds -gt 300) {'STALE_STATUS'} else {$continuousGuard.phase}); pid=$(if ($continuousGuard) {$continuousGuard.guard_pid} else {0})"
Write-Output "Keep-awake request: $(if (-not $awake) {'NO_STATUS'} elseif (((Get-Date) - [datetime]$awake.updated_local).TotalSeconds -gt 90) {'STALE_STATUS'} else {$awake.phase})"
if ($health) {
    $healthAge = ((Get-Date) - [datetime]$health.checked_local).TotalSeconds
    $healthState = if ($healthAge -gt 150) {'STALE_STATUS'} elseif (@($health.warnings).Count) { @($health.warnings) -join ',' } else {'OK'}
    Write-Output "Health: $healthState; free RAM=$($health.free_ram_gib) GiB; D:=$($health.free_d_gib) GiB; workers=$(@($health.workers).Count)"
}
if ($parentHealth) {
    $parentAge = ((Get-Date) - [datetime]$parentHealth.checked_local).TotalSeconds
    $parentPhase = if ($parentAge -gt 150) {'STALE_STATUS'} else {$parentHealth.phase}
    Write-Output "OEH-B parent recovery: $parentPhase; checkpoints=$oehbGroups/25; worker=$($parentHealth.worker_pid); idle_checks=$($parentHealth.idle_checks)"
} elseif ($orphan) {
    Write-Output "OEH-B orphan recovery: $($orphan.phase); checkpoints=$($orphan.completed_groups)/25; idle_checks=$($orphan.idle_checks)"
}
