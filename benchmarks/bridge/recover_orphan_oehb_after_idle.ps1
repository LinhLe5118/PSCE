param(
    [int]$OrphanPid = 31260,
    [int]$ExpectedParentPid = 12444,
    [int]$SupervisorPid = 27804,
    [int]$IntervalSeconds = 60,
    [int]$IdleChecksBeforeRecovery = 6
)

$ErrorActionPreference = 'Stop'
if ($IntervalSeconds -lt 30 -or $IdleChecksBeforeRecovery -lt 5) {
    throw 'Recovery requires at least 30-second checks and five idle checks'
}
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$week8 = Join-Path $repo 'artifacts\week8'
$status = Join-Path $week8 'bridge_orphan_oehb_recovery_2026-10-10.json'
$progress = Join-Path $week8 'bridge_expanded_pipeline_progress_2026-10-08.json'
$oehbDir = Join-Path $week8 'bridge_expanded_full_oehb_preoutcome_2026-10-08\groups'
$manifest = Join-Path $week8 'bridge_expanded_full_oehb_preoutcome_2026-10-08\manifest.json'
$utf8 = New-Object System.Text.UTF8Encoding($false)
$previousCpu = $null
$idle = 0

function Write-State([string]$phase, [string]$detail, [double]$cpu) {
    $record = [ordered]@{
        phase = $phase
        detail = $detail
        checked_local = (Get-Date).ToString('o')
        orphan_pid = $OrphanPid
        former_parent_pid = $ExpectedParentPid
        supervisor_pid = $SupervisorPid
        orphan_cpu_seconds = [math]::Round($cpu, 1)
        idle_checks = $idle
        completed_groups = @(Get-ChildItem -LiteralPath $oehbDir -Filter '*.json' -ErrorAction SilentlyContinue).Count
        labels_opened = Test-Path -LiteralPath (Join-Path $week8 'bridge_expanded_test_labels_2026-10-08\manifest.json')
    }
    [IO.File]::WriteAllText($status, (($record | ConvertTo-Json) + [Environment]::NewLine), $utf8)
}

function Resume-From-Retained-Checkpoints([string]$reason, [double]$cpu) {
    Write-State 'RECOVERING_ORPHAN' $reason $cpu
    Stop-Process -Id $SupervisorPid -ErrorAction Stop
    if (Get-Process -Id $OrphanPid -ErrorAction SilentlyContinue) {
        Stop-Process -Id $OrphanPid -ErrorAction Stop
    }
    # The old status watcher is bound to the stopped supervisor PID. Let it
    # exit before the resume helper decides whether a new watcher is needed.
    Start-Sleep -Seconds 35
    & (Join-Path $PSScriptRoot 'resume_checkpointed_runs.ps1') -WithSmallMiniGrid | Out-Null
    Write-State 'RECOVERY_ISSUED' 'Restarted pipeline from retained group checkpoints' $cpu
}

while (-not (Test-Path -LiteralPath $manifest)) {
    $child = Get-CimInstance Win32_Process -Filter "ProcessId=$OrphanPid" -ErrorAction SilentlyContinue
    $parent = Get-CimInstance Win32_Process -Filter "ProcessId=$ExpectedParentPid" -ErrorAction SilentlyContinue
    $supervisor = Get-CimInstance Win32_Process -Filter "ProcessId=$SupervisorPid" -ErrorAction SilentlyContinue
    $stage = if (Test-Path -LiteralPath $progress) {
        (Get-Content -LiteralPath $progress -Raw | ConvertFrom-Json).phase
    } else { '' }
    if ($parent -or -not $supervisor -or
        $supervisor.CommandLine -notmatch 'expanded_overnight_pipeline\.ps1' -or
        $stage -ne 'benchmarks.bridge.expanded_preoutcome_oehb') {
        Write-State 'REVIEW_REQUIRED' 'Process identity or stage changed; no process stopped' 0
        exit 1
    }
    if (-not $child) {
        Resume-From-Retained-Checkpoints 'The orphan ended; resuming incomplete OEH-B stage' 0
        exit 0
    }
    if ($child.ParentProcessId -ne $ExpectedParentPid -or
        $child.CommandLine -notmatch 'multiprocessing\.spawn.*spawn_main') {
        Write-State 'REVIEW_REQUIRED' 'Orphan process identity changed; no process stopped' 0
        exit 1
    }
    $p = Get-Process -Id $OrphanPid -ErrorAction Stop
    $cpu = [double]$p.CPU
    if ($null -ne $previousCpu -and $cpu -le $previousCpu + 0.05) { $idle++ }
    else { $idle = 0 }
    $previousCpu = $cpu
    Write-State 'MONITORING_ACTIVE_ORPHAN' 'Current computation is preserved while CPU increases' $cpu
    if ($idle -ge $IdleChecksBeforeRecovery) {
        Resume-From-Retained-Checkpoints 'Parent is gone; child CPU stayed idle for six checks' $cpu
        exit 0
    }
    Start-Sleep -Seconds $IntervalSeconds
}
Write-State 'COMPLETE' 'OEH-B manifest appeared; no intervention required' 0
