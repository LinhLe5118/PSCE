param(
    [int]$IntervalSeconds = 60,
    [int]$IdleChecksBeforeRecovery = 6,
    [int]$MissingWorkerChecksBeforeRecovery = 3
)

$ErrorActionPreference = 'Stop'
if ($IntervalSeconds -lt 30 -or $IdleChecksBeforeRecovery -lt 5 -or
    $MissingWorkerChecksBeforeRecovery -lt 2) {
    throw 'Unsafe OEH-B watcher intervals'
}
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$week8 = Join-Path $repo 'artifacts\week8'
$progressFile = Join-Path $week8 'bridge_expanded_pipeline_progress_2026-10-08.json'
$oehbDir = Join-Path $week8 'bridge_expanded_full_oehb_preoutcome_2026-10-08\groups'
$oehbManifest = Join-Path $week8 'bridge_expanded_full_oehb_preoutcome_2026-10-08\manifest.json'
$labelsManifest = Join-Path $week8 'bridge_expanded_test_labels_2026-10-08\manifest.json'
$mediumGuardStatus = Join-Path $week8 'bridge_opened_minigrid_medium_guard_2026-10-10.json'
$statusFile = Join-Path $week8 'bridge_oehb_parent_health_2026-10-10.json'
$historyFile = Join-Path $week8 'bridge_oehb_parent_health_2026-10-10.jsonl'
$utf8 = New-Object System.Text.UTF8Encoding($false)
$trackedWorkerPids = @()
$previousCpu = $null
$idleChecks = 0
$missingChecks = 0

function Write-State([string]$phase, [string]$detail, [int]$supervisorPid,
                     [int]$workerPid, [double]$workerCpu, [int[]]$workerPids = @()) {
    $record = [ordered]@{
        phase = $phase
        detail = $detail
        checked_local = (Get-Date).ToString('o')
        supervisor_pid = $supervisorPid
        worker_pid = $workerPid
        worker_pids = @($workerPids)
        worker_cpu_seconds = [math]::Round($workerCpu, 1)
        idle_checks = $idleChecks
        missing_worker_checks = $missingChecks
        completed_groups = @(Get-ChildItem -LiteralPath $oehbDir -Filter '*.json' -ErrorAction SilentlyContinue).Count
        labels_opened = Test-Path -LiteralPath $labelsManifest
    }
    $line = ($record | ConvertTo-Json -Compress) + [Environment]::NewLine
    [IO.File]::WriteAllText($statusFile, $line, $utf8)
    [IO.File]::AppendAllText($historyFile, $line, $utf8)
}

function Resume-IncompleteStage([int]$supervisorPid, [int[]]$workerPids, [string]$reason) {
    if (Test-Path -LiteralPath $oehbManifest) { return }
    if (Test-Path -LiteralPath $labelsManifest) { throw 'Labels opened before OEH-B manifest; refusing recovery' }
    $progress = Get-Content -LiteralPath $progressFile -Raw | ConvertFrom-Json
    if ($progress.phase -ne 'benchmarks.bridge.expanded_preoutcome_oehb' -or
        [int]$progress.supervisor_pid -ne $supervisorPid) {
        throw 'OEH-B stage or supervisor changed during recovery check'
    }
    $supervisor = Get-CimInstance Win32_Process -Filter "ProcessId=$supervisorPid" -ErrorAction SilentlyContinue
    if (-not $supervisor -or
        $supervisor.CommandLine -notmatch 'expanded_overnight_pipeline\.ps1') {
        throw 'Supervisor identity changed; no process stopped'
    }
    foreach ($workerPid in $workerPids) {
        $worker = Get-CimInstance Win32_Process -Filter "ProcessId=$workerPid" -ErrorAction SilentlyContinue
        if ($worker -and $worker.CommandLine -notmatch 'multiprocessing\.spawn.*spawn_main') {
            throw 'Worker identity changed; no process stopped'
        }
    }
    $firstWorkerPid = if ($workerPids.Count) { [int]$workerPids[0] } else { 0 }
    Write-State 'RECOVERING' $reason $supervisorPid $firstWorkerPid 0 $workerPids
    Stop-Process -Id $supervisorPid -ErrorAction Stop
    foreach ($workerPid in $workerPids) {
        if (Get-Process -Id $workerPid -ErrorAction SilentlyContinue) {
            Stop-Process -Id $workerPid -ErrorAction Stop
        }
    }
    # The old progress watcher is tied to the stopped supervisor and exits
    # within 30 seconds. A new supervisor then resumes from atomic group files.
    Start-Sleep -Seconds 35
    & (Join-Path $PSScriptRoot 'resume_checkpointed_runs.ps1') | Out-Null
    Write-State 'RECOVERY_ISSUED' 'Checkpoint resume helper started the missing supervisor' 0 0 0
}

try {
    while (-not (Test-Path -LiteralPath $oehbManifest)) {
        if (-not (Test-Path -LiteralPath $progressFile)) {
            Write-State 'WAITING_FOR_STAGE' 'No prospective pipeline progress file yet' 0 0 0
            Start-Sleep -Seconds $IntervalSeconds
            continue
        }
        $progress = Get-Content -LiteralPath $progressFile -Raw | ConvertFrom-Json
        if ($progress.phase -eq 'NO_GO') { throw 'Prospective pipeline has a recorded NO_GO' }
        if ($progress.phase -ne 'benchmarks.bridge.expanded_preoutcome_oehb') {
            Write-State 'WAITING_FOR_STAGE' "Current stage: $($progress.phase)" 0 0 0
            Start-Sleep -Seconds $IntervalSeconds
            continue
        }
        $supervisorPid = [int]$progress.supervisor_pid
        $supervisor = Get-CimInstance Win32_Process -Filter "ProcessId=$supervisorPid" -ErrorAction SilentlyContinue
        if (-not $supervisor) {
            Write-State 'SUPERVISOR_MISSING' 'Resume helper will restore the missing supervisor' $supervisorPid 0 0
            & (Join-Path $PSScriptRoot 'resume_checkpointed_runs.ps1') | Out-Null
            Start-Sleep -Seconds $IntervalSeconds
            continue
        }
        if ($supervisor.CommandLine -notmatch 'expanded_overnight_pipeline\.ps1') {
            throw 'Progress supervisor PID belongs to another process'
        }
        $python = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction Stop)
        $parents = @($python | Where-Object { $_.CommandLine -match '(?i)-m\s+benchmarks\.bridge\.expanded_preoutcome_oehb' })
        if ($parents.Count -gt 1) { throw 'Multiple OEH-B parent runners; inspect before recovery' }
        if ($parents.Count -eq 1) {
            if (@($trackedWorkerPids | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }).Count) {
                throw 'OEH-B parent restarted while an orphan worker is still active'
            }
            $trackedWorkerPids = @()
            $previousCpu = $null
            $idleChecks = 0
            $missingChecks = 0
            Write-State 'NORMAL_PARENT_ACTIVE' 'OEH-B parent runner is active' $supervisorPid 0 0
            Start-Sleep -Seconds $IntervalSeconds
            continue
        }
        $stageTime = [datetime]$progress.updated_local
        $mediumChildPid = 0
        if (Test-Path -LiteralPath $mediumGuardStatus) {
            $mediumGuard = Get-Content -LiteralPath $mediumGuardStatus -Raw | ConvertFrom-Json
            if ($mediumGuard.phase -eq 'RUNNING' -and [int]$mediumGuard.child_pid -gt 0) {
                $mediumChildPid = [int]$mediumGuard.child_pid
            }
        }
        # Guard state can lag a newly spawned MiniGrid child during a task
        # transition. Child records are written at spawn time, so exclude all
        # recorded live exploratory children before classifying OEH-B orphans.
        $mediumChildPids = @($mediumChildPid)
        foreach ($childRecordFile in @(Get-ChildItem -LiteralPath $week8 -Filter 'bridge_medium_child_*.json' -ErrorAction SilentlyContinue)) {
            try {
                $childRecord = Get-Content -LiteralPath $childRecordFile.FullName -Raw | ConvertFrom-Json
                if ([int]$childRecord.child_pid -gt 0) {
                    $mediumChildPids += [int]$childRecord.child_pid
                }
            } catch {
                # A child record being written is retried on the next poll.
            }
        }
        $orphans = @($python | Where-Object {
            $_.CommandLine -match 'multiprocessing\.spawn.*spawn_main' -and
            [int]$_.ProcessId -notin $mediumChildPids -and
            [datetime]$_.CreationDate -ge $stageTime.AddMinutes(-5)
        })
        if ($orphans.Count -gt 2) { throw 'More than two possible orphan workers; no process stopped' }
        if ($orphans.Count -ge 1) {
            $workerPids = @($orphans | ForEach-Object { [int]$_.ProcessId } | Sort-Object)
            $workers = @($workerPids | ForEach-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
            if ($workers.Count -ne $workerPids.Count) { Start-Sleep -Seconds $IntervalSeconds; continue }
            $cpu = [double](($workers | Measure-Object -Property CPU -Sum).Sum)
            if ((@($trackedWorkerPids) -join ',') -ne ($workerPids -join ',')) {
                $previousCpu = $null; $idleChecks = 0
            }
            $trackedWorkerPids = $workerPids
            # An idle multiprocessing child may consume a few scheduler ticks
            # while waiting on its dead parent. Count only meaningful CPU work
            # as progress, and retain the baseline across idle polls.
            if ($null -ne $previousCpu -and $cpu -le $previousCpu + 2.0) {
                $idleChecks++
            } else {
                $idleChecks = 0
                $previousCpu = $cpu
            }
            $missingChecks = 0
            Write-State 'MONITORING_ACTIVE_ORPHAN' 'Workers write atomic group checkpoints; aggregate CPU is monitored' $supervisorPid $workerPids[0] $cpu $workerPids
            if ($idleChecks -ge $IdleChecksBeforeRecovery) {
                Resume-IncompleteStage $supervisorPid $workerPids 'Orphan CPU idle for six checks'
                $trackedWorkerPids = @(); $previousCpu = $null; $idleChecks = 0
            }
        } else {
            $missingChecks++
            $lastWorkerPid = if ($trackedWorkerPids.Count) { [int]$trackedWorkerPids[0] } else { 0 }
            Write-State 'PARENT_AND_WORKER_MISSING' 'Waiting briefly before checkpoint resume' $supervisorPid $lastWorkerPid 0 $trackedWorkerPids
            if ($missingChecks -ge $MissingWorkerChecksBeforeRecovery) {
                Resume-IncompleteStage $supervisorPid @() 'OEH-B parent and workers all absent'
                $trackedWorkerPids = @(); $previousCpu = $null; $missingChecks = 0
            }
        }
        Start-Sleep -Seconds $IntervalSeconds
    }
    Write-State 'COMPLETE' 'OEH-B manifest sealed; watcher finished' 0 0 0
} catch {
    $lastWorkerPid = if ($trackedWorkerPids.Count) { [int]$trackedWorkerPids[0] } else { 0 }
    Write-State 'REVIEW_REQUIRED' $_.Exception.Message 0 $lastWorkerPid 0 $trackedWorkerPids
    exit 1
}
