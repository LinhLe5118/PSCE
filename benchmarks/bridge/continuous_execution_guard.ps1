param(
    [int]$IntervalSeconds = 120,
    [switch]$Once
)

$ErrorActionPreference = 'Stop'
if ($IntervalSeconds -lt 60) { throw 'IntervalSeconds must be at least 60' }
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$week8 = Join-Path $repo 'artifacts\week8'
$statusFile = Join-Path $week8 'continuous_execution_guard_2026-10-10.json'
$historyFile = Join-Path $week8 'continuous_execution_guard_2026-10-10.jsonl'
$utf8 = New-Object System.Text.UTF8Encoding($false)

function Read-Status([string]$name) {
    $path = Join-Path $week8 $name
    if (-not (Test-Path -LiteralPath $path)) { return $null }
    for ($attempt = 0; $attempt -lt 3; $attempt++) {
        try { return Get-Content -LiteralPath $path -Raw | ConvertFrom-Json }
        catch { Start-Sleep -Milliseconds 200 }
    }
    throw "Could not read status after three attempts: $name"
}

function Write-State([string]$phase, [string]$detail, [object[]]$actions) {
    $record = [ordered]@{
        phase = $phase
        detail = $detail
        checked_local = (Get-Date).ToString('o')
        guard_pid = $PID
        actions = @($actions)
        prospective_groups = @(Get-ChildItem -LiteralPath (Join-Path $week8 'bridge_expanded_full_oehb_preoutcome_2026-10-08\groups') -Filter '*.json' -ErrorAction SilentlyContinue).Count
        prospective_labels_opened = Test-Path -LiteralPath (Join-Path $week8 'bridge_expanded_test_labels_2026-10-08\manifest.json')
        common_metric_complete = Test-Path -LiteralPath (Join-Path $week8 'difficulty_common_metric_final_2026-10-09\manifest.json')
    }
    $json = ($record | ConvertTo-Json -Depth 6 -Compress) + [Environment]::NewLine
    [IO.File]::WriteAllText($statusFile, $json, $utf8)
    [IO.File]::AppendAllText($historyFile, $json, $utf8)
}

function Ensure-Watcher([string]$name, [object[]]$processes, [string]$extra = '') {
    $matches = @($processes | Where-Object {
        $_.CommandLine -match '(?i)-File\s+' -and $_.CommandLine.Contains($name)
    })
    if ($matches.Count -gt 1) { throw "Multiple active $name watchers; no action taken" }
    if ($matches.Count -eq 1) {
        return [ordered]@{ name = $name; state = 'running'; pid = [int]$matches[0].ProcessId }
    }
    if ($Once) { return [ordered]@{ name = $name; state = 'would_restart'; pid = 0 } }
    $path = Join-Path $PSScriptRoot $name
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing watcher $path" }
    $arguments = '-NoProfile -ExecutionPolicy Bypass -File "' + $path + '"' + $extra
    $job = Start-Process -FilePath 'powershell.exe' -ArgumentList $arguments `
        -WindowStyle Hidden -WorkingDirectory $repo -PassThru
    return [ordered]@{ name = $name; state = 'restarted'; pid = $job.Id }
}

try {
    while ($true) {
      try {
        $progress = Read-Status 'bridge_expanded_pipeline_progress_2026-10-08.json'
        $common = Read-Status 'difficulty_common_metric_watch_2026-10-09.json'
        $opened = Read-Status 'bridge_opened_minigrid_queue_status_2026-10-09.json'
        $manuscript = Read-Status 'difficulty_manuscript_watch_2026-10-09.json'
        if (($progress -and $progress.phase -eq 'NO_GO') -or
            ($common -and $common.phase -eq 'NO_GO') -or
            ($opened -and $opened.phase -eq 'NO_GO')) {
            Write-State 'NO_GO_REVIEW_REQUIRED' 'A scientific pipeline or audit recorded NO_GO; no automatic bypass' @()
            break
        }
        if ($manuscript -and $manuscript.phase -eq 'REVIEW_REQUIRED') {
            Write-State 'MANUSCRIPT_REVIEW_REQUIRED' 'Build or source generation needs inspection; no blind retry' @()
            break
        }
        if ($manuscript -and $manuscript.phase -eq 'COMPLETE_REVIEW_READY') {
            Write-State 'COMPLETE_REVIEW_READY' 'Verified results, paper and thesis builds completed' @()
            break
        }
        $processes = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction Stop)
        $actions = @()
        if (-not (Test-Path -LiteralPath (Join-Path $week8 'bridge_expanded_full_oehb_preoutcome_2026-10-08\manifest.json'))) {
            $actions += Ensure-Watcher 'watch_oehb_parent_health.ps1' $processes
        }
        if (-not (Test-Path -LiteralPath (Join-Path $week8 'bridge_expanded_full_test_analysis_2026-10-08\manifest.json'))) {
            $supervisor = if ($progress) { Get-Process -Id ([int]$progress.supervisor_pid) -ErrorAction SilentlyContinue } else { $null }
            if ($supervisor) {
                $actions += Ensure-Watcher 'expanded_progress_watch.ps1' $processes " -PipelinePid $($supervisor.Id)"
            }
        }
        if (-not (Test-Path -LiteralPath (Join-Path $week8 'bridge_opened_minigrid_full_analysis_exploratory_2026-10-09\manifest.json'))) {
            $actions += Ensure-Watcher 'opened_minigrid_queue.ps1' $processes
        }
        if (-not (Test-Path -LiteralPath (Join-Path $week8 'difficulty_common_metric_final_2026-10-09\manifest.json'))) {
            $actions += Ensure-Watcher 'final_common_metric_watch.ps1' $processes
            $actions += Ensure-Watcher 'watch_run_health.ps1' $processes
        }
        $actions += Ensure-Watcher 'final_manuscript_watch.ps1' $processes
        $actions += Ensure-Watcher 'keep_awake_until_complete.ps1' $processes
        Write-State 'MONITORING' 'Required watchers checked; absent watchers restarted without changing scientific sources' $actions
        if ($Once) { break }
        Start-Sleep -Seconds $IntervalSeconds
      } catch {
        Write-State 'GUARD_RETRYING' $_.Exception.Message @()
        if ($Once) { exit 1 }
        Start-Sleep -Seconds $IntervalSeconds
      }
    }
} catch {
    Write-State 'REVIEW_REQUIRED' $_.Exception.Message @()
    exit 1
}
