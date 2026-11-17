param(
    [switch]$DryRun,
    [switch]$WithSmallMiniGrid
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$week8 = Join-Path $repo 'artifacts\week8'
$python = 'D:\anaconda3\python.exe'
$panel = Join-Path $repo 'benchmarks\bridge\expanded_panel_2026-10-08'
$prospectiveResult = Join-Path $week8 'bridge_expanded_full_test_analysis_2026-10-08\manifest.json'
$openedResult = Join-Path $week8 'bridge_opened_minigrid_full_analysis_exploratory_2026-10-09\manifest.json'
$progressFile = Join-Path $week8 'bridge_expanded_pipeline_progress_2026-10-08.json'
$processes = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe' OR Name='python.exe'")
$record = [ordered]@{ checked_local = (Get-Date).ToString('o'); dry_run = [bool]$DryRun; jobs = @() }

function Existing-PowerShell([string]$name) {
    $matches = @($processes | Where-Object {
        $_.Name -ieq 'powershell.exe' -and
        $_.CommandLine -match '(?i)-File\s+' -and
        $_.CommandLine.Contains($name)
    })
    if ($matches.Count -gt 1) { throw "Multiple active $name processes; inspect before resuming" }
    return $matches | Select-Object -First 1
}

function Ensure-Script([string]$name, [string]$extra = '') {
    $existing = Existing-PowerShell $name
    if ($existing) {
        $record.jobs += [ordered]@{ name = $name; state = 'already_running'; pid = [int]$existing.ProcessId }
        return [int]$existing.ProcessId
    }
    if ($DryRun) {
        $record.jobs += [ordered]@{ name = $name; state = 'would_start'; pid = $null }
        return 0
    }
    $path = Join-Path $PSScriptRoot $name
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing script $path" }
    $arguments = '-NoProfile -ExecutionPolicy Bypass -File "' + $path + '"' + $extra
    $job = Start-Process -FilePath 'powershell.exe' -ArgumentList $arguments -WindowStyle Hidden -WorkingDirectory $repo -PassThru
    $record.jobs += [ordered]@{ name = $name; state = 'started'; pid = $job.Id }
    return [int]$job.Id
}

if (-not (Test-Path -LiteralPath $python)) { throw "Missing Python executable $python" }
$phase = if (Test-Path -LiteralPath $progressFile) {
    (Get-Content -LiteralPath $progressFile -Raw | ConvertFrom-Json).phase
} else { 'NOT_STARTED' }
if ($phase -eq 'NO_GO') { throw 'Prospective pipeline has a NO_GO record; inspect the failure before resuming' }

$pipelinePid = 0
if (-not (Test-Path -LiteralPath $prospectiveResult)) {
    $pipelinePid = Ensure-Script 'expanded_overnight_pipeline.ps1'
    $watcher = Existing-PowerShell 'expanded_progress_watch.ps1'
    if ($watcher) {
        $record.jobs += [ordered]@{ name = 'expanded_progress_watch.ps1'; state = 'already_running'; pid = [int]$watcher.ProcessId }
    } elseif ($pipelinePid -gt 0) {
        $null = Ensure-Script 'expanded_progress_watch.ps1' (" -PipelinePid $pipelinePid")
    } elseif ($DryRun) {
        $record.jobs += [ordered]@{ name = 'expanded_progress_watch.ps1'; state = 'would_start_after_pipeline'; pid = $null }
    }
}

if (-not (Test-Path -LiteralPath $openedResult)) {
    $null = Ensure-Script 'opened_minigrid_queue.ps1'
}
if (-not (Test-Path -LiteralPath (Join-Path $week8 'difficulty_common_metric_final_2026-10-09\manifest.json'))) {
    $null = Ensure-Script 'final_common_metric_watch.ps1'
}
$manuscriptStatus = Join-Path $week8 'difficulty_manuscript_watch_2026-10-09.json'
$manuscriptComplete = if (Test-Path -LiteralPath $manuscriptStatus) {
    (Get-Content -LiteralPath $manuscriptStatus -Raw | ConvertFrom-Json).phase -eq 'COMPLETE_REVIEW_READY'
} else { $false }
if (-not $manuscriptComplete) {
    $null = Ensure-Script 'final_manuscript_watch.ps1'
}
if (-not ((Test-Path -LiteralPath $prospectiveResult) -and (Test-Path -LiteralPath $openedResult))) {
    $null = Ensure-Script 'keep_awake_until_complete.ps1'
}
if (-not (Test-Path -LiteralPath (Join-Path $week8 'difficulty_common_metric_final_2026-10-09\manifest.json'))) {
    $null = Ensure-Script 'watch_run_health.ps1'
}
if (-not (Test-Path -LiteralPath (Join-Path $week8 'bridge_expanded_full_oehb_preoutcome_2026-10-08\manifest.json'))) {
    $null = Ensure-Script 'watch_oehb_parent_health.ps1'
}
if (-not $manuscriptComplete) {
    $null = Ensure-Script 'continuous_execution_guard.ps1'
}

if ($WithSmallMiniGrid -and -not (Test-Path -LiteralPath $openedResult)) {
    $small = @($processes | Where-Object {
        $_.Name -ieq 'python.exe' -and
        $_.CommandLine -match '(?i)-m\s+benchmarks\.bridge\.opened_minigrid_small_precompute'
    })
    if ($small.Count -gt 1) { throw 'Multiple small-MiniGrid workers; inspect before resuming' }
    if ($small.Count -eq 1) {
        $record.jobs += [ordered]@{ name = 'opened_minigrid_small_precompute'; state = 'already_running'; pid = [int]$small[0].ProcessId }
    } elseif ($phase -ne 'benchmarks.bridge.expanded_preoutcome_horizon') {
        $record.jobs += [ordered]@{ name = 'opened_minigrid_small_precompute'; state = 'not_started_outside_horizon_stage'; pid = $null }
    } elseif ($DryRun) {
        $record.jobs += [ordered]@{ name = 'opened_minigrid_small_precompute'; state = 'would_start'; pid = $null }
    } else {
        $stamp = (Get-Date).ToString('yyyy-MM-dd_HHmmss')
        $stdout = Join-Path $week8 "bridge_opened_minigrid_small_resume_$stamp.out.log"
        $stderr = Join-Path $week8 "bridge_opened_minigrid_small_resume_$stamp.err.log"
        $job = Start-Process -FilePath $python -ArgumentList '-u -m benchmarks.bridge.opened_minigrid_small_precompute' `
            -WindowStyle Hidden -WorkingDirectory $repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
        $record.jobs += [ordered]@{ name = 'opened_minigrid_small_precompute'; state = 'started'; pid = $job.Id; stdout = $stdout; stderr = $stderr }
    }
}

if (-not $DryRun) {
    $stamp = (Get-Date).ToString('yyyy-MM-dd_HHmmss')
    $out = Join-Path $week8 "bridge_resume_$stamp.json"
    [IO.File]::WriteAllText($out, (($record | ConvertTo-Json -Depth 6) + [Environment]::NewLine),
        (New-Object System.Text.UTF8Encoding($false)))
    Write-Output "Recovery record: $out"
}
$record.jobs | ForEach-Object { [pscustomobject]$_ } | Format-Table name, state, pid -AutoSize
