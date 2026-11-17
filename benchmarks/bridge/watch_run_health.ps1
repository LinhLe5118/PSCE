param(
    [int]$IntervalSeconds = 60
)

$ErrorActionPreference = 'Stop'
if ($IntervalSeconds -lt 15) { throw 'IntervalSeconds must be at least 15' }
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$week8 = Join-Path $repo 'artifacts\week8'
$prospective = Join-Path $week8 'bridge_expanded_full_test_analysis_2026-10-08\manifest.json'
$opened = Join-Path $week8 'bridge_opened_minigrid_full_analysis_exploratory_2026-10-09\manifest.json'
$common = Join-Path $week8 'difficulty_common_metric_final_2026-10-09\manifest.json'
$pipelineStatus = Join-Path $week8 'bridge_expanded_live_status_2026-10-08.json'
$pipelineProgress = Join-Path $week8 'bridge_expanded_pipeline_progress_2026-10-08.json'
$output = Join-Path $week8 'bridge_run_health_2026-10-09.json'
$history = Join-Path $week8 'bridge_run_health_2026-10-09.jsonl'
$utf8 = New-Object System.Text.UTF8Encoding($false)
$previousCpu = @{}
$idleChecks = @{}

function Count-Checkpoints([string]$subdir) {
    $dir = Join-Path $week8 $subdir
    $files = @(Get-ChildItem -LiteralPath $dir -Filter '*.json' -ErrorAction SilentlyContinue)
    $latest = $files | Sort-Object LastWriteTimeUtc | Select-Object -Last 1
    return [ordered]@{
        count = $files.Count
        latest_local = if ($latest) { $latest.LastWriteTime.ToString('o') } else { $null }
        age_minutes = if ($latest) { [math]::Round(((Get-Date) - $latest.LastWriteTime).TotalMinutes, 1) } else { $null }
    }
}

while ($true) {
    $now = Get-Date
    $live = if (Test-Path -LiteralPath $pipelineStatus) {
        try { Get-Content -LiteralPath $pipelineStatus -Raw | ConvertFrom-Json } catch { $null }
    } else { $null }
    $progress = if (Test-Path -LiteralPath $pipelineProgress) {
        try { Get-Content -LiteralPath $pipelineProgress -Raw | ConvertFrom-Json } catch { $null }
    } else { $null }
    $allPython = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'")
    $pythonRows = @($allPython | Where-Object {
        $_.CommandLine -match 'benchmarks\.bridge\.(expanded_preoutcome_horizon|expanded_preoutcome_oehb|expanded_run_test_labels|opened_minigrid_small_precompute|opened_minigrid_full_exploratory)'
    })
    $workers = @()
    foreach ($row in $pythonRows) {
        $family = @($row)
        do {
            $knownIds = @($family | ForEach-Object { [int]$_.ProcessId })
            $children = @($allPython | Where-Object {
                ([int]$_.ParentProcessId -in $knownIds) -and ([int]$_.ProcessId -notin $knownIds)
            })
            $family += $children
        } while ($children.Count -gt 0)
        $liveFamily = @($family | ForEach-Object { Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue })
        if (-not $liveFamily.Count) { continue }
        $key = [string]$row.ProcessId
        $cpu = [double](($liveFamily | Measure-Object -Property CPU -Sum).Sum)
        $increased = if ($previousCpu.ContainsKey($key)) { $cpu -gt ($previousCpu[$key] + 0.01) } else { $null }
        if ($null -eq $increased -or $increased) { $idleChecks[$key] = 0 }
        else { $idleChecks[$key] = [int]$idleChecks[$key] + 1 }
        $previousCpu[$key] = $cpu
        $workers += [ordered]@{
            pid = [int]$row.ProcessId
            child_pids = @($family | Where-Object { $_.ProcessId -ne $row.ProcessId } | ForEach-Object { [int]$_.ProcessId })
            module = [regex]::Match($row.CommandLine, 'benchmarks\.bridge\.[A-Za-z0-9_]+').Value
            cpu_seconds = [math]::Round($cpu, 1)
            cpu_increased_since_last_check = $increased
            idle_checks = [int]$idleChecks[$key]
            rss_gib = [math]::Round((($liveFamily | Measure-Object -Property WorkingSet64 -Sum).Sum) / 1GB, 2)
        }
    }
    $allPythonIds = @($allPython | ForEach-Object { [int]$_.ProcessId })
    $orphanRows = @($allPython | Where-Object {
        $_.CommandLine -match 'multiprocessing\.spawn.*spawn_main' -and
        ([int]$_.ParentProcessId -notin $allPythonIds)
    })
    foreach ($row in $orphanRows) {
        $process = Get-Process -Id $row.ProcessId -ErrorAction SilentlyContinue
        if (-not $process) { continue }
        $key = [string]$row.ProcessId
        $cpu = [double]$process.CPU
        $increased = if ($previousCpu.ContainsKey($key)) { $cpu -gt ($previousCpu[$key] + 0.01) } else { $null }
        if ($null -eq $increased -or $increased) { $idleChecks[$key] = 0 }
        else { $idleChecks[$key] = [int]$idleChecks[$key] + 1 }
        $previousCpu[$key] = $cpu
        $workers += [ordered]@{
            pid = [int]$row.ProcessId
            parent_pid = [int]$row.ParentProcessId
            child_pids = @()
            module = 'orphan_multiprocessing_child'
            cpu_seconds = [math]::Round($cpu, 1)
            cpu_increased_since_last_check = $increased
            idle_checks = [int]$idleChecks[$key]
            rss_gib = [math]::Round($process.WorkingSet64 / 1GB, 2)
        }
    }
    foreach ($key in @($previousCpu.Keys)) {
        if (-not @($workers | Where-Object { [string]$_.pid -eq $key }).Count) {
            $null = $previousCpu.Remove($key)
            $null = $idleChecks.Remove($key)
        }
    }
    $os = Get-CimInstance Win32_OperatingSystem
    $disk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='D:'"
    $horizon = Count-Checkpoints 'bridge_expanded_horizon_preoutcome_2026-10-08\groups'
    $oehb = Count-Checkpoints 'bridge_expanded_full_oehb_preoutcome_2026-10-08\groups'
    $mini = Count-Checkpoints 'bridge_opened_minigrid_full_oehb_exploratory_2026-10-09\groups'
    $prospectiveDone = Test-Path -LiteralPath $prospective
    $openedDone = Test-Path -LiteralPath $opened
    $commonDone = Test-Path -LiteralPath $common
    $phase = if ($live) { [string]$live.phase } else { 'NO_LIVE_STATUS' }
    $stageAgeMinutes = if ($progress) {
        ($now - [datetime]$progress.updated_local).TotalMinutes
    } else { $null }
    $warnings = @()
    if ($live -and ($now - [datetime]$live.updated_local).TotalSeconds -gt 2 * $IntervalSeconds) {
        $warnings += 'PROSPECTIVE_WATCHER_STALE'
    }
    if ($phase -eq 'NO_GO') { $warnings += 'PIPELINE_NO_GO' }
    if (-not $prospectiveDone -and $phase -ne 'NO_GO' -and $live -and -not $live.pipeline_running) {
        $warnings += 'PROSPECTIVE_SUPERVISOR_MISSING'
    }
    if (@($workers | Where-Object { $_.idle_checks -ge 10 }).Count) {
        $warnings += 'PYTHON_WORKER_CPU_IDLE_FOR_TEN_CHECKS'
    }
    if ($orphanRows.Count -gt 0) { $warnings += 'ORPHAN_PYTHON_PARENT_MISSING_INSPECT_CHECKPOINT' }
    if ($phase -eq 'benchmarks.bridge.expanded_preoutcome_oehb' -and
        $stageAgeMinutes -gt 120 -and $oehb.count -eq 0) {
        $warnings += 'OEHB_NO_FIRST_CHECKPOINT_AFTER_TWO_HOURS'
    }
    if ($stageAgeMinutes -gt 120 -and
        (($phase -eq 'benchmarks.bridge.expanded_preoutcome_horizon' -and $horizon.age_minutes -gt 120) -or
         ($phase -eq 'benchmarks.bridge.expanded_preoutcome_oehb' -and $oehb.age_minutes -gt 120))) {
        $warnings += 'CHECKPOINT_OLDER_THAN_TWO_HOURS_INSPECT_LONG_GROUP'
    }
    if ($os.FreePhysicalMemory / 1MB -lt 2) { $warnings += 'LOW_FREE_RAM_UNDER_2_GIB' }
    if ($disk.FreeSpace / 1GB -lt 10) { $warnings += 'LOW_D_FREE_SPACE_UNDER_10_GIB' }
    $state = [ordered]@{
        checked_local = $now.ToString('o')
        phase = $phase
        prospective_analysis = $prospectiveDone
        opened_analysis = $openedDone
        common_metric_audit = $commonDone
        prospective_horizon = $horizon
        prospective_oehb = $oehb
        opened_minigrid_oehb = $mini
        workers = $workers
        free_ram_gib = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
        free_d_gib = [math]::Round($disk.FreeSpace / 1GB, 2)
        warnings = $warnings
    }
    $json = $state | ConvertTo-Json -Depth 7 -Compress
    [IO.File]::WriteAllText($output, ($json + [Environment]::NewLine), $utf8)
    [IO.File]::AppendAllText($history, ($json + [Environment]::NewLine), $utf8)
    if ($commonDone -or $phase -eq 'NO_GO') { break }
    Start-Sleep -Seconds $IntervalSeconds
}
