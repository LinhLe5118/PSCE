param([int]$MinimumFreeMiB = 2048)

$ErrorActionPreference = 'Stop'
if ($MinimumFreeMiB -lt 2048) { throw 'Unsafe memory floor' }
$rootDir = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$week8 = Join-Path $rootDir 'artifacts\week8'
$groups = Join-Path $week8 'bridge_opened_minigrid_full_oehb_exploratory_2026-10-09\groups'
$statusFile = Join-Path $week8 'bridge_opened_minigrid_medium_guard_2026-10-10.json'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $week8 "bridge_medium_precompute_$stamp.out.log"
$stderr = Join-Path $week8 "bridge_medium_precompute_$stamp.err.log"
$pidFile = Join-Path $week8 "bridge_medium_child_$stamp.json"
$utf8 = New-Object System.Text.UTF8Encoding($false)
$worker = $null
$childPid = 0
$lowChecks = 0
$before = @(Get-ChildItem -LiteralPath $groups -Filter '*.json' -ErrorAction SilentlyContinue).Count

function Write-State([string]$phase, [string]$detail, [double]$freeMiB) {
    $record = [ordered]@{
        phase = $phase
        detail = $detail
        checked_local = (Get-Date).ToString('o')
        worker_pid = if ($worker) { $worker.Id } else { $null }
        child_pid = $childPid
        free_memory_mib = [math]::Round($freeMiB)
        completed_groups = @(Get-ChildItem -LiteralPath $groups -Filter '*.json' -ErrorAction SilentlyContinue).Count
        groups_before = $before
        stdout = $stdout
        stderr = $stderr
    }
    [IO.File]::WriteAllText($statusFile, (($record | ConvertTo-Json -Depth 4) + [Environment]::NewLine), $utf8)
}

function Free-MiB {
    return [double](Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue
}

try {
    $free = Free-MiB
    if ($free -lt 6144) {
        Write-State 'NOT_STARTED_LOW_MEMORY' 'At least 6 GiB free RAM required to overlap one 469-state exploratory group' $free
        exit 0
    }
    $other = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction Stop | Where-Object {
        $_.CommandLine -match 'benchmarks\.bridge\.opened_minigrid_medium_precompute'
    })
    if ($other.Count) { throw 'Another medium precompute is running' }
    $env:PYTHONIOENCODING = 'utf-8'
    $env:OMP_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $pipelineScript = Join-Path $PSScriptRoot 'medium_precompute_pipeline.ps1'
    $arguments = '-NoProfile -ExecutionPolicy Bypass -File "' + $pipelineScript + '" -PidFile "' + $pidFile + '" -LogFile "' + $stdout + '"'
    $worker = Start-Process -FilePath 'powershell.exe' -WorkingDirectory $rootDir `
        -ArgumentList $arguments `
        -WindowStyle Hidden -RedirectStandardOutput ($stdout + '.supervisor') -RedirectStandardError $stderr -PassThru
    for ($attempt = 0; $attempt -lt 24 -and -not (Test-Path -LiteralPath $pidFile); $attempt++) {
        Start-Sleep -Seconds 5
    }
    if (-not (Test-Path -LiteralPath $pidFile)) { throw 'Spawn child PID file missing' }
    $child = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
    if ($child.status -ne 'EXPLORATORY_CHILD_STARTED' -or [int]$child.child_pid -le 0) {
        throw 'Invalid spawned child record'
    }
    $childPid = [int]$child.child_pid
    Write-State 'RUNNING' 'Five existing exploratory checkpoints scheduled with low-memory guard' $free
    while (Get-Process -Id $childPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 30
        $free = Free-MiB
        if ($free -lt $MinimumFreeMiB) { $lowChecks++ } else { $lowChecks = 0 }
        if ((Get-Process -Id $childPid -ErrorAction SilentlyContinue) -and $lowChecks -ge 2) {
            $identity = Get-CimInstance Win32_Process -Filter "ProcessId=$childPid" -ErrorAction Stop
            if (-not $identity -or $identity.CommandLine -notmatch 'multiprocessing\.spawn.*spawn_main') {
                throw 'Medium process identity changed; no process stopped'
            }
            Stop-Process -Id $childPid -ErrorAction Stop
            Write-State 'PAUSED_LOW_MEMORY' 'Stopped only exploratory worker after two low-memory checks' $free
            exit 0
        }
        Write-State 'RUNNING' 'Prospective worker and exploratory worker monitored independently' $free
    }
    $after = @(Get-ChildItem -LiteralPath $groups -Filter '*.json' -ErrorAction SilentlyContinue).Count
    if ($after -le $before -or $after -gt $before + 5) {
        throw "Medium child disappeared; checkpoint count $before -> $after"
    }
    Write-State 'COMPLETE_GROUPS' "$($after - $before) exploratory checkpoints written by unchanged runner" (Free-MiB)
} catch {
    Write-State 'REVIEW_REQUIRED' $_.Exception.Message (Free-MiB)
    exit 1
}
