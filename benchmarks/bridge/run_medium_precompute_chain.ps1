param()

$ErrorActionPreference = 'Stop'
$rootDir = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$week8 = Join-Path $rootDir 'artifacts\week8'
$groups = Join-Path $week8 'bridge_opened_minigrid_full_oehb_exploratory_2026-10-09\groups'
$guardStatus = Join-Path $week8 'bridge_opened_minigrid_medium_guard_2026-10-10.json'
$statusFile = Join-Path $week8 'bridge_opened_minigrid_medium_chain_2026-10-10.json'
$guardScript = Join-Path $PSScriptRoot 'run_medium_precompute_guarded.ps1'
$targets = @('MiniGrid-LavaCrossingS11N5-v0', 'MiniGrid-LavaGapS7-v0',
             'MiniGrid-SimpleCrossingS11N5-v0')
$utf8 = New-Object System.Text.UTF8Encoding($false)
$awaitingStart = $false
$launchedAt = [datetime]::MinValue
$launches = 0

function Counts {
    $all = @(Get-ChildItem -LiteralPath $groups -Filter '*.json' -ErrorAction SilentlyContinue |
        ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json })
    $result = [ordered]@{}
    foreach ($name in $targets) {
        $result[$name] = @($all | Where-Object { $_.mdp_id -eq $name -and
            $_.status -eq 'EXPLORATORY_FULL_OEHB_AFTER_LABEL_OPEN' }).Count
    }
    return $result
}

function Write-State([string]$phase, [string]$detail, $counts) {
    $record = [ordered]@{
        phase = $phase
        detail = $detail
        checked_local = (Get-Date).ToString('o')
        target_group_counts = $counts
        launches = $launches
    }
    [IO.File]::WriteAllText($statusFile, (($record | ConvertTo-Json -Depth 5) + [Environment]::NewLine), $utf8)
}

try {
    while ($true) {
        $counts = Counts
        if (@($counts.Values | Where-Object { $_ -ne 5 }).Count -eq 0) {
            Write-State 'COMPLETE_15_GROUPS' 'Three 469-493-state exploratory MDPs checkpointed' $counts
            break
        }
        $guard = if (Test-Path -LiteralPath $guardStatus) {
            Get-Content -LiteralPath $guardStatus -Raw | ConvertFrom-Json
        } else { $null }
        if ($awaitingStart) {
            if ($guard -and [datetime]$guard.checked_local -gt $launchedAt) {
                $awaitingStart = $false
            } elseif ((Get-Date) -gt $launchedAt.AddMinutes(2)) {
                throw 'New medium guard did not write a fresh status within two minutes'
            } else {
                Write-State 'WAITING_FOR_GUARD_START' 'New guard is starting' $counts
                Start-Sleep -Seconds 10
                continue
            }
        }
        if ($guard -and $guard.phase -eq 'RUNNING') {
            if ((Get-Process -Id ([int]$guard.child_pid) -ErrorAction SilentlyContinue) -or
                (Get-Date) -lt ([datetime]$guard.checked_local).AddMinutes(2)) {
                Write-State 'RUNNING' 'Existing guarded MiniGrid child is active' $counts
                Start-Sleep -Seconds 30
                continue
            }
            throw 'Medium child absent while guard status stayed RUNNING'
        }
        if ($guard -and $guard.phase -eq 'REVIEW_REQUIRED') {
            throw "Medium guard requires review: $($guard.detail)"
        }
        $free = [double](Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue
        if ($free -lt 6144) {
            Write-State 'WAITING_FOR_MEMORY' "Free RAM $([math]::Round($free)) MiB below 6144 MiB start floor" $counts
            Start-Sleep -Seconds 60
            continue
        }
        $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        $stdout = Join-Path $week8 "bridge_medium_chain_guard_$stamp.out.log"
        $stderr = Join-Path $week8 "bridge_medium_chain_guard_$stamp.err.log"
        $job = Start-Process -FilePath 'powershell.exe' `
            -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',('"'+$guardScript+'"')) `
            -WorkingDirectory $rootDir -WindowStyle Hidden `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
        $launches++
        $launchedAt = Get-Date
        $awaitingStart = $true
        Write-State 'GUARD_LAUNCHED' "Guard PID $($job.Id), free RAM $([math]::Round($free)) MiB" $counts
        Start-Sleep -Seconds 10
    }
} catch {
    Write-State 'REVIEW_REQUIRED' $_.Exception.Message (Counts)
    exit 1
}
