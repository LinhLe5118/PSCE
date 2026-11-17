param()

$ErrorActionPreference = 'Stop'
$rootDir = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $rootDir
$pythonExe = 'D:\anaconda3\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

$week8 = Join-Path $rootDir 'artifacts\week8'
$upstream = Join-Path $week8 'bridge_expanded_full_test_analysis_2026-10-08\manifest.json'
$upstreamStatus = Join-Path $week8 'bridge_expanded_live_status_2026-10-08.json'
$statusFile = Join-Path $week8 'bridge_opened_minigrid_queue_status_2026-10-09.json'
$logFile = Join-Path $week8 'bridge_opened_minigrid_queue_2026-10-09.log'
$utf8 = New-Object System.Text.UTF8Encoding($false)

function Write-State([string]$phase, [string]$detail) {
    $groupDir = Join-Path $week8 'bridge_opened_minigrid_full_oehb_exploratory_2026-10-09\groups'
    $state = [ordered]@{
        phase = $phase
        detail = $detail
        updated_local = (Get-Date).ToString('o')
        oehb_groups = @(Get-ChildItem -LiteralPath $groupDir -Filter '*.json' -ErrorAction SilentlyContinue).Count
        total_groups = 70
        labels_were_already_opened = $true
        full_forecast = Test-Path -LiteralPath (Join-Path $week8 'bridge_opened_minigrid_full_forecast_exploratory_2026-10-09\manifest.json')
        analysis = Test-Path -LiteralPath (Join-Path $week8 'bridge_opened_minigrid_full_analysis_exploratory_2026-10-09\manifest.json')
    }
    [System.IO.File]::WriteAllText($statusFile, (($state | ConvertTo-Json -Depth 4) + [Environment]::NewLine), $utf8)
}

function Run-Step([string]$module, [string]$manifest) {
    if (Test-Path -LiteralPath $manifest) { return }
    Write-State $module 'Running checkpointed exploratory stage'
    Add-Content -LiteralPath $logFile -Value "[$((Get-Date).ToString('o'))] START $module" -Encoding UTF8
    # Windows PowerShell 5.1 promotes a native process's stderr warnings to
    # terminating errors under Stop, even when the process continues normally.
    # Capture them in the log and judge the stage by its exit code and manifest.
    $previousErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $pythonExe -m $module *>> $logFile
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
    if ($LASTEXITCODE -ne 0) { throw "$module exited with $LASTEXITCODE" }
    if (-not (Test-Path -LiteralPath $manifest)) { throw "$module did not write its manifest" }
    Add-Content -LiteralPath $logFile -Value "[$((Get-Date).ToString('o'))] PASS $module" -Encoding UTF8
}

try {
    Write-State 'WAITING_FOR_FRESH_PANEL' 'The 14-task exploratory job starts after the five-task prospective BRIDGE analysis completes'
    while (-not (Test-Path -LiteralPath $upstream)) {
        if (Test-Path -LiteralPath $upstreamStatus) {
            $live = Get-Content -LiteralPath $upstreamStatus -Raw | ConvertFrom-Json
            if ($live.phase -eq 'NO_GO') { throw "Prospective BRIDGE panel stopped: $($live.detail)" }
        }
        Start-Sleep -Seconds 30
    }
    Run-Step 'benchmarks.bridge.opened_minigrid_full_exploratory' (Join-Path $week8 'bridge_opened_minigrid_full_oehb_exploratory_2026-10-09\manifest.json')
    Run-Step 'benchmarks.bridge.opened_minigrid_full_forecast_exploratory' (Join-Path $week8 'bridge_opened_minigrid_full_forecast_exploratory_2026-10-09\manifest.json')
    Run-Step 'benchmarks.bridge.opened_minigrid_full_score_exploratory' (Join-Path $week8 'bridge_opened_minigrid_full_analysis_exploratory_2026-10-09\manifest.json')
    Write-State 'COMPLETE' 'Fourteen previously opened MiniGrid MDPs scored with full OEH; exploratory only'
} catch {
    $message = $_.Exception.Message
    Add-Content -LiteralPath $logFile -Value "[$((Get-Date).ToString('o'))] NO_GO $message" -Encoding UTF8
    Write-State 'NO_GO' $message
    exit 1
}
