param()

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$week8 = Join-Path $repo 'artifacts\week8'
$statusPath = Join-Path $week8 'bridge_scheduled_recovery_2026-10-11.json'
$resume = Join-Path $PSScriptRoot 'resume_checkpointed_runs.ps1'
$utf8 = New-Object System.Text.UTF8Encoding($false)

function Read-Status([string]$relative) {
    $path = Join-Path $week8 $relative
    if (-not (Test-Path -LiteralPath $path)) { return $null }
    for ($attempt = 0; $attempt -lt 3; $attempt++) {
        try { return Get-Content -LiteralPath $path -Raw | ConvertFrom-Json }
        catch { Start-Sleep -Milliseconds 200 }
    }
    throw "Status file could not be read: $path"
}

function Write-Status([string]$phase, [string]$detail) {
    $record = [ordered]@{
        checked_local = (Get-Date).ToString('o')
        phase = $phase
        detail = $detail
        wrapper_pid = $PID
    }
    [IO.File]::WriteAllText($statusPath, (($record | ConvertTo-Json -Compress) + [Environment]::NewLine), $utf8)
}

try {
    $progress = Read-Status 'bridge_expanded_pipeline_progress_2026-10-08.json'
    $opened = Read-Status 'bridge_opened_minigrid_queue_status_2026-10-09.json'
    $common = Read-Status 'difficulty_common_metric_watch_2026-10-09.json'
    $manuscript = Read-Status 'difficulty_manuscript_watch_2026-10-09.json'
    if (($progress -and $progress.phase -eq 'NO_GO') -or
        ($opened -and $opened.phase -eq 'NO_GO') -or
        ($common -and $common.phase -eq 'NO_GO') -or
        ($manuscript -and $manuscript.phase -eq 'REVIEW_REQUIRED')) {
        Write-Status 'REVIEW_REQUIRED' 'An upstream gate requires human inspection; no automatic retry'
        exit 0
    }
    if ($manuscript -and $manuscript.phase -eq 'COMPLETE_REVIEW_READY') {
        Write-Status 'COMPLETE' 'All audited manuscripts built; no recovery needed'
        exit 0
    }
    & $resume | Out-Null
    if (-not $?) { throw 'Checkpoint helper failed' }
    Write-Status 'CHECKED' 'Missing processes resumed; existing processes retained'
} catch {
    Write-Status 'REVIEW_REQUIRED' $_.Exception.Message
    exit 1
}
