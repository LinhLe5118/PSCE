param(
    [int]$TimeoutHours = 72,
    [ValidateSet('full', 'reduced')][string]$Mode = 'full'
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$pythonExe = 'D:\anaconda3\python.exe'
$artifactRoot = Join-Path $projectRoot 'artifacts\week8'
$logPath = Join-Path $artifactRoot 'bridge_overnight_pipeline_2026-10-07.log'
$statusPath = Join-Path $artifactRoot 'bridge_overnight_pipeline_2026-10-07.json'
$horizonDir = Join-Path $artifactRoot 'bridge_horizon_preoutcome_2026-10-07'
$oehbDir = Join-Path $artifactRoot 'bridge_oehb_preoutcome_2026-10-07'
$resourcePath = Join-Path $artifactRoot 'bridge_resource_gate_addendum_2026-10-07.json'
$started = Get-Date

function Write-Status([string]$state, [string]$detail) {
    $entry = [ordered]@{
        state = $state
        mode = $Mode
        detail = $detail
        updated_local = (Get-Date).ToString('o')
        horizon_groups = @(Get-ChildItem -LiteralPath (Join-Path $horizonDir 'groups') -Filter '*.json' -ErrorAction SilentlyContinue).Count
        oehb_groups = @(Get-ChildItem -LiteralPath (Join-Path $oehbDir 'groups') -Filter '*.json' -ErrorAction SilentlyContinue).Count
    }
    $entry | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

function Record-Status([string]$state, [string]$detail) {
    Write-Status $state $detail
    "$(Get-Date -Format o) $state : $detail" | Tee-Object -FilePath $logPath -Append
}

function Run-Module([string]$name, [string[]]$moduleArgs = @()) {
    Record-Status 'RUNNING' $name
    $priorErrorAction = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $pythonExe -m $name @moduleArgs 2>&1 | Tee-Object -FilePath $logPath -Append
    $ErrorActionPreference = $priorErrorAction
    if ($LASTEXITCODE -ne 0) {
        throw "$name exited with code $LASTEXITCODE"
    }
}

Set-Location -LiteralPath $projectRoot
if ($Mode -eq 'reduced') {
    $resource = Get-Content -LiteralPath $resourcePath -Raw | ConvertFrom-Json
    if ($resource.status -ne 'PRELABEL_RESOURCE_NO_GO_FULL_14_TASK_OEHB' -or $resource.test_labels_read) {
        throw 'Reduced comparison was not frozen before test labels'
    }
}
Record-Status 'WAITING_PREOUTCOME' "Awaiting $Mode preoutcome manifest(s); no test labels read."
try {
    while ($true) {
        $requiredDirs = if ($Mode -eq 'full') { @($horizonDir, $oehbDir) } else { @($horizonDir) }
        foreach ($dir in $requiredDirs) {
            if (Test-Path -LiteralPath (Join-Path $dir 'failure_report.json')) {
                throw "Preoutcome failure report: $dir"
            }
        }
        $ready = (Test-Path -LiteralPath (Join-Path $horizonDir 'manifest.json'))
        if ($Mode -eq 'full') {
            $ready = $ready -and (Test-Path -LiteralPath (Join-Path $oehbDir 'manifest.json'))
        }
        if ($ready) { break }
        if ((Get-Date) -gt $started.AddHours($TimeoutHours)) {
            throw "Preoutcome compute cap of $TimeoutHours hours elapsed without complete manifests"
        }
        Write-Status 'WAITING_PREOUTCOME' "Awaiting $Mode preoutcome manifest(s); no test labels read."
        Start-Sleep -Seconds 45
    }
    if ($Mode -eq 'full') {
        if (-not (Test-Path -LiteralPath (Join-Path $artifactRoot 'bridge_full_forecast_preoutcome_2026-10-07\manifest.json'))) {
            Run-Module 'benchmarks.bridge.preoutcome_full_forecast'
        }
    } else {
        if (-not (Test-Path -LiteralPath (Join-Path $artifactRoot 'bridge_reduced_forecast_preoutcome_2026-10-07\manifest.json'))) {
            Run-Module 'benchmarks.bridge.preoutcome_reduced_forecast'
        }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $artifactRoot "bridge_${Mode}_test_labels_2026-10-07\manifest.json"))) {
        Run-Module 'benchmarks.bridge.run_test_labels' @('--mode', $Mode)
    }
    if (-not (Test-Path -LiteralPath (Join-Path $artifactRoot "bridge_${Mode}_test_analysis_2026-10-07\manifest.json"))) {
        Run-Module 'benchmarks.bridge.score_test' @('--mode', $Mode)
    }
    Record-Status 'COMPLETE' "BRIDGE $Mode matched forecast, one label run and score complete."
} catch {
    Record-Status 'NO_GO' $_.Exception.Message
    throw
}
