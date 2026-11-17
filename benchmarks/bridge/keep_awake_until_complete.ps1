param()

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$week8 = Join-Path $root 'artifacts\week8'
$prospective = Join-Path $week8 'bridge_expanded_full_test_analysis_2026-10-08\manifest.json'
$exploratory = Join-Path $week8 'bridge_opened_minigrid_full_analysis_exploratory_2026-10-09\manifest.json'
$prospectiveStatus = Join-Path $week8 'bridge_expanded_live_status_2026-10-08.json'
$exploratoryStatus = Join-Path $week8 'bridge_opened_minigrid_queue_status_2026-10-09.json'
$statusFile = Join-Path $week8 'bridge_keep_awake_status_2026-10-09.json'
$utf8 = New-Object System.Text.UTF8Encoding($false)

Add-Type -Namespace BridgeRun -Name Power -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint flags);
'@

function Write-State([string]$phase, [string]$detail = '') {
    $state = [ordered]@{
        phase = $phase
        detail = $detail
        updated_local = (Get-Date).ToString('o')
        prospective_analysis = Test-Path -LiteralPath $prospective
        exploratory_analysis = Test-Path -LiteralPath $exploratory
        screen_may_turn_off = $true
    }
    [IO.File]::WriteAllText($statusFile, (($state | ConvertTo-Json) + [Environment]::NewLine), $utf8)
}

try {
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: prevent idle sleep without keeping the display on.
    $result = [BridgeRun.Power]::SetThreadExecutionState([uint32]2147483649)
    if ($result -eq 0) { throw 'Windows did not accept the system-awake request' }
    while (-not ((Test-Path -LiteralPath $prospective) -and (Test-Path -LiteralPath $exploratory))) {
        foreach ($file in @($prospectiveStatus, $exploratoryStatus)) {
            if (Test-Path -LiteralPath $file) {
                $phase = (Get-Content -LiteralPath $file -Raw | ConvertFrom-Json).phase
                if ($phase -eq 'NO_GO') { throw "Upstream stopped: $file" }
            }
        }
        Write-State 'KEEP_AWAKE_WHILE_EXPERIMENTS_RUN'
        Start-Sleep -Seconds 60
    }
    Write-State 'COMPLETE'
} catch {
    Write-State 'NO_GO' $_.Exception.Message
} finally {
    [void][BridgeRun.Power]::SetThreadExecutionState([uint32]2147483648)
}
