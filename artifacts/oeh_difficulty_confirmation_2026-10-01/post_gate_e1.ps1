$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $repoRoot
$outputRoot = $PSScriptRoot
$statusPath = Join-Path $outputRoot 'post_gate_e1_status.json'

function Save-Status([string]$phase, [string]$detail) {
    @{phase=$phase; detail=$detail; at_utc=[DateTime]::UtcNow.ToString('o')} |
        ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

try {
    Save-Status 'waiting' 'Waiting for complete independent difficulty result and registered GO gate.'
    while ($true) {
        $resultPath = Join-Path $outputRoot 'result_manifest.json'
        $pipelinePath = Join-Path $outputRoot 'pipeline_status.json'
        if (Test-Path -LiteralPath $resultPath) {
            $result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
            if ($result.status -ne 'complete') { throw 'Result manifest is not complete' }
            if ($result.difficulty_gate -ne 'GO') {
                Save-Status 'withheld_no_go' 'Gate NO-GO: do not run E1 comparison.'
                exit 0
            }
            $e1Directory = Join-Path $repoRoot 'artifacts/oeh_difficulty_e1_descriptive_2026-10-01'
            if (Test-Path -LiteralPath $e1Directory) {
                Save-Status 'already_present' 'E1 output exists; do not run a duplicate comparison.'
                exit 0
            }
            Save-Status 'running' 'Gate GO: run grouped out-of-fold comparison on original E1.'
            $process = Start-Process -FilePath 'D:\anaconda3\python.exe' `
                -ArgumentList @('-m','scripts.analysis.compare_oeh_difficulty_e1') `
                -WorkingDirectory $repoRoot `
                -RedirectStandardOutput (Join-Path $outputRoot 'post_gate_e1.stdout.log') `
                -RedirectStandardError (Join-Path $outputRoot 'post_gate_e1.stderr.log') `
                -WindowStyle Hidden -Wait -PassThru
            if ($process.ExitCode -ne 0) { throw "E1 comparison exited $($process.ExitCode)" }
            Save-Status 'complete' 'Original E1 descriptive comparison complete after independent GO.'
            exit 0
        }
        if (Test-Path -LiteralPath $pipelinePath) {
            $pipeline = Get-Content -LiteralPath $pipelinePath -Raw | ConvertFrom-Json
            if ($pipeline.phase -eq 'failed') {
                Save-Status 'withheld_failure' 'Confirmation pipeline failed; no E1 comparison.'
                exit 1
            }
        }
        Start-Sleep -Seconds 30
    }
} catch {
    Save-Status 'failed' $_.Exception.Message
    $_ | Out-String | Add-Content -LiteralPath (Join-Path $outputRoot 'post_gate_e1.stderr.log')
    exit 1
}
