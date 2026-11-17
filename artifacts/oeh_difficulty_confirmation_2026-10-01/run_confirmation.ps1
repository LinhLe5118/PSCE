$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $repoRoot
$pythonExe = 'D:\anaconda3\python.exe'
$outputRoot = $PSScriptRoot
$statusPath = Join-Path $outputRoot 'pipeline_status.json'

function Save-Status([string]$phase, [string]$detail) {
    @{phase=$phase; detail=$detail; at_utc=[DateTime]::UtcNow.ToString('o')} |
        ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

function Invoke-PythonPhase([string]$phase) {
    $process = Start-Process -FilePath $pythonExe `
        -ArgumentList @('-m','scripts.analysis.confirm_oeh_difficulty',$phase) `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput (Join-Path $outputRoot "$phase.stdout.log") `
        -RedirectStandardError (Join-Path $outputRoot "$phase.stderr.log") `
        -WindowStyle Hidden -Wait -PassThru
    if ($process.ExitCode -ne 0) { throw "$phase exited $($process.ExitCode)" }
}

try {
    Save-Status 'preparing' 'Generate new data, pilot features, and sealed forecasts before learner outcomes.'
    Invoke-PythonPhase 'prepare'
    Save-Status 'evaluating' 'All 6,300 forecasts sealed; execute full 54,000-job learner grid.'
    Invoke-PythonPhase 'evaluate'
    Save-Status 'analyzing' 'Score complete sealed forecasts and calculate registered simultaneous bounds.'
    Invoke-PythonPhase 'analyze'
    $gate = Get-Content -LiteralPath (Join-Path $outputRoot 'result_manifest.json') -Raw | ConvertFrom-Json
    if ($gate.difficulty_gate -eq 'GO') {
        Save-Status 'e1_descriptive' 'Independent gate GO; run grouped out-of-fold comparison on exact original E1.'
        $comparison = Start-Process -FilePath $pythonExe `
            -ArgumentList @('-m','scripts.analysis.compare_oeh_difficulty_e1') `
            -WorkingDirectory $repoRoot `
            -RedirectStandardOutput (Join-Path $outputRoot 'e1_descriptive.stdout.log') `
            -RedirectStandardError (Join-Path $outputRoot 'e1_descriptive.stderr.log') `
            -WindowStyle Hidden -Wait -PassThru
        if ($comparison.ExitCode -ne 0) { throw "E1 comparison exited $($comparison.ExitCode)" }
        Save-Status 'complete_go' 'Independent gate GO and original E1 descriptive comparison complete.'
    } else {
        Save-Status 'complete_no_go' 'Independent gate NO-GO; original E1 comparison withheld.'
    }
} catch {
    Save-Status 'failed' $_.Exception.Message
    $_ | Out-String | Add-Content -LiteralPath (Join-Path $outputRoot 'pipeline.stderr.log')
    exit 1
}
