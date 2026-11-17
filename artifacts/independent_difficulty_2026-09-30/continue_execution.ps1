$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$pythonExe = 'D:\anaconda3\python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
Set-Location -LiteralPath $repo
$statusPath = Join-Path $PSScriptRoot 'background_status.json'
function Write-Status([string]$phase, [string]$detail) {
    @{ phase = $phase; detail = $detail; at_utc = [DateTime]::UtcNow.ToString('o') } |
        ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
}
try {
    if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'preoutcome_seal.json'))) {
        throw 'Difficulty forecasts were not sealed before evaluation.'
    }
    Write-Status 'difficulty_evaluate' 'Running all 54,000 core learner-budget-seed outcomes.'
    & $pythonExe -m scripts.analysis.independent_difficulty evaluate *>> (Join-Path $PSScriptRoot 'execution.log')
    if ($LASTEXITCODE -ne 0) { throw "Difficulty evaluation exited $LASTEXITCODE; see execution.log and failure_evaluate.json." }
    Write-Status 'difficulty_analyze' 'All difficulty outcomes complete; fixed analysis started.'
    & $pythonExe -m scripts.analysis.independent_difficulty analyze *>> (Join-Path $PSScriptRoot 'analysis.log')
    if ($LASTEXITCODE -ne 0) { throw "Difficulty analysis exited $LASTEXITCODE; see analysis.log." }
    Write-Status 'routing_evaluate' 'Running the separately sealed 21,000 seven-method routing jobs.'
    & $pythonExe -m scripts.analysis.run_unseen_validation evaluate *>> (Join-Path $PSScriptRoot 'routing_execution.log')
    if ($LASTEXITCODE -ne 0) { throw "Routing evaluation exited $LASTEXITCODE; see routing_execution.log." }
    Write-Status 'routing_analyze' 'All routing outcomes complete; fixed analysis started.'
    & $pythonExe -m scripts.analysis.run_unseen_validation analyze *>> (Join-Path $PSScriptRoot 'routing_analysis.log')
    if ($LASTEXITCODE -ne 0) { throw "Routing analysis exited $LASTEXITCODE; see routing_analysis.log." }
    Write-Status 'complete' 'Both independent result manifests are available.'
} catch {
    Write-Status 'failed' $_.Exception.Message
    $_ | Out-String | Add-Content -LiteralPath (Join-Path $PSScriptRoot 'background_error.log') -Encoding UTF8
    exit 1
}
