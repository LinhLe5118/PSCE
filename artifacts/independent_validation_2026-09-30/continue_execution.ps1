param([int]$PreparePid = 11976)
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
    Write-Status 'waiting_for_choices_seal' 'Pilot-only routing must finish before any validation outcome.'
    $deadline = [DateTime]::UtcNow.AddHours(3)
    while (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'choices_seal.json'))) {
        if ([DateTime]::UtcNow -gt $deadline) { throw 'Prepare timed out before choices seal.' }
        if (-not (Get-Process -Id $PreparePid -ErrorAction SilentlyContinue)) {
            throw 'Prepare process exited before choices seal.'
        }
        Start-Sleep -Seconds 10
    }
    Write-Status 'evaluating' 'Frozen 21,000 matched method-seed jobs started.'
    & $pythonExe -m scripts.analysis.run_unseen_validation evaluate *>> (Join-Path $PSScriptRoot 'execution.log')
    if ($LASTEXITCODE -ne 0) { throw "Evaluation exited $LASTEXITCODE; see execution.log and failure_evaluate.json." }
    Write-Status 'analyzing' 'All jobs complete; reading outcomes once for fixed analysis.'
    & $pythonExe -m scripts.analysis.run_unseen_validation analyze *>> (Join-Path $PSScriptRoot 'analysis.log')
    if ($LASTEXITCODE -ne 0) { throw "Analysis exited $LASTEXITCODE; see analysis.log." }
    Write-Status 'complete' 'Independent validation result_manifest.json is available.'
} catch {
    Write-Status 'failed' $_.Exception.Message
    $_ | Out-String | Add-Content -LiteralPath (Join-Path $PSScriptRoot 'background_error.log') -Encoding UTF8
    exit 1
}
