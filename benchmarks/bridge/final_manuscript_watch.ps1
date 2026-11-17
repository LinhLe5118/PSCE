param()

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$workspace = (Resolve-Path (Join-Path $repo '..')).Path
$week8 = Join-Path $repo 'artifacts\week8'
$commonManifest = Join-Path $week8 'difficulty_common_metric_final_2026-10-09\manifest.json'
$commonStatus = Join-Path $week8 'difficulty_common_metric_watch_2026-10-09.json'
$status = Join-Path $week8 'difficulty_manuscript_watch_2026-10-09.json'
$log = Join-Path $week8 'difficulty_manuscript_watch_2026-10-09.log'
$python = 'D:\anaconda3\python.exe'
$utf8 = New-Object System.Text.UTF8Encoding($false)
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
Set-Location -LiteralPath $repo

function Write-State([string]$phase, [string]$detail) {
    $record = [ordered]@{
        phase = $phase
        detail = $detail
        updated_local = (Get-Date).ToString('o')
        strict_common_metric = Test-Path -LiteralPath $commonManifest
        paper_pdf = Test-Path -LiteralPath (Join-Path $repo 'paper\main.pdf')
        thesis_docx = Test-Path -LiteralPath (Join-Path $workspace 'Drive\Thesis\DAORL_Thesis_Current_2026-10-09.docx')
    }
    [IO.File]::WriteAllText($status, (($record | ConvertTo-Json) + [Environment]::NewLine), $utf8)
}

try {
    while (-not (Test-Path -LiteralPath $commonManifest)) {
        if (Test-Path -LiteralPath $commonStatus) {
            $upstream = Get-Content -LiteralPath $commonStatus -Raw | ConvertFrom-Json
            if ($upstream.phase -eq 'NO_GO') { throw 'Strict common-metric watcher reported NO_GO' }
        }
        Write-State 'WAITING_FOR_STRICT_AUDIT' 'Both BRIDGE analyses and the four-panel hash/metric audit must finish first'
        Start-Sleep -Seconds 60
    }
    $manifest = Get-Content -LiteralPath $commonManifest -Raw | ConvertFrom-Json
    if ($manifest.status -ne 'ALL_PANELS_COMMON_METRIC_VERIFIED') {
        throw "Unexpected final common-metric status: $($manifest.status)"
    }
    Write-State 'GENERATING_SOURCES' 'Writing verified tables to paper supplement and thesis source'
    & $python -m scripts.analysis.finalize_verified_manuscripts *>> $log
    if ($LASTEXITCODE -ne 0) { throw "Manuscript source generator exited with $LASTEXITCODE" }

    Write-State 'BUILDING_PAPER' 'Building paper and supplement from verified results'
    & (Join-Path $repo 'paper\build.ps1') *>> $log
    if (-not $?) { throw 'Paper build failed; inspect log and page budget' }
    & (Join-Path $repo 'paper\build_supplement.ps1') *>> $log
    if (-not $?) { throw 'Supplement build failed; inspect log' }

    Write-State 'BUILDING_THESIS' 'Building DOCX with the generated BRIDGE results chapter'
    & (Join-Path $workspace 'Drive\Thesis\build_thesis_current.ps1') *>> $log
    if (-not $?) { throw 'Thesis build failed; inspect log' }
    $outputs = [ordered]@{}
    foreach ($entry in @(
        @{ name = 'paper_pdf'; path = (Join-Path $repo 'paper\main.pdf') },
        @{ name = 'supplement_pdf'; path = (Join-Path $repo 'paper\supplement.pdf') },
        @{ name = 'thesis_docx'; path = (Join-Path $workspace 'Drive\Thesis\DAORL_Thesis_Current_2026-10-09.docx') }
    )) {
        if (-not (Test-Path -LiteralPath $entry.path)) { throw "Built output missing: $($entry.path)" }
        $outputs[$entry.name] = (Get-FileHash -LiteralPath $entry.path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $build = [ordered]@{
        status = 'COMPLETE_REVIEW_READY'
        common_metric_manifest_sha256 = (Get-FileHash -LiteralPath $commonManifest -Algorithm SHA256).Hash.ToLowerInvariant()
        manuscript_source_manifest_sha256 = (Get-FileHash -LiteralPath (Join-Path $week8 'difficulty_common_metric_final_2026-10-09\manuscript_source_manifest.json') -Algorithm SHA256).Hash.ToLowerInvariant()
        output_sha256 = $outputs
        completed_local = (Get-Date).ToString('o')
    }
    [IO.File]::WriteAllText((Join-Path $week8 'difficulty_manuscript_build_manifest_2026-10-09.json'),
        (($build | ConvertTo-Json -Depth 5) + [Environment]::NewLine), $utf8)
    Write-State 'COMPLETE_REVIEW_READY' 'Paper PDF, supplement PDF and thesis DOCX rebuilt; review interpretation before submission'
} catch {
    $message = $_.Exception.Message
    Add-Content -LiteralPath $log -Value "[$((Get-Date).ToString('o'))] REVIEW_REQUIRED $message" -Encoding UTF8
    Write-State 'REVIEW_REQUIRED' $message
    exit 1
}
