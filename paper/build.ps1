$ErrorActionPreference = 'Stop'
$paperRoot = $PSScriptRoot
if (-not (Get-Command pdflatex -ErrorAction SilentlyContinue)) { throw 'pdflatex is required' }
if (-not (Get-Command bibtex -ErrorAction SilentlyContinue)) { throw 'bibtex is required' }
Push-Location $paperRoot
try {
    & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'First LaTeX pass failed; inspect paper/main.log' }
    & bibtex main | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'BibTeX failed; inspect paper/main.blg' }
    & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Second LaTeX pass failed; inspect paper/main.log' }
    & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Final LaTeX pass failed; inspect paper/main.log' }
    if (-not (Test-Path -LiteralPath main.pdf)) { throw 'Paper PDF was not produced' }
    if (Get-Command pdfinfo -ErrorAction SilentlyContinue) {
        $pagesLine = & pdfinfo main.pdf | Select-String '^Pages:\s+(\d+)$'
        if (-not $pagesLine) { throw 'Cannot read the PDF page count' }
        $pages = [int]$pagesLine.Matches[0].Groups[1].Value
        if ($pages -gt 8) { throw "Paper exceeds the eight-page budget: $pages pages" }
        Write-Host "Built paper/main.pdf ($pages pages)"
    } else {
        Write-Host 'Built paper/main.pdf; pdfinfo unavailable for page-budget check'
    }
} finally {
    Pop-Location
}
