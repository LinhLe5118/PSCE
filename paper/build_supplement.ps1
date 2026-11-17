$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error supplement.tex | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'First supplement LaTeX pass failed' }
    & bibtex supplement | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Supplement BibTeX failed' }
    & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error supplement.tex | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Second supplement LaTeX pass failed' }
    & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error supplement.tex | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Final supplement LaTeX pass failed' }
    Write-Host 'Built paper/supplement.pdf'
} finally {
    Pop-Location
}
