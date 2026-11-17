param(
    [Parameter(Mandatory = $true)][string]$DocxPath,
    [string]$PdfPath
)
# Opens the generated thesis in Word, fills the table of contents and caption
# lists, saves the DOCX and exports a PDF for page-by-page review.
$ErrorActionPreference = 'Stop'
$docx = [System.IO.Path]::GetFullPath($DocxPath)
if (-not $PdfPath) { $PdfPath = [System.IO.Path]::ChangeExtension($docx, '.pdf') }
$pdf = [System.IO.Path]::GetFullPath($PdfPath)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($docx, $false, $false)
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Fields.Update() | Out-Null
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Save()
    $doc.ExportAsFixedFormat($pdf, 17)
    $pages = $doc.ComputeStatistics(2)
    $doc.Close($false)
    Write-Output "Updated $docx ($pages pages); PDF $pdf"
} finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
