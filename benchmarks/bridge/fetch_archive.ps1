param(
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'source_manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
$targetDir = Join-Path $repoRoot 'datasets\bridge_v1_1_1'
$archive = Join-Path $targetDir $manifest.source_zip
$url = 'https://zenodo.org/api/records/12556298/files/bridge_dataset_v1.1.1.zip/content'
$drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($targetDir))
$freeGiB = [math]::Round($drive.AvailableFreeSpace / 1GB, 1)
$existingBytes = if (Test-Path -LiteralPath $archive) { (Get-Item -LiteralPath $archive).Length } else { 0 }
if ($existingBytes -gt $manifest.source_zip_bytes) {
    throw 'Existing archive is larger than the pinned release; inspect it before resuming.'
}
$neededGiB = [math]::Ceiling(($manifest.source_zip_bytes - $existingBytes) / 1GB) + 15

Write-Host "BRIDGE source: $($manifest.source_record)"
Write-Host "Archive: $archive"
Write-Host "Free on $($drive.Name): $freeGiB GiB; download plus staging reserve: $neededGiB GiB"
if ($freeGiB -lt $neededGiB) {
    throw 'Insufficient free space for a safe download plus selective extraction.'
}
if ($CheckOnly) { return }

New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
if ((Test-Path -LiteralPath $archive) -and
    (Get-Item -LiteralPath $archive).Length -eq $manifest.source_zip_bytes) {
    Write-Host 'Archive already has expected length; verifying checksum.'
} else {
    $oldTemp, $oldTmp = $env:TEMP, $env:TMP
    try {
        $env:TEMP = $targetDir
        $env:TMP = $targetDir
        & curl.exe --fail --location --retry 5 --retry-delay 10 --connect-timeout 30 --continue-at - --output $archive $url
        if ($LASTEXITCODE -ne 0) { throw "curl download failed with exit code $LASTEXITCODE" }
    } finally {
        $env:TEMP = $oldTemp
        $env:TMP = $oldTmp
    }
}
if ((Get-Item -LiteralPath $archive).Length -ne $manifest.source_zip_bytes) {
    throw 'Archive length differs from pinned Zenodo release; keep file for inspection.'
}
$actual = (Get-FileHash -LiteralPath $archive -Algorithm MD5).Hash.ToLowerInvariant()
if ($actual -ne $manifest.source_zip_md5) {
    throw "Archive MD5 $actual differs from pinned $($manifest.source_zip_md5); keep file for inspection."
}
Write-Host "PASS: BRIDGE archive length and MD5 match v1.1.1 ($actual)"
