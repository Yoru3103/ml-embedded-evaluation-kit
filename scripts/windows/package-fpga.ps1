# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<#
.SYNOPSIS
Package an ATFE MPS4 SSE-320 FPGA build for either TFLM or PTE.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$BuildDirectory,
    [Parameter(Mandatory=$true)][string]$CompilerBin,
    [string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).ProviderPath
if (-not [IO.Path]::IsPathRooted($BuildDirectory)) {
    $BuildDirectory = Join-Path $repoRoot $BuildDirectory
}
$BuildDirectory = (Resolve-Path -LiteralPath $BuildDirectory).ProviderPath
$info = Get-Content -LiteralPath (Join-Path $BuildDirectory 'build-info.json') -Raw |
    ConvertFrom-Json
$configuration = $info.configuration
if ($info.status -ne 'built' -or $configuration.TARGET_PLATFORM -ne 'mps4' -or
    $configuration.TARGET_SUBSYSTEM -ne 'sse-320' -or
    $configuration.FPGA_PLATFORM_SSE_320 -notin @('ON', 'TRUE', '1')) {
    throw 'Select a completed MPS4 SSE-320 FPGA build.'
}
$cacheKiB = [int]$configuration.ETHOS_U_NPU_CACHE_SIZE / 1024
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $repoRoot `
        "build/board-test-packages/$($info.use_case)/$($info.framework)/${cacheKiB}k"
} elseif (-not [IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory = Join-Path $repoRoot $OutputDirectory
}
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
foreach ($folder in @('SOFTWARE', 'debug', 'model', 'validation')) {
    New-Item -ItemType Directory -Path (Join-Path $OutputDirectory $folder) -Force | Out-Null
}
$axf = Join-Path $BuildDirectory "bin/mlek_$($info.use_case).axf"
$objcopy = Join-Path $CompilerBin 'llvm-objcopy.exe'
New-Item -ItemType Directory -Path (Join-Path $BuildDirectory 'package-validation') -Force |
    Out-Null
foreach ($tag in @('boot', 'bram', 'ddr')) {
    $source = Join-Path $BuildDirectory "bin/sectors/$($info.use_case)/$tag.bin"
    if ((Get-Item -LiteralPath $source).Length -eq 0) { throw "Empty image: $tag" }
    $verification = Join-Path $BuildDirectory "package-validation/$tag.axf.bin"
    & $objcopy -O binary "--only-section=*.at_$tag" $axf $verification
    if ($LASTEXITCODE -ne 0) { throw 'AXF section extraction failed.' }
    if ((Get-FileHash -LiteralPath $source).Hash -ne
        (Get-FileHash -LiteralPath $verification).Hash) { throw "AXF/bin mismatch: $tag" }
    Copy-Item -LiteralPath $source -Destination (Join-Path $OutputDirectory "SOFTWARE/$tag.bin")
}
Copy-Item -LiteralPath $axf -Destination (Join-Path $OutputDirectory 'debug')
Copy-Item -LiteralPath (Join-Path $BuildDirectory "bin/mlek_$($info.use_case).map") `
    -Destination (Join-Path $OutputDirectory 'debug')
Copy-Item -LiteralPath (Join-Path $BuildDirectory 'bin/sectors/images.txt') `
    -Destination (Join-Path $OutputDirectory 'images.txt')
if ((Get-FileHash -LiteralPath $info.model_path).Hash.ToLowerInvariant() -ne $info.model_sha256) {
    throw 'Model changed after the firmware build; rebuild before packaging.'
}
Copy-Item -LiteralPath $info.model_path -Destination (Join-Path $OutputDirectory 'model')
foreach ($report in @([IO.Path]::ChangeExtension($info.model_path, '.json'),
                     (Join-Path (Split-Path $info.model_path) 'compilation.json'))) {
    if (Test-Path -LiteralPath $report) {
        Copy-Item -LiteralPath $report -Destination (Join-Path $OutputDirectory 'model')
    }
}
$summaryIndex = 0
foreach ($summary in Get-ChildItem -LiteralPath (Split-Path $info.model_path) `
        -Recurse -Filter '*summary*.csv' -File | Sort-Object FullName) {
    Copy-Item -LiteralPath $summary.FullName `
        -Destination (Join-Path $OutputDirectory "model/summary-$summaryIndex.csv")
    $summaryIndex++
}
Copy-Item -LiteralPath (Join-Path $BuildDirectory 'build-info.json') -Destination $OutputDirectory
& (Join-Path $CompilerBin 'llvm-readelf.exe') -l $axf |
    Set-Content -LiteralPath (Join-Path $OutputDirectory 'validation/elf-segments.txt') -Encoding utf8
if ($LASTEXITCODE -ne 0) { throw 'ELF segment inspection failed.' }
$info.recorded_at_utc = ([DateTimeOffset]$info.recorded_at_utc).ToUniversalTime().ToString('o')
$info | Add-Member -NotePropertyName board_execution_tested -NotePropertyValue $false
$info | Add-Member -NotePropertyName binary_sections_verified -NotePropertyValue $true
$info | ConvertTo-Json -Depth 6 | Set-Content `
    -LiteralPath (Join-Path $OutputDirectory 'manifest.json') -Encoding utf8
$readme = @"
# MPS4 SSE-320 FPGA — $($info.framework), ${cacheKiB} KiB

Use case: $($info.use_case). $($configuration.CMAKE_BUILD_TYPE),
$($configuration.ETHOS_U_NPU_CONFIG_ID), $($configuration.ETHOS_U_NPU_MEMORY_MODE),
PMU $($configuration.ETHOS_U_NPU_PMU_GROUP), TA $($configuration.TA_CONFIG_FILE).

Copy SOFTWARE/boot.bin, bram.bin and ddr.bin into the board's SOFTWARE directory.
Copy images.txt into the active FPGA image directory (for example
MB/HBI0376B/FI101 for revision B). Load addresses are 0x11000000, 0x12000000
and 0x70000000. Eject the drive before starting the board.

debug contains the matching AXF and map. Release AXF follows the project's
symbol-stripping settings. build-info.json records the exact configure/build
commands and model hash. SHA256SUMS.txt verifies the package files.

Binary sections have been checked against AXF. Board execution has not been tested.
"@
Set-Content -LiteralPath (Join-Path $OutputDirectory 'README.md') -Value $readme -Encoding utf8
$hashes = Get-ChildItem -LiteralPath $OutputDirectory -Recurse -File |
    Where-Object Name -ne 'SHA256SUMS.txt' | Sort-Object FullName | ForEach-Object {
        $relative = $_.FullName.Substring($OutputDirectory.Length + 1).Replace('\', '/')
        '{0}  {1}' -f (Get-FileHash -LiteralPath $_.FullName).Hash.ToLowerInvariant(), $relative
    }
Set-Content -LiteralPath (Join-Path $OutputDirectory 'SHA256SUMS.txt') -Value $hashes -Encoding ascii
$zipPath = "$OutputDirectory.zip"
Compress-Archive -Path (Join-Path $OutputDirectory '*') -DestinationPath $zipPath -Force
$zipHash = (Get-FileHash -LiteralPath $zipPath).Hash.ToLowerInvariant()
Set-Content -LiteralPath "$zipPath.sha256" `
    -Value ($zipHash + '  ' + [IO.Path]::GetFileName($zipPath)) -Encoding ascii
Write-Output "FPGA package: $zipPath"
