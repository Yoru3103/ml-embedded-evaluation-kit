# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# Publish the next image to a running QR demo without restarting its model. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$ImagePath,
    [string]$BuildDirectory = 'build/fvp-qrcode/dedicated-384k',
    [string]$Python = 'resources_downloaded/windows-executorch/env/Scripts/python.exe'
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).ProviderPath
function Resolve-QrDemoPath {
    param([string]$Path)
    if (-not [IO.Path]::IsPathRooted($Path)) { $Path = Join-Path $repoRoot $Path }
    return (Resolve-Path -LiteralPath $Path).ProviderPath
}
$BuildDirectory = Resolve-QrDemoPath $BuildDirectory
$Python = Resolve-QrDemoPath $Python
$ImagePath = Resolve-QrDemoPath $ImagePath
$cache = Get-Content -LiteralPath (Join-Path $BuildDirectory 'CMakeCache.txt')
if (-not ($cache -match '^qrcode_INPUT_MODE:[^=]+=SEMIHOST$')) {
    throw 'Build must use qrcode_INPUT_MODE=SEMIHOST.'
}
$dimensions = @{}
foreach ($axis in @('WIDTH', 'HEIGHT')) {
    $line = @($cache | Where-Object { $_ -match "^qrcode_IMAGE_${axis}:[^=]+=([0-9]+)$" })
    if ($line.Count -ne 1) { throw "Missing model $axis in CMakeCache.txt." }
    $dimensions[$axis] = ($line[0] -split '=', 2)[1]
}
$inputDirectory = Join-Path $BuildDirectory 'demo-input'
$saved = @{}
foreach ($name in @('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV')) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    [Environment]::SetEnvironmentVariable($name, $null, 'Process')
}
try {
    & $Python (Join-Path $repoRoot 'scripts/py/prepare_qrcode_demo.py') `
        --input $ImagePath --output (Join-Path $inputDirectory 'qrcode-input.bin') `
        --width $dimensions.WIDTH --height $dimensions.HEIGHT `
        --preview (Join-Path $BuildDirectory 'qrcode-input-preview.png')
    if ($LASTEXITCODE -ne 0) { throw 'QR image preparation failed.' }
} finally {
    foreach ($name in $saved.Keys) {
        [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process')
    }
}
