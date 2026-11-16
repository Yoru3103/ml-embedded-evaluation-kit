# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# Publish one image without restarting the running hand pose model. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$ImagePath,
    [ValidateSet('square', 'letterbox')][string]$Framing = 'square',
    [string]$BuildDirectory = 'build/fvp-hand-pose/dedicated-256k',
    [string]$Python = 'resources_downloaded/windows-executorch/env/Scripts/python.exe'
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).ProviderPath
function Resolve-HandDemoPath {
    param([string]$Path)
    if (-not [IO.Path]::IsPathRooted($Path)) { $Path = Join-Path $repoRoot $Path }
    return (Resolve-Path -LiteralPath $Path).ProviderPath
}
$BuildDirectory = Resolve-HandDemoPath $BuildDirectory
$Python = Resolve-HandDemoPath $Python
$ImagePath = Resolve-HandDemoPath $ImagePath
$cache = Get-Content -LiteralPath (Join-Path $BuildDirectory 'CMakeCache.txt')
if (-not ($cache -match '^yolov8_pose_INPUT_MODE:[^=]+=SEMIHOST$')) {
    throw 'Build must use yolov8_pose_INPUT_MODE=SEMIHOST.'
}
$sizeLine = @($cache | Where-Object { $_ -match '^yolov8_pose_IMAGE_SIZE:[^=]+=([0-9]+)$' })
if ($sizeLine.Count -ne 1) { throw 'Missing hand model image size in CMakeCache.txt.' }
$imageSize = ($sizeLine[0] -split '=', 2)[1]
$saved = @{}
foreach ($name in @('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV')) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    [Environment]::SetEnvironmentVariable($name, $null, 'Process')
}
try {
    & $Python (Join-Path $repoRoot 'scripts/py/prepare_hand_pose_demo.py') `
        --input $ImagePath --output (Join-Path $BuildDirectory 'demo-input/hand-pose-input.bin') `
        --size $imageSize --framing $Framing `
        --preview (Join-Path $BuildDirectory 'hand-pose-input-preview.png')
    if ($LASTEXITCODE -ne 0) { throw 'Hand image preparation failed.' }
} finally {
    foreach ($name in $saved.Keys) {
        [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process')
    }
}
