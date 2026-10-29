# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# Create original/denoised WAV copies and a profile-selection report. No upload. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$WavPath,
    [string]$OutputDirectory = 'build/asr-denoise',
    [ValidateRange(-100, -20)][double]$NoiseThresholdDbfs = -40,
    [ValidateSet('auto', 'head', 'tail')][string]$NoiseSource = 'auto',
    [ValidateRange(0.2, 5)][double]$EdgeSeconds = 1,
    [ValidateRange(0, 12)][double]$MaxReductionDb = 6
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).ProviderPath
$python = Join-Path $repoRoot 'resources_downloaded/windows-executorch/env/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Requires the existing Windows Python environment with numpy and soundfile.'
}
$inputPath = (Resolve-Path -LiteralPath $WavPath).ProviderPath
if (-not [IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory = Join-Path $repoRoot $OutputDirectory
}
$culture = [Globalization.CultureInfo]::InvariantCulture
& $python -B (Join-Path $repoRoot 'scripts/py/denoise_asr_wav.py') $inputPath `
    --output-directory $OutputDirectory --noise-source $NoiseSource `
    --noise-threshold-dbfs $NoiseThresholdDbfs.ToString($culture) `
    --edge-seconds $EdgeSeconds.ToString($culture) `
    --max-reduction-db $MaxReductionDb.ToString($culture)
if ($LASTEXITCODE -ne 0) { throw 'ASR denoising experiment failed.' }
