# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# Optional host preprocessing shared by the normal sender and host tests. #>
function Invoke-AsrDenoise {
    param(
        [string]$Path, [double]$ThresholdDbfs=-40, [string]$Source='auto',
        [double]$EdgeSeconds=1, [double]$ReductionDb=6, [string]$Python
    )
    if (-not $Python) {
        $directory = $PSScriptRoot
        while ($directory) {
            $candidate = Join-Path $directory 'resources_downloaded/windows-executorch/env/Scripts/python.exe'
            if (Test-Path -LiteralPath $candidate) { $Python=$candidate; break }
            $directory = Split-Path -Parent $directory
        }
        if (-not $Python) {
            $command = Get-Command python -ErrorAction SilentlyContinue
            if ($command) { $Python=$command.Source }
        }
    }
    if (-not $Python) { throw '-Denoise requires Python with numpy and soundfile; use -DenoisePython.' }
    $script = Join-Path $PSScriptRoot 'denoise_asr_wav.py'
    if (-not (Test-Path -LiteralPath $script)) {
        $script = Join-Path $PSScriptRoot '../py/denoise_asr_wav.py'
    }
    $temporary = Join-Path ([IO.Path]::GetTempPath()) "mlek-asr-denoise-$([guid]::NewGuid())"
    New-Item -ItemType Directory -Path $temporary | Out-Null
    $culture = [Globalization.CultureInfo]::InvariantCulture
    try {
        $null = & $Python -B $script $Path --output-directory $temporary `
            --noise-threshold-dbfs $ThresholdDbfs.ToString($culture) --noise-source $Source `
            --edge-seconds $EdgeSeconds.ToString($culture) `
            --max-reduction-db $ReductionDb.ToString($culture)
        if ($LASTEXITCODE -ne 0) {
            throw 'Denoising failed. Check Python dependencies, WAV format and noise options.'
        }
        $report = Get-Content -LiteralPath (Join-Path $temporary 'report.json') -Raw | ConvertFrom-Json
        $audio = [Mlek.AsrAudio]::Load((Join-Path $temporary 'denoised.wav'))
        return [pscustomobject]@{ Audio=$audio; Report=$report }
    } finally {
        foreach ($name in @('original.wav','denoised.wav','report.json')) {
            Remove-Item -LiteralPath (Join-Path $temporary $name) -Force -ErrorAction SilentlyContinue
        }
        Remove-Item -LiteralPath $temporary -ErrorAction SilentlyContinue
    }
}
