# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
<# Build, run and score a paired 48 kHz VoiceBank-DEMAND test set. #>
[CmdletBinding()]
param(
    [string]$DatasetDirectory = 'resources_downloaded/noise_reduction',
    [string]$ModelPath = 'resources_downloaded/linux-batch/dedicated/noise_reduction/rnnoise_INT8_vela_Z1024_Dedicated_Sram.tflite',
    [string]$CompilerBin,
    [string]$BuildDirectory = 'build/fvp-noise-reduction-voicebank',
    [string]$OutputDirectory = 'build/fvp-noise-reduction-voicebank/results',
    [string]$FvpExecutable = 'C:/Program Files/Arm/FVP_Corstone_SSE-320/models/Win64_VC2019/FVP_Corstone_SSE-320.exe',
    [ValidateRange(4, 24)][int]$BatchMegabytes = 16,
    [ValidateRange(1, 128)][int]$Jobs = 8,
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).ProviderPath
Push-Location $repoRoot
try {
    $DatasetDirectory = (Resolve-Path -LiteralPath $DatasetDirectory).ProviderPath
    $ModelPath = (Resolve-Path -LiteralPath $ModelPath).ProviderPath
    $noisyDirectory = Join-Path $DatasetDirectory 'noisy_testset_wav'
    $cleanDirectory = Join-Path $DatasetDirectory 'clean_testset_wav'
    if (-not (Test-Path -LiteralPath $noisyDirectory -PathType Container) -or
        -not (Test-Path -LiteralPath $cleanDirectory -PathType Container)) {
        throw 'The dataset folder must contain noisy_testset_wav and clean_testset_wav.'
    }
    if (Test-Path -LiteralPath $OutputDirectory) {
        throw 'Choose an output directory that does not exist.'
    }
    $noisyFiles = @(Get-ChildItem -LiteralPath $noisyDirectory -File -Filter '*.wav' | Sort-Object Name)
    $cleanNames = @(Get-ChildItem -LiteralPath $cleanDirectory -File -Filter '*.wav' |
        Sort-Object Name | ForEach-Object Name)
    if ($noisyFiles.Count -eq 0 -or
        (($noisyFiles.Name -join [char]0) -cne ($cleanNames -join [char]0))) {
        throw 'Noisy and clean WAV sets must have identical, non-empty sorted filenames.'
    }

    $BuildDirectory = [IO.Path]::GetFullPath($BuildDirectory)
    $OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
    New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
    $stagingRoot = Join-Path $BuildDirectory 'voicebank-staging'
    New-Item -ItemType Directory -Force -Path $stagingRoot | Out-Null
    $batchLimit = $BatchMegabytes * 1MB
    $batches = [Collections.Generic.List[object]]::new()
    $batchFiles = [Collections.Generic.List[object]]::new()
    $batchBytes = 0L
    foreach ($file in $noisyFiles) {
        if ($batchFiles.Count -gt 0 -and $batchBytes + $file.Length -gt $batchLimit) {
            $batches.Add($batchFiles.ToArray())
            $batchFiles.Clear()
            $batchBytes = 0
        }
        $batchFiles.Add($file)
        $batchBytes += $file.Length
    }
    if ($batchFiles.Count -gt 0) { $batches.Add($batchFiles.ToArray()) }

    if ($PlanOnly) {
        Write-Host "$($noisyFiles.Count) paired WAV files, $($batches.Count) FVP batches at $BatchMegabytes MiB each."
        return
    }

    $buildScript = Join-Path $PSScriptRoot 'build-noise-reduction-fvp-demo.ps1'
    $runScript = Join-Path $PSScriptRoot 'run-noise-reduction-fvp-demo.ps1'
    $fileOffset = 0
    for ($batchIndex = 0; $batchIndex -lt $batches.Count; $batchIndex++) {
        $batchName = 'batch-{0:D3}' -f $batchIndex
        $inputPath = Join-Path $stagingRoot "$batchName/input"
        New-Item -ItemType Directory -Path $inputPath | Out-Null
        $files = [object[]]$batches[$batchIndex]
        foreach ($file in $files) {
            New-Item -ItemType HardLink -Path (Join-Path $inputPath $file.Name) -Target $file.FullName | Out-Null
        }
        $sizeMiB = [math]::Round(($files | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
        Write-Host "Building and running $batchName ($($files.Count) files, $sizeMiB MiB)"

        & $buildScript -InputPath $inputPath -ModelPath $ModelPath -CompilerBin $CompilerBin -BuildDirectory $BuildDirectory -Jobs $Jobs
        if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { throw "Firmware build failed for $batchName." }

        $batchOutput = Join-Path $OutputDirectory $batchName
        & $runScript -BuildDirectory $BuildDirectory -OutputDirectory $batchOutput -FvpExecutable $FvpExecutable -Headless -Quiet
        if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { throw "FVP run failed for $batchName." }
        $denoisedFiles = @(Get-ChildItem -LiteralPath $batchOutput -File -Filter 'denoised_loop*.wav' |
            Sort-Object { [int]$_.BaseName.Substring('denoised_loop'.Length) })
        if ($denoisedFiles.Count -ne $files.Count) {
            throw "$batchName produced $($denoisedFiles.Count) outputs for $($files.Count) inputs."
        }
        for ($index = 0; $index -lt $files.Count; $index++) {
            Move-Item -LiteralPath $denoisedFiles[$index].FullName -Destination (Join-Path $OutputDirectory "denoised_$($files[$index].Name)")
        }
        $fileOffset += $files.Count
        Write-Host "Completed $fileOffset/$($noisyFiles.Count) utterances."
    }

    $reportDirectory = Join-Path $OutputDirectory 'report'
    & python (Join-Path $repoRoot 'scripts/py/noise_reduction_dataset_report.py') --noisy-dir $noisyDirectory --clean-dir $cleanDirectory --denoised-dir $OutputDirectory --output-dir $reportDirectory
    if ($LASTEXITCODE -ne 0) { throw 'VoiceBank-DEMAND metric calculation failed.' }
    Write-Host "Open $(Join-Path $reportDirectory 'dataset_report.html')"
} finally {
    Pop-Location
}
