# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<#
.SYNOPSIS
Build optional wav2letter stage profiling from a caller-supplied TFLite model.
.DESCRIPTION
Reuses the standard Windows build environment. No local reports or previously
extracted build dependencies are required. Use -Package to create an FPGA ZIP.
.EXAMPLE
./scripts/windows/build.ps1 -Profile AsrStageProfile -ModelPath model.tflite -MemoryMode Dedicated_Sram
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$ModelPath,
    [ValidateSet('Dedicated_Sram', 'Shared_Sram')][string]$MemoryMode = 'Dedicated_Sram',
    [ValidateSet('Simple', 'Detailed')][string]$TimingMode = 'Simple',
    [ValidateSet('FPGA', 'FVP')][string]$Target = 'FPGA',
    [ValidateSet('Configure', 'Build')][string]$Action = 'Build',
    [string]$CompilerBin,
    [string]$InputPath = 'resources/asr/samples/another_door.wav',
    [string]$BuildDirectory,
    [string]$PackageDirectory,
    [string]$ActivationBufferSize = '0x00400000',
    [ValidateRange(1, 1048576)][int]$CacheSizeKiB = 256,
    [ValidateRange(1, 128)][int]$Jobs = 4,
    [string[]]$CMakeArgs = @(),
    [switch]$Package
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).ProviderPath
if ($Package -and ($Target -ne 'FPGA' -or $Action -ne 'Build')) {
    throw '-Package requires -Target FPGA -Action Build.'
}
Push-Location $repoRoot
try {
    $tag = "$($MemoryMode.ToLowerInvariant())-$($Target.ToLowerInvariant())-$($TimingMode.ToLowerInvariant())"
    if (-not $BuildDirectory) { $BuildDirectory = "build/asr-stage/tflm-$tag" }
    if (-not [IO.Path]::IsPathRooted($BuildDirectory)) {
        $BuildDirectory = Join-Path $repoRoot $BuildDirectory
    }
    $options = @{
        Framework = 'tflm'; Toolchain = 'ATFE'; Target = $Target; Action = $Action
        UseCase = 'asr'; BuildType = 'Release'; MemoryMode = $MemoryMode; PmuGroup = 'A'
        CompilerBin = $CompilerBin; Jobs = $Jobs; ModelPath = $ModelPath; InputPath = $InputPath
        TimingAdapterConfig = 'ta_config_u85_sys_dram_1024'
        ActivationBufferSize = $ActivationBufferSize; BuildDirectory = $BuildDirectory
        CMakeArgs = @(
            "-DASR_STAGE_PROFILING=$(if ($TimingMode -eq 'Detailed') {'ON'} else {'OFF'})",
            "-DASR_SIMPLE_STAGE_PROFILING=$(if ($TimingMode -eq 'Simple') {'ON'} else {'OFF'})",
            '-DASR_ET_STAGE_PROFILING=OFF') + $CMakeArgs
    }
    if ($MemoryMode -eq 'Dedicated_Sram') { $options.CacheSizeKiB = $CacheSizeKiB }
    & (Join-Path $PSScriptRoot 'build-firmware.ps1') @options
    $flag = if ($TimingMode -eq 'Simple') {'ASR_SIMPLE_STAGE_PROFILING'} else {'ASR_STAGE_PROFILING'}
    if (-not (Select-String -LiteralPath "$BuildDirectory/CMakeCache.txt" `
            -Pattern "^${flag}:BOOL=ON$" -Quiet)) {
        throw 'Stage profiling was not enabled in this build.'
    }
    $buildInfoPath = Join-Path $BuildDirectory 'build-info.json'
    $buildInfo = Get-Content -LiteralPath $buildInfoPath -Raw | ConvertFrom-Json
    $buildInfo | Add-Member -Force -NotePropertyName profiling_output_format `
        -NotePropertyValue $TimingMode.ToLowerInvariant()
    $buildInfo | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $buildInfoPath -Encoding utf8
    if ($Package) {
        . (Join-Path $PSScriptRoot 'windows-venv.ps1')
        $compiler = Resolve-WindowsToolchainBin -Toolchain ATFE -CompilerBin $CompilerBin
        if (-not $PackageDirectory) {
            $PackageDirectory = "build/board-test-packages/asr-stage/tflm-$tag"
        }
        New-Item -ItemType Directory -Force -Path $PackageDirectory | Out-Null
        Copy-Item -LiteralPath 'docs/sections/wav2letter_stage_timing.md' `
            -Destination (Join-Path $PackageDirectory 'TIMING_GUIDE.md')
        & (Join-Path $PSScriptRoot '../package-fpga.ps1') -BuildDirectory $BuildDirectory `
            -CompilerBin $compiler -OutputDirectory $PackageDirectory
    }
} finally {
    Pop-Location
}
