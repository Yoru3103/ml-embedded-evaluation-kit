# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
<# Build the SSE-320 RNNoise WAV export demo. #>
[CmdletBinding()]
param(
    [string]$InputPath = 'resources/noise_reduction/samples/p232_113.wav',
    [string]$ModelPath,
    [string]$CompilerBin,
    [string]$BuildDirectory = 'build/fvp-noise-reduction',
    [ValidateSet('AC6', 'ATFE')][string]$Toolchain = 'ATFE',
    [ValidateSet('Setup', 'Configure', 'Build', 'All')][string]$Action = 'Build',
    [ValidateRange(1, 128)][int]$Jobs = 4,
    [string[]]$CMakeArgs = @()
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).ProviderPath
$options = @{
    Framework='tflm'; Toolchain=$Toolchain; Target='FVP'; Action=$Action
    UseCase='noise_reduction'; InputPath=$InputPath; ModelPath=$ModelPath
    CompilerBin=$CompilerBin; BuildDirectory=$BuildDirectory; BuildType='Release'; Jobs=$Jobs
    MemoryMode='Dedicated_Sram'
    CMakeArgs=@('-DSEMIHOSTING_ENABLED=ON', '-Dnoise_reduction_SAVE_WAV=ON',
        '-Dnoise_reduction_AUDIO_DURATION=0', '-Dnoise_reduction_AUDIO_OFFSET=0',
        '-Dnoise_reduction_AUDIO_RATE=48000', '-Dnoise_reduction_AUDIO_MONO=ON',
        '-DUSE_SINGLE_INPUT=OFF', '-DINTERACTIVE_MODE=OFF', '-DHAL_AUDIO_LOOP=OFF',
        '-DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=OFF', '-DFVP_VSI_ENABLED=OFF') + $CMakeArgs
}
Push-Location $repoRoot
try { & (Join-Path $PSScriptRoot 'build-firmware.ps1') @options }
finally { Pop-Location }
