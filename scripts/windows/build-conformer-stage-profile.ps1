# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<#
.SYNOPSIS
Build optional Conformer six-stage profiling from a caller-supplied PTE.
.DESCRIPTION
The model must target U85-1024 and match MemoryMode and the dedicated cache
budget. Shared SRAM uses the MPS4 4 MiB layout: 1 MiB method pool and 3 MiB
scratch pool. No dated export directory or local report is required.
.EXAMPLE
./scripts/windows/build-conformer-stage-profile.ps1 -ModelPath model.pte -MemoryMode Shared_Sram
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$ModelPath,
    [ValidateSet('Dedicated_Sram', 'Shared_Sram')][string]$MemoryMode = 'Dedicated_Sram',
    [ValidateSet('FPGA', 'FVP')][string]$Target = 'FPGA',
    [ValidateSet('Configure', 'Build')][string]$Action = 'Build',
    [string]$CompilerBin,
    [string]$InputPath = 'resources/asr/samples/another_door.wav',
    [string]$BuildDirectory,
    [string]$PackageDirectory,
    [ValidateRange(1, 1048576)][int]$CacheSizeKiB = 256,
    [ValidateRange(1, 128)][int]$Jobs = 4,
    [string[]]$CMakeArgs = @(),
    [switch]$Package
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).ProviderPath
if ($Package -and ($Target -ne 'FPGA' -or $Action -ne 'Build')) {
    throw '-Package requires -Target FPGA -Action Build.'
}
Push-Location $repoRoot
try {
    $tag = "$($MemoryMode.ToLowerInvariant())-$($Target.ToLowerInvariant())"
    if (-not $BuildDirectory) { $BuildDirectory = "build/asr-stage/executorch-$tag" }
    if (-not [IO.Path]::IsPathRooted($BuildDirectory)) {
        $BuildDirectory = Join-Path $repoRoot $BuildDirectory
    }
    $options = @{
        Action = $Action; Toolchain = 'ATFE'; Target = $Target; UseCase = 'asr'
        BuildType = 'Release'; PmuGroup = 'A'; Jobs = $Jobs; CompilerBin = $CompilerBin
        ModelPath = $ModelPath; InputPath = $InputPath; MemoryMode = $MemoryMode
        TimingAdapterConfig = 'ta_config_u85_sys_dram_1024'; BuildDirectory = $BuildDirectory
        CMakeArgs = @('-DASR_ET_STAGE_PROFILING=ON', '-DASR_STAGE_PROFILING=OFF',
            '-DINTERACTIVE_MODE=OFF',
            "-DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=$(if ($Target -eq 'FPGA') {'ON'} else {'OFF'})")
    }
    if ($MemoryMode -eq 'Shared_Sram') {
        $options.ActivationBufferSize = '0x00100000'
        $options.CMakeArgs += @('-DETHOS_U_NPU_CACHE_SIZE=0',
            '-DML_FWK_TMP_MEM_BASE=0x31100000', '-DML_FWK_TMP_MEM_SIZE=0x00300000',
            '-DHDLCD_FRAME_BUFFER_BASE_ADDRESS=0x74000000')
    } else {
        $options.CacheSizeKiB = $CacheSizeKiB
    }
    $options.CMakeArgs += $CMakeArgs
    & (Join-Path $PSScriptRoot 'build-executorch.ps1') @options
    if (-not (Select-String -LiteralPath "$BuildDirectory/CMakeCache.txt" `
            -Pattern '^ASR_ET_STAGE_PROFILING:BOOL=ON$' -Quiet)) {
        throw 'Stage profiling was not enabled in this build.'
    }
    if ($Package) {
        . (Join-Path $PSScriptRoot 'windows-venv.ps1')
        $compiler = Resolve-WindowsToolchainBin -Toolchain ATFE -CompilerBin $CompilerBin
        if (-not $PackageDirectory) {
            $PackageDirectory = "build/board-test-packages/asr-stage/executorch-$tag"
        }
        & (Join-Path $PSScriptRoot 'package-fpga.ps1') -BuildDirectory $BuildDirectory `
            -CompilerBin $compiler -OutputDirectory $PackageDirectory
    }
} finally {
    Pop-Location
}
