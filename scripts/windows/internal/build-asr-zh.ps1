# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# Build the static Zipformer use case with selectable U85 SRAM placement. #>
[CmdletBinding()]
param(
    [ValidateSet('FVP', 'FPGA')][string]$Target = 'FVP',
    [ValidateSet('Dedicated_Sram', 'Shared_Sram')][string]$MemoryMode = 'Dedicated_Sram',
    [ValidateSet('Configure', 'Build')][string]$Action = 'Build',
    [string]$ModelPath,
    [string]$InputPath = 'resources/asr_zh/samples/demo.wav',
    [string]$CompilerBin,
    [string]$BuildDirectory,
    [ValidateSet('AUTO', 'OFF', 'SUMMARY', 'DETAILED')][string]$TimingOutput = 'DETAILED',
    [ValidateRange(1, 128)][int]$Jobs = 4,
    [switch]$FixedFeatures,
    [switch]$Int8State,
    [switch]$AllInputs,
    [switch]$NoValidation,
    [switch]$NoTensorDump
)
$ErrorActionPreference = 'Stop'
if ($MemoryMode -eq 'Shared_Sram') {
    $Int8State = $true
}
if (-not $PSBoundParameters.ContainsKey('ModelPath')) {
    if ($MemoryMode -eq 'Shared_Sram') {
        $ModelPath = 'resources_downloaded/asr_zh/int8-state-shared/zipformer_small_ctc_zh_u85_1024_shared_int8_state.pte'
    } elseif ($Int8State) {
        $ModelPath = 'resources_downloaded/asr_zh/int8-state-768k/zipformer_small_ctc_zh_u85_1024_dedicated_768k_int8_state.pte'
    } else {
        $ModelPath = 'resources_downloaded/asr_zh/dedicated-768k/zipformer_small_ctc_zh_u85_1024_dedicated_768k.pte'
    }
}
$root = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).ProviderPath
if (-not $BuildDirectory) {
    $suffix = if ($FixedFeatures) { '-fixed' } else { '' }
    if ($Int8State) { $suffix += '-int8-state' }
    $memoryTag = if ($MemoryMode -eq 'Shared_Sram') { 'shared' } else { '768k' }
    $BuildDirectory = "build/asr_zh/$($Target.ToLowerInvariant())-$memoryTag$suffix"
}
Push-Location $root
try {
    $options = @{
        Action=$Action; Toolchain='ATFE'; Target=$Target; UseCase='asr_zh'
        ModelPath=$ModelPath; InputPath=$InputPath; CompilerBin=$CompilerBin
        BuildDirectory=$BuildDirectory; BuildType='Release'; MemoryMode=$MemoryMode; Jobs=$Jobs
        ActivationBufferSize='0x00800000'
        CMakeArgs=@(
            '-Dasr_zh_INPUT_MODE=STATIC',
            "-Dasr_zh_TIMING_OUTPUT=$TimingOutput",
            "-DCPU_PROFILE_ENABLED=$(if ($TimingOutput -eq 'OFF') {'OFF'} else {'ON'})",
            "-DUSE_SINGLE_INPUT=$(if ($AllInputs) {'OFF'} else {'ON'})",
            '-DINTERACTIVE_MODE=OFF',
            "-DSEMIHOSTING_ENABLED=$(if ($Target -eq 'FVP') {'ON'} else {'OFF'})",
            '-DMPS4_DISPLAY_ENABLED=OFF', '-DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=OFF',
            "-DETHOS_U_NPU_CACHE_SIZE=$(if ($MemoryMode -eq 'Shared_Sram') {'0'} else {'786432'})",
            "-DML_FWK_TMP_MEM_BASE=$(if ($MemoryMode -eq 'Shared_Sram') {'0x31100000'} else {'0x72000000'})",
            "-DML_FWK_TMP_MEM_SIZE=$(if ($MemoryMode -eq 'Shared_Sram') {'0x00300000'} else {'0x00180000'})",
            "-Dasr_zh_FIXED_FEATURES=$(if ($FixedFeatures) {'ON'} else {'OFF'})",
            "-Dasr_zh_INT8_STATE=$(if ($Int8State) {'ON'} else {'OFF'})",
            "-Dasr_zh_DUMP_TENSORS=$(if ($Target -eq 'FVP' -and -not $NoTensorDump) {'ON'} else {'OFF'})",
            "-Dasr_zh_VALIDATE=$(if ($NoValidation) {'OFF'} else {'ON'})"
        )
    }
    & (Join-Path $PSScriptRoot 'build-executorch.ps1') @options
} finally {
    Pop-Location
}
