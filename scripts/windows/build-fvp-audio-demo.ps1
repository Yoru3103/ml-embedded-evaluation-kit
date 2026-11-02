# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# Build the SSE-320 U85-1024 Dedicated256 custom VSI audio demo. #>
[CmdletBinding()]
param(
    [ValidateSet('kws_zh', 'asr')][string]$UseCase = 'kws_zh',
    [ValidateSet('VSI', 'STATIC')][string]$InputMode = 'VSI',
    [ValidateSet('Configure', 'Build')][string]$Action = 'Build',
    [string]$ModelPath,
    [string]$InputPath,
    [string]$CompilerBin,
    [string]$BuildDirectory,
    [ValidateRange(1, 32767)][int]$VadRms = 800,
    [ValidateRange(20, 5000)][int]$SilenceMs = 1000,
    [ValidateRange(200, 2000)][int]$PreRollMs = 600,
    [ValidateRange(1, 128)][int]$Jobs = 4
)
$ErrorActionPreference = 'Stop'
if ($UseCase -ne 'asr' -and
    ($PSBoundParameters.ContainsKey('VadRms') -or $PSBoundParameters.ContainsKey('SilenceMs') -or
        $PSBoundParameters.ContainsKey('PreRollMs'))) {
    throw '-VadRms, -SilenceMs and -PreRollMs apply only to -UseCase asr.'
}
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).ProviderPath
if (-not $BuildDirectory) {
    $BuildDirectory = "build/fvp-audio/$UseCase"
    if ($InputMode -ne 'VSI') { $BuildDirectory += '-static' }
}
if (-not $ModelPath) {
    $ModelPath = if ($UseCase -eq 'kws_zh') {
        'resources_downloaded/kws_zh/pte_Dedicated_Sram_256k/wekws_dstcn_u85.pte'
    } else {
        'resources_downloaded/asr/et151-u85-1024-dedicated-256k/' +
        'conformer_fp32_cln_wer_6_47_arm_delegate_ethos-u85-1024_Dedicated_Sram.pte'
    }
}
if (-not $InputPath) {
    $InputPath = if ($UseCase -eq 'kws_zh') { 'resources/kws_zh/samples' }
        else { 'resources/asr/samples/another_door.wav' }
}
$options = @{
    Framework='pte'; Toolchain='ATFE'; Target='FVP'; Action=$Action; UseCase=$UseCase
    MemoryMode='Dedicated_Sram'; CacheSizeKiB=256; BuildType='Release'; Jobs=$Jobs
    ModelPath=$ModelPath; InputPath=$InputPath; CompilerBin=$CompilerBin
    BuildDirectory=$BuildDirectory
    CMakeArgs=@("-D${UseCase}_INPUT_MODE=$InputMode", '-DSEMIHOSTING_ENABLED=OFF',
        '-DMPS4_DISPLAY_ENABLED=ON', '-DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=OFF',
        '-DINTERACTIVE_MODE=OFF', '-DFVP_VSI_ENABLED=OFF',
        '-DHDLCD_FRAME_BUFFER_BASE_ADDRESS=0x74000000')
}
if ($UseCase -eq 'kws_zh') {
    $options.CMakeArgs += @('-DML_FWK_TMP_MEM_BASE=0x72000000',
        '-DML_FWK_TMP_MEM_SIZE=0x00300000', '-Dkws_zh_STREAM_HOP_SAMPLES=16000')
} else {
    $options.ActivationBufferSize = '0x00200000'
    $options.CMakeArgs += @('-DML_FWK_TMP_MEM_BASE=0x94000000',
        '-DML_FWK_TMP_MEM_SIZE=0x02000000', '-DASR_ET_STAGE_PROFILING=OFF',
        '-DASR_STAGE_PROFILING=OFF', '-DASR_SIMPLE_STAGE_PROFILING=OFF',
        "-Dasr_VSI_VAD_RMS=$VadRms", "-Dasr_VSI_SILENCE_MS=$SilenceMs",
        "-Dasr_VSI_PRE_ROLL_MS=$PreRollMs")
}
Push-Location $repoRoot
try {
    & (Join-Path $PSScriptRoot 'build.ps1') @options
} finally {
    Pop-Location
}
