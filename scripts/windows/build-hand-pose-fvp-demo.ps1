# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# Build the hand pose camera demo for FVP, or the static-image FPGA application. #>
[CmdletBinding()]
param(
    [string]$ModelPath = 'resources_downloaded/yolov8_pose/hand_pose_u85-1024_ded_256KiB_mixed.pte',
    [string]$BuildDirectory,
    [ValidateSet('FVP', 'FPGA')][string]$Target = 'FVP',
    [string]$CompilerBin,
    [ValidateSet('Configure', 'Build')][string]$Action = 'Build',
    [ValidateRange(32, 1024)][int]$ImageSize = 224,
    [ValidateRange(1, 128)][int]$Jobs = 4
)
$ErrorActionPreference = 'Stop'
if ($ImageSize % 32) { throw 'ImageSize must be a multiple of 32.' }
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).ProviderPath
if (-not $BuildDirectory) {
    $BuildDirectory = if ($Target -eq 'FVP') { 'build/fvp-hand-pose/dedicated-256k' }
        else { 'build/fpga-hand-pose/dedicated-256k' }
}
$inputMode = if ($Target -eq 'FVP') { 'SEMIHOST' } else { 'STATIC' }
$semihost = if ($Target -eq 'FVP') { 'ON' } else { 'OFF' }
$timing = if ($Target -eq 'FVP') { 'OFF' } else { 'ON' }
$temporaryBase = if ($Target -eq 'FVP') { '0x94000000' } else { '0x72000000' }
$imageLimit = if ($Target -eq 'FVP') { 224.0 } else { 192.0 }
$downscale = [Math]::Max(1, [Math]::Ceiling($ImageSize / $imageLimit))
$options = @{
    Framework='pte'; Toolchain='ATFE'; Target=$Target; Action=$Action; UseCase='yolov8_pose'
    ModelPath=$ModelPath; BuildDirectory=$BuildDirectory; CompilerBin=$CompilerBin
    MemoryMode='Dedicated_Sram'; CacheSizeKiB=256; BuildType='Release'; Jobs=$Jobs
    ImageSize=$ImageSize; ActivationBufferSize='0x00200000'
    InputPath='resources/yolov8_pose/samples/test03.jpg'
    CMakeArgs=@("-Dyolov8_pose_INPUT_MODE=$inputMode", "-DSEMIHOSTING_ENABLED=$semihost",
        '-DMPS4_DISPLAY_ENABLED=ON', '-DFVP_VSI_ENABLED=OFF', '-DINTERACTIVE_MODE=OFF',
        "-DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=$timing", '-Dyolov8_pose_MAX_DETECTIONS=2',
        '-Dyolov8_pose_OUTPUT_TOPOLOGY=SPLIT', '-Dyolov8_pose_COORDINATES_NORMALIZED=OFF',
        "-Dyolov8_pose_DISPLAY_DOWNSCALE=$downscale", '-Dyolov8_pose_PMU_WARMUP_RUNS=0',
        '-Dyolov8_pose_PMU_MEASURED_RUNS=1',
        "-DML_FWK_TMP_MEM_BASE=$temporaryBase", '-DML_FWK_TMP_MEM_SIZE=0x02000000',
        '-DHDLCD_FRAME_BUFFER_BASE_ADDRESS=0x74000000')
}
Push-Location $repoRoot
try { & (Join-Path $PSScriptRoot 'build.ps1') @options }
finally { Pop-Location }
