# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# Build a resident WAV/image UART file receiver, also executable on FVP for validation. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet('kws_zh','asr','asr_zh','hand-pose','qrcode')][string]$UseCase,
    [ValidateSet('FPGA','FVP')][string]$Target = 'FPGA',
    [string]$BuildDirectory,
    [string]$ModelPath,
    [string]$CompilerBin,
    [ValidateSet('Configure','Build')][string]$Action = 'Build',
    [ValidateRange(1,128)][int]$Jobs = 4
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).ProviderPath
$case = if ($UseCase -eq 'hand-pose') { 'yolov8_pose' } else { $UseCase }
if (-not $BuildDirectory) { $BuildDirectory = "build/$($Target.ToLowerInvariant())-file/$UseCase" }
$models = @{
    kws_zh='resources_downloaded/kws_zh/pte_Dedicated_Sram_256k/wekws_dstcn_u85.pte'
    asr_zh='resources_downloaded/asr_zh/int8-state-768k/zipformer_small_ctc_zh_u85_1024_dedicated_768k_int8_state.pte'
    asr='resources_downloaded/asr/et151-u85-1024-dedicated-256k/conformer_fp32_cln_wer_6_47_arm_delegate_ethos-u85-1024_Dedicated_Sram.pte'
    'hand-pose'='resources_downloaded/yolov8_pose/hand_pose_u85-1024_ded_256KiB_mixed.pte'
    qrcode='resources_downloaded/yolo_QRcode/usr_quant_nhwc_angle3d/pte/best_bifpn_gray_nhwc_angle3d_u85_1024_384KiB.pte'
}
if (-not $ModelPath) { $ModelPath = $models[$UseCase] }
$temporaryBase = if ($Target -eq 'FPGA') { '0x72000000' } else { '0x94000000' }
$timing = if ($Target -eq 'FPGA') { 'ON' } else { 'OFF' }
$options = @{
    Framework='pte'; Toolchain='ATFE'; Target=$Target; Action=$Action; UseCase=$case
    ModelPath=$ModelPath; BuildDirectory=$BuildDirectory; CompilerBin=$CompilerBin
    MemoryMode='Dedicated_Sram'; CacheSizeKiB=256; BuildType='Release'; Jobs=$Jobs
    ActivationBufferSize='0x00200000'; PmuGroup='A'
    CMakeArgs=@("-D${case}_INPUT_MODE=UART", '-DSEMIHOSTING_ENABLED=OFF',
        '-DMPS4_DISPLAY_ENABLED=ON', '-DFVP_VSI_ENABLED=OFF', '-DINTERACTIVE_MODE=OFF',
        "-DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=$timing", "-DML_FWK_TMP_MEM_BASE=$temporaryBase",
        '-DML_FWK_TMP_MEM_SIZE=0x02000000', '-DHDLCD_FRAME_BUFFER_BASE_ADDRESS=0x74000000')
}
switch ($UseCase) {
    'kws_zh' { $options.InputPath='resources/kws_zh/samples' }
    'asr_zh' {
        $options.InputPath='resources/asr_zh/samples/demo.wav'
        $options.CacheSizeKiB=768; $options.ActivationBufferSize='0x00800000'
        $options.CMakeArgs += @('-DCPU_PROFILE_ENABLED=ON', '-Dasr_zh_INT8_STATE=ON',
            '-Dasr_zh_FIXED_FEATURES=OFF', '-Dasr_zh_VALIDATE=OFF', '-Dasr_zh_DUMP_TENSORS=OFF',
            '-DML_FWK_TMP_MEM_BASE=0x76000000', '-DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=OFF')
    }
    'asr' {
        $options.InputPath='resources/asr/samples/another_door.wav'
        $options.CMakeArgs += @('-DASR_ET_STAGE_PROFILING=ON', '-DASR_STAGE_PROFILING=OFF',
            '-DASR_SIMPLE_STAGE_PROFILING=OFF')
    }
    'hand-pose' {
        $options.InputPath='resources/yolov8_pose/samples/test03.jpg'; $options.ImageSize=224
        $options.CMakeArgs += @('-Dyolov8_pose_MAX_DETECTIONS=2',
            '-Dyolov8_pose_OUTPUT_TOPOLOGY=SPLIT', '-Dyolov8_pose_COORDINATES_NORMALIZED=OFF',
            '-Dyolov8_pose_DISPLAY_DOWNSCALE=1', '-Dyolov8_pose_PMU_WARMUP_RUNS=0',
            '-Dyolov8_pose_PMU_MEASURED_RUNS=1')
    }
    'qrcode' {
        $options.InputPath='resources/qrcode/smaples/01.bmp'; $options.CacheSizeKiB=384
        $options.ActivationBufferSize='0x00400000'
        $options.CMakeArgs += @('-Dqrcode_ET_METHOD_MEMORY_IN_DDR=ON',
            '-Dqrcode_MAX_DETECTIONS=5', '-Dqrcode_IMAGE_WIDTH=256', '-Dqrcode_IMAGE_HEIGHT=256',
            '-Dqrcode_DISPLAY_DOWNSCALE=2', '-Dqrcode_IMAGE_RESIZE_MODE=letterbox')
    }
}
Push-Location $repoRoot
try { & (Join-Path $PSScriptRoot 'build.ps1') @options }
finally { Pop-Location }
