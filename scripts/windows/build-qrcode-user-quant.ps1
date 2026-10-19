# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<#
.SYNOPSIS
Build the supplied native NHWC QRCode models without recompiling or changing them.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet('tflm', 'pte')][string]$Framework,
    [ValidateSet('usr_quant_nhwc', 'usr_quant_nhwc_angle3d')]
    [string]$Variant = 'usr_quant_nhwc',
    [ValidateSet('FPGA', 'FVP')][string]$Target = 'FPGA',
    [switch]$SingleInput,
    [string]$CompilerBin,
    [ValidateRange(1, 32)][int]$Jobs = 4
)
$ErrorActionPreference = 'Stop'
$taskRepo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).ProviderPath
$taskRoot = Join-Path $taskRepo "build/qrcode_gray_obb/$Variant"
$taskResources = Join-Path $taskRepo "resources_downloaded/yolo_QRcode/$Variant"
$taskBuild = Join-Path $taskRoot "$Framework-$($Target.ToLowerInvariant())"
New-Item -ItemType Directory -Force -Path $taskBuild | Out-Null
$taskSingle = if ($SingleInput) { 'ON' } else { 'OFF' }
$taskDefinitions = @(
    '-Dyolov8_detection_DUMP_RAW_OUTPUTS=OFF',
    '-Dyolov8_detection_PROFILE_STAGES=ON', '-Dyolov8_detection_PROFILE_POSTPROCESS=ON',
    '-Dyolov8_detection_INFER_LAYOUT_FROM_SHAPE=ON',
    '-Dyolov8_detection_INPUT_IS_GRAYSCALE=ON',
    '-Dyolov8_detection_MAX_DETECTIONS=5', '-Dyolov8_detection_DFL_BINS=8',
    '-Dyolov8_detection_OBB=ON', "-DUSE_SINGLE_INPUT=$taskSingle",
    '-DINTERACTIVE_MODE=OFF', '-DSEMIHOSTING_ENABLED=OFF',
    '-DETHOS_U_NPU_ID=U85', '-DETHOS_U_NPU_CONFIG_ID=Z1024',
    '-DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=ON'
)
if ($Framework -eq 'pte') {
    $taskPteName = if ($Variant -eq 'usr_quant_nhwc_angle3d') {
        'best_bifpn_gray_nhwc_angle3d_u85_1024_384KiB.pte'
    } else {
        'best_bifpn_gray_nhwc_u85_1024_384KiB.pte'
    }
    $taskModel = Join-Path $taskResources "pte/$taskPteName"
    $taskDefinitions += '-Dyolov8_detection_ET_METHOD_MEMORY_IN_DDR=ON'
} else {
    $taskModel = Join-Path $taskResources 'tflite/vela/MPS4_AN101_TA_VM0_384KiB/model_int8_vela.tflite'
}
$taskOptions = @{
    Framework=$Framework; Toolchain='ATFE'; Target=$Target; Action='Build';
    UseCase='yolov8_detection'; CompilerBin=$CompilerBin; ModelPath=$taskModel;
    InputPath=$(if ($SingleInput) {
        Join-Path $taskRepo 'build/qrcode_gray_obb/samples/1.bmp'
    } else {
        Join-Path $taskRepo 'build/qrcode_gray_obb/samples'
    });
    LabelsPath=(Join-Path $taskRepo 'resources_downloaded/yolo_QRcode/gray_qoperator_384k/labels.txt');
    ImageSize=256; NumClasses=15; ActivationBufferSize='0x00400000';
    MemoryMode='Dedicated_Sram'; BuildType='Release'; CacheSizeKiB=384;
    TimingAdapterConfig='ta_config_u85_sys_dram_mid'; PmuGroup='BW'; Jobs=$Jobs;
    BuildDirectory=$taskBuild; CMakeArgs=$taskDefinitions
}
& (Join-Path $PSScriptRoot 'build.ps1') @taskOptions
