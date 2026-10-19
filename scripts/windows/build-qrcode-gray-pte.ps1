# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<#
.SYNOPSIS
Build the checked gray PTE with the same preprocessing/Invoke boundary as TFLM.
#>
[CmdletBinding()]
param(
    [ValidateSet('FVP', 'FPGA')][string]$Target = 'FPGA',
    [switch]$DumpRawOutputs,
    [string]$CompilerBin,
    [ValidateRange(1, 32)][int]$Jobs = 8
)
$ErrorActionPreference = 'Stop'
$taskRepo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).ProviderPath
$taskRoot = Join-Path $taskRepo 'build/qrcode_gray_obb'
$taskSuffix = if ($DumpRawOutputs) { '-raw' } else { '' }
$taskBuild = Join-Path $taskRoot "pte-$($Target.ToLowerInvariant())-384k$taskSuffix"
$taskRaw = if ($DumpRawOutputs) { 'ON' } else { 'OFF' }
$taskDefinitions = @(
    "-Dyolov8_detection_DUMP_RAW_OUTPUTS=$taskRaw",
    '-Dyolov8_detection_PROFILE_POSTPROCESS=ON',
    '-Dyolov8_detection_PROFILE_STAGES=ON',
    '-Dyolov8_detection_INPUT_IS_GRAYSCALE=ON',
    '-Dyolov8_detection_ET_METHOD_MEMORY_IN_DDR=ON',
    '-Dyolov8_detection_MAX_DETECTIONS=5',
    '-Dyolov8_detection_DFL_BINS=8',
    '-Dyolov8_detection_OBB=ON',
    '-DUSE_SINGLE_INPUT=OFF',
    '-DINTERACTIVE_MODE=OFF', '-DSEMIHOSTING_ENABLED=OFF',
    '-DETHOS_U_NPU_ID=U85', '-DETHOS_U_NPU_CONFIG_ID=Z1024',
    '-DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=ON'
)
$taskOptions = @{
    Framework='pte'; Toolchain='ATFE'; Target=$Target; Action='Build';
    UseCase='yolov8_detection'; CompilerBin=$CompilerBin;
    ModelPath=(Join-Path $taskRoot 'pte_output_dq/natural_output_dq.pte');
    InputPath=(Join-Path $taskRoot 'samples');
    LabelsPath=(Join-Path $taskRepo 'resources_downloaded/yolo_QRcode/gray_qoperator_384k/labels.txt');
    ImageSize=256; NumClasses=15; ActivationBufferSize='0x00400000';
    MemoryMode='Dedicated_Sram'; BuildType='Release'; CacheSizeKiB=384;
    TimingAdapterConfig='ta_config_u85_sys_dram_mid'; PmuGroup='BW'; Jobs=$Jobs;
    BuildDirectory=$taskBuild; CMakeArgs=$taskDefinitions
}
& (Join-Path $PSScriptRoot 'build.ps1') @taskOptions
