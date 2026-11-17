# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet('gesture', 'coco', 'img-class')][string]$Demo,
    [ValidateSet('VSI', 'Camera', 'Static')][string]$Mode = 'VSI',
    [ValidateSet('Setup', 'Configure', 'Build', 'All')][string]$Action = 'Build',
    [string]$CompilerBin,
    [string]$BuildDirectory,
    [string]$ModelPath,
    [string]$InputPath,
    [switch]$AutoCapture,
    [switch]$Fast,
    [ValidateRange(1, 128)][int]$Jobs = 4,
    [ValidateSet('FVP', 'FPGA')][string]$Target = 'FVP',
    [string]$LabelsPath
)
$ErrorActionPreference = 'Stop'
if ($Target -eq 'FPGA' -and ($Mode -ne 'Static' -or $Fast)) {
    throw 'MPS4 FPGA builds require -Mode Static and use the timing adapter.'
}
if ($Mode -eq 'Camera' -and $Demo -ne 'img-class') {
    throw 'Semihosted Camera mode is available for img-class. Select VSI for YOLO camera input.'
}
if ($AutoCapture -and $Mode -ne 'VSI') { throw '-AutoCapture requires VSI mode.' }
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).ProviderPath
if (-not $BuildDirectory) {
    $variant = if ($AutoCapture) { 'vsi-auto' } else { $Mode.ToLowerInvariant() }
    if ($Fast) { $variant += '-fast' }
    $BuildDirectory = "build/$($Target.ToLowerInvariant())-$Demo/$variant"
}

$vsi = if ($Mode -eq 'VSI') { 'ON' } else { 'OFF' }
$button = if ($Mode -eq 'VSI' -and -not $AutoCapture) { 'ON' } else { 'OFF' }
$semihost = if ($Mode -eq 'Camera') { 'ON' } else { 'OFF' }
$timing = if ($Fast) { 'OFF' } else { 'ON' }
$options = @{
    Framework='tflm'; Toolchain='ATFE'; Target=$Target; Action=$Action
    BuildDirectory=$BuildDirectory; CompilerBin=$CompilerBin; Jobs=$Jobs; BuildType='Release'
    MemoryMode='Dedicated_Sram'; CacheSizeKiB=384; TimingAdapterConfig='ta_config_u85_sys_dram_mid'
    ActivationBufferSize='0x00200000'
    CMakeArgs=@("-DFVP_VSI_ENABLED=$vsi", "-DSEMIHOSTING_ENABLED=$semihost",
        '-DBUILD_FVP_TESTS=OFF', '-DINTERACTIVE_MODE=OFF', '-DUSE_SINGLE_INPUT=ON',
        '-DMPS4_DISPLAY_ENABLED=ON', "-DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=$timing")
}
if ($Demo -eq 'img-class') {
    $options.UseCase = 'img_class'
    $options.CMakeArgs += @("-Dimg_class_BUTTON_CAPTURE=$button",
        "-Dimg_class_SEMIHOST_INPUT=$semihost")
    if (-not $InputPath) { $InputPath = 'resources/img_class/samples/cat.bmp' }
} else {
    $options.UseCase = 'yolov8_detection'
    $options.CMakeArgs += @("-Dyolov8_detection_CAMERA_INPUT=$vsi",
        "-Dyolov8_detection_BUTTON_CAPTURE=$button", '-Dyolov8_detection_BOX_COORDINATES=normalized',
        '-Dyolov8_detection_INPUT_IS_GRAYSCALE=OFF', '-Dyolov8_detection_OBB=OFF',
        '-Dyolov8_detection_DISPLAY_DOWNSCALE=2', '-Dyolov8_detection_MAX_DETECTIONS=20',
        '-Dyolov8_detection_NMS_THRESHOLD=0.45')
    if ($Demo -eq 'gesture') {
        $gesture = 'resources_downloaded/gesture_detection/exported_33_fulltrain_320'
        if (-not $ModelPath) { $ModelPath = "$gesture/pte_Dedicated_Sram/gesture_33.pte" }
        if (-not $InputPath) { $InputPath = 'resources/gesture_detection/samples' }
        $options.Framework = 'pte'
        if (-not $LabelsPath) { $LabelsPath = "$gesture/labels.txt" }
        $options.NumClasses = 33
        $options.ImageSize = 320
        $options.ActivationBufferSize = '0x00400000'
        $options.CMakeArgs += @('-Dyolov8_detection_SCORE_THRESHOLD=0.25',
            '-Dyolov8_detection_INFER_LAYOUT_FROM_SHAPE=ON',
            '-Dyolov8_detection_ET_METHOD_MEMORY_IN_DDR=ON')
    } else {
        if (-not $ModelPath) {
            $ModelPath = 'resources_downloaded/coco_detection/tflm_Dedicated_Sram/best_int8_vela.tflite'
        }
        if (-not $InputPath) { $InputPath = 'resources/coco_detection/samples/000000058350.jpg' }
        if (-not $LabelsPath) { $LabelsPath = 'resources/coco_detection/coco128.yaml' }
        $options.NumClasses = 80
        $options.ImageSize = 256
        $options.CMakeArgs += '-Dyolov8_detection_SCORE_THRESHOLD=0.1'
    }
}
$options.ModelPath = $ModelPath
if ($LabelsPath) { $options.LabelsPath = $LabelsPath }
$options.InputPath = $InputPath
Push-Location $repoRoot
try {
    & (Join-Path $PSScriptRoot 'build-firmware.ps1') @options
    if ($Mode -ne 'Static' -and $Action -in @('Setup', 'All')) {
        . (Join-Path $PSScriptRoot 'windows-venv.ps1')
        $python = Ensure-WindowsVenv -VenvDir (Join-Path $repoRoot 'resources_downloaded/windows/env')
        & $python -m pip install -r (Join-Path $repoRoot 'scripts/windows/requirements-qrcode-camera.txt')
        if ($LASTEXITCODE -ne 0) { throw 'Failed to install Windows camera requirements.' }
    }
} finally {
    Pop-Location
}
