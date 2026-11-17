# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# Default to camera preview; Space submits an RGB frame to the resident hand model. #>
[CmdletBinding(DefaultParameterSetName='Camera')]
param(
    [Parameter(ParameterSetName='Camera')][switch]$Camera,
    [Parameter(Mandatory=$true, ParameterSetName='Image')][string]$ImagePath,
    [Parameter(ParameterSetName='Camera')][ValidateRange(0, 32)][int]$CameraIndex = 0,
    [Parameter(ParameterSetName='Camera')][ValidateSet('auto', 'dshow', 'msmf')]
    [string]$CameraBackend = 'dshow',
    [ValidateSet('square', 'letterbox')][string]$Framing = 'square',
    [string]$BuildDirectory = 'build/fvp-hand-pose/dedicated-256k',
    [string]$Python = 'resources_downloaded/windows-executorch/env/Scripts/python.exe',
    [string]$FvpExecutable = 'C:/Program Files/Arm/FVP_Corstone_SSE-320/models/Win64_VC2019/FVP_Corstone_SSE-320.exe',
    [switch]$Headless,
    [ValidateRange(0, 86400)][int]$SmokeTimeoutSeconds = 0
)
$ErrorActionPreference = 'Stop'
$options = @{
    Demo='hand-pose'; BuildDirectory=$BuildDirectory; Python=$Python
    FvpExecutable=$FvpExecutable; Headless=$Headless; SmokeTimeoutSeconds=$SmokeTimeoutSeconds
    CaptureFraming=$Framing
}
if ($PSCmdlet.ParameterSetName -eq 'Image') { $options.ImagePath = $ImagePath }
else {
    $options.Camera = $true
    $options.CameraIndex = $CameraIndex
    $options.CameraBackend = $CameraBackend
}
& (Join-Path $PSScriptRoot 'run-qrcode-fvp-demo.ps1') @options
