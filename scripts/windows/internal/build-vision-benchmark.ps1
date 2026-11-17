# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<#
.SYNOPSIS
Build fixed-image vision benchmarks and package the MPS4 firmware.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet('gesture', 'coco', 'img-class')][string]$Demo,
    [ValidateSet('Both', 'FVP', 'MPS4')][string]$Target = 'Both',
    [string]$InputPath,
    [string]$ModelPath,
    [string]$LabelsPath,
    [string]$CompilerBin,
    [ValidateRange(1, 128)][int]$Jobs = 4
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).ProviderPath
. (Join-Path $PSScriptRoot 'windows-venv.ps1')

Push-Location $repoRoot
try {
    $CompilerBin = Resolve-WindowsToolchainBin -Toolchain ATFE -CompilerBin $CompilerBin
    if (-not $InputPath -and $Demo -eq 'gesture') {
        $InputPath = 'resources/gesture_detection/samples/like_fd70802a-a87f-400c-9ccb-583bf1c43b53.jpg'
    }
    if (-not $ModelPath -and $Demo -eq 'img-class') {
        $ModelPath = 'resources_downloaded/windows/img_class/' +
            'mobilenet_v2_1.0_224_INT8_vela_Z1024_Dedicated_Sram.tflite'
    }

    $root = "build/vision-benchmark/$Demo"
    $options = @{
        Demo=$Demo; Mode='Static'; CompilerBin=$CompilerBin
        InputPath=$InputPath; ModelPath=$ModelPath; LabelsPath=$LabelsPath; Jobs=$Jobs
    }
    $buildScript = Join-Path $PSScriptRoot 'build-vision-fvp-demo.ps1'
    if ($Target -in @('Both', 'FVP')) {
        & $buildScript @options -Target FVP -BuildDirectory "$root/fvp"
    }
    if ($Target -in @('Both', 'MPS4')) {
        & $buildScript @options -Target FPGA -BuildDirectory "$root/mps4"
        & (Join-Path $PSScriptRoot '../package-fpga.ps1') -CompilerBin $CompilerBin `
            -BuildDirectory "$root/mps4" -OutputDirectory "$root/mps4-package"
    }
} finally {
    Pop-Location
}
