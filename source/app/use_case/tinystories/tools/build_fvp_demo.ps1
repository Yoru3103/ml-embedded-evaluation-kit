# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [string]$CompilerBin = 'C:/Program Files/Arm/Development Studio 2026.0-1/sw/atfe-pro22.1.0/bin',
    [string]$BuildDirectory,
    [string]$ModelPath,
    [ValidateRange(1, 1024)][int]$Context = 128,
    [ValidateRange(1, 64)][int]$Jobs = 8,
    [switch]$TimingAdapter,
    [switch]$LogitDump
)
$forward = @{
    Target = 'FVP'; Mode = 'DEMO'; CompilerBin = $CompilerBin; Jobs = $Jobs; Context = $Context
}
if ($BuildDirectory) { $forward.BuildDirectory = $BuildDirectory }
if ($ModelPath) { $forward.ModelPath = $ModelPath }
if ($TimingAdapter) { $forward.TimingAdapter = $true }
if ($LogitDump) { $forward.LogitDump = $true }
& (Join-Path $PSScriptRoot 'build_fpga_demo.ps1') @forward
