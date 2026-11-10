# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [string]$CompilerBin = 'C:/Program Files/Arm/Development Studio 2026.0-1/sw/atfe-pro22.1.0/bin',
    [string]$BuildDirectory = 'build/tinystories42m/demo/fvp',
    [ValidateRange(1, 64)][int]$Jobs = 8
)
& (Join-Path $PSScriptRoot 'build_fpga_demo.ps1') -Target FVP -CompilerBin $CompilerBin `
    -BuildDirectory $BuildDirectory -Jobs $Jobs
