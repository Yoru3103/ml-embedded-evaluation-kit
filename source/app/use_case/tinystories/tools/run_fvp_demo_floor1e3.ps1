# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [ValidateSet(64, 128)][int]$Context = 64,
    [ValidateRange(1, 65532)][int]$TcpPort = 5000
)
$ErrorActionPreference = 'Stop'
$build = "build/tinystories42m/fvp-demo/demo-ctx$Context"
& (Join-Path $PSScriptRoot 'run_fvp_demo.ps1') -BuildDirectory $build -TcpPort $TcpPort