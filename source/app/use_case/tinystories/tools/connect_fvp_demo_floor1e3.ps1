# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int]$TcpPort = 5000,
    [string]$Prompt,
    [switch]$Greedy
)
$ErrorActionPreference = 'Stop'
$forward = @{ TcpPort = $TcpPort }
if ($PSBoundParameters.ContainsKey('Prompt')) { $forward.Prompt = $Prompt }
if ($Greedy) { $forward.Greedy = $true }
& (Join-Path $PSScriptRoot 'connect_fvp_demo.ps1') @forward