# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int]$TcpPort = 5000,
    [string]$Prompt
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../../../..')).ProviderPath
$python = Join-Path $repoRoot 'resources_downloaded/windows-executorch/env/Scripts/python.exe'
$clientArgs = @((Join-Path $PSScriptRoot 'fvp_console.py'), '--port', $TcpPort)
if ($PSBoundParameters.ContainsKey('Prompt')) { $clientArgs += @('--prompt', $Prompt) }
& $python @clientArgs
if ($LASTEXITCODE -ne 0) { throw 'FVP prompt console failed; check its error above' }
