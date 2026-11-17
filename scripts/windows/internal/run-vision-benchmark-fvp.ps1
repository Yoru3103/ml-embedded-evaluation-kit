# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<#
.SYNOPSIS
Run one fixed-image vision benchmark in the SSE-320 FVP timing model.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet('gesture', 'coco', 'img-class')][string]$Demo,
    [string]$FvpExecutable = $env:FVP_PATH,
    [switch]$Headless,
    [ValidateRange(1, 86400)][int]$TimeoutSeconds = 300
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).ProviderPath
$build = Join-Path $repoRoot "build/vision-benchmark/$Demo/fvp"
$infoPath = Join-Path $build 'build-info.json'
if (-not (Test-Path -LiteralPath $infoPath -PathType Leaf)) {
    throw "Build the FVP benchmark first: build.ps1 -Profile VisionBenchmark -Demo $Demo -Target FVP"
}
$info = Get-Content -LiteralPath $infoPath -Raw | ConvertFrom-Json
$config = $info.configuration
if ($info.status -ne 'built' -or
    $config.ETHOS_U_NPU_TIMING_ADAPTER_ENABLED -notin @('ON', 'TRUE', '1') -or
    $config.ETHOS_U_NPU_CACHE_SIZE -ne '393216' -or
    $config.ETHOS_U_NPU_MEMORY_MODE -ne 'Dedicated_Sram') {
    throw 'The benchmark requires a completed FVP build with TA and 384 KiB Dedicated SRAM cache.'
}

$options = @{
    Demo=$Demo; Mode='Static'; BuildDirectory=$build; Headless=$Headless
    SmokeTimeoutSeconds=$TimeoutSeconds
}
. (Join-Path $PSScriptRoot 'fvp-tools.ps1')
$options.FvpExecutable = Resolve-WindowsFvpExecutable -Executable $FvpExecutable
& (Join-Path $PSScriptRoot 'run-vision-fvp-demo.ps1') @options
