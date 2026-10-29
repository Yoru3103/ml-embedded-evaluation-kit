# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# Build the independent MPS4 Conformer UART demo, Dedicated SRAM 256 KiB only. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$ModelPath,
    [string]$CompilerBin,
    [ValidateSet('Configure', 'Build')][string]$Action = 'Build',
    [ValidateSet('FPGA', 'FVP')][string]$Target = 'FPGA',
    [ValidateSet(115200, 460800, 921600)][int]$BaudRate = 115200,
    [string]$BuildDirectory,
    [string]$PackageDirectory = 'build/board-test-packages/asr-uart/dedicated-256k',
    [ValidateRange(1, 128)][int]$Jobs = 4,
    [switch]$Package
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).ProviderPath
if ($Package -and ($Action -ne 'Build' -or $Target -ne 'FPGA')) {
    throw '-Package requires -Action Build -Target FPGA.'
}
if (-not $BuildDirectory) {
    $BuildDirectory = if ($Target -eq 'FPGA') { 'build/asr-uart/dedicated-256k' }
        else { 'build/asr-uart/dedicated-256k-fvp' }
}
if (-not [IO.Path]::IsPathRooted($PackageDirectory)) {
    $PackageDirectory = Join-Path $repoRoot $PackageDirectory
}
if ($Package) {
    $hostDirectory = Join-Path $PackageDirectory 'host'
    New-Item -ItemType Directory -Force -Path $hostDirectory | Out-Null
    foreach ($file in @('send-asr-audio.ps1', 'asr-uart-tools.ps1', 'asr-uart-transfer.ps1',
            'asr-tcp-transport.ps1', 'asr-denoise-tools.ps1')) {
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot $file) -Destination $hostDirectory
    }
    Copy-Item -LiteralPath (Join-Path $repoRoot 'scripts/py/denoise_asr_wav.py') `
        -Destination (Join-Path $hostDirectory 'denoise_asr_wav.py')
    Copy-Item -LiteralPath (Join-Path $repoRoot 'docs/sections/conformer_uart_demo.md') `
        -Destination (Join-Path $PackageDirectory 'UART_DEMO.md')
    Copy-Item -LiteralPath (Join-Path $repoRoot 'docs/sections/asr_denoise_experiment.md') `
        -Destination (Join-Path $PackageDirectory 'asr_denoise_experiment.md')
}
$options = @{
    ModelPath=$ModelPath; CompilerBin=$CompilerBin; Action=$Action; Target=$Target
    MemoryMode='Dedicated_Sram'; CacheSizeKiB=256; Jobs=$Jobs
    BuildDirectory=$BuildDirectory; PackageDirectory=$PackageDirectory; Package=$Package
    CMakeArgs=@('-Dasr_INPUT_MODE=UART', '-DSEMIHOSTING_ENABLED=OFF',
        '-DASR_SIMPLE_STAGE_PROFILING=OFF',
        "-DMPS4_DISPLAY_ENABLED=$(if ($Target -eq 'FVP') {'ON'} else {'OFF'})",
        "-DUART0_BAUDRATE=$BaudRate")
}
& (Join-Path $PSScriptRoot 'build-conformer-stage-profile.ps1') @options
