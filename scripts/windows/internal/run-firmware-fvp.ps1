# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<#
.SYNOPSIS
Run a completed Corstone-320 FVP firmware build and capture its UART output.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$BuildDirectory,
    [string]$FvpExecutable,
    [switch]$Fast,
    [switch]$Headless
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).ProviderPath
. (Join-Path $PSScriptRoot 'fvp-tools.ps1')
if (-not [IO.Path]::IsPathRooted($BuildDirectory)) {
    $BuildDirectory = Join-Path $repoRoot $BuildDirectory
}
$BuildDirectory = (Resolve-Path -LiteralPath $BuildDirectory).ProviderPath
$info = Get-Content -LiteralPath (Join-Path $BuildDirectory 'build-info.json') -Raw |
    ConvertFrom-Json
$config = $info.configuration
$enabled = @('ON', 'TRUE', '1')
if ($info.status -ne 'built' -or $config.TARGET_PLATFORM -ne 'mps4' -or
    $config.TARGET_SUBSYSTEM -ne 'sse-320' -or $config.FPGA_PLATFORM_SSE_320 -in $enabled) {
    throw 'Select a completed Corstone-320 FVP build made with build.ps1 -Target FVP.'
}
if ($Fast -and $config.ETHOS_U_NPU_TIMING_ADAPTER_ENABLED -in $enabled) {
    throw 'FVP -Fast requires firmware built with the timing adapter disabled.'
}
$fvp = Resolve-WindowsFvpExecutable -Executable $FvpExecutable
$firmware = (Resolve-Path -LiteralPath (
    Join-Path $BuildDirectory "bin/mlek_$($info.use_case).axf")).ProviderPath
$semihosting = [int]($config.SEMIHOSTING_ENABLED -in $enabled)
$arguments = @('-a', $firmware, '-C', "mps4_board.subsystem.cpu0.semihosting-enable=$semihosting",
    '-C', "mps4_board.visualisation.disable-visualisation=$([int]$Headless.IsPresent)",
    '-C', "vis_hdlcd.disable_visualisation=$([int]$Headless.IsPresent)",
    '-C', 'mps4_board.uart0.out_file=-', '-C', 'mps4_board.uart0.unbuffered_output=1',
    '-C', 'mps4_board.uart0.shutdown_on_eot=1')
if ($config.ETHOS_U_NPU_CONFIG_ID -match '^Z(\d+)$') {
    $arguments += @('-C', "mps4_board.subsystem.ethosu.num_macs=$($Matches[1])")
}
if ($Fast) { $arguments += @('-C', 'mps4_board.subsystem.ethosu.extra_args=--fast') }
foreach ($uart in @(0, 1, 2, 5)) {
    $telnet = [int]($uart -eq 0 -and $config.INTERACTIVE_MODE -in $enabled)
    $arguments += @('-C', "mps4_board.telnetterminal$uart.start_telnet=$telnet")
}

$log = Join-Path $BuildDirectory 'fvp.log'
Push-Location $BuildDirectory
try {
    & $fvp @arguments | Tee-Object -FilePath $log
    $fvpExit = $LASTEXITCODE
    $failed = Select-String -LiteralPath $log -Pattern 'terminated with failure|Failed to initialise platform' -Quiet
    if ($fvpExit -ne 0 -or $failed) { throw "FVP run failed. See $log" }
} finally {
    Pop-Location
}
