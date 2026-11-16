# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [string]$BuildDirectory = 'build/minimind2_small/test/fvp',
    [ValidateRange(1024, 65532)][int]$UartPort = 15650,
    [string]$FvpExecutable = 'C:/Program Files/Arm/FVP_Corstone_SSE-320/models/Win64_VC2019/FVP_Corstone_SSE-320.exe'
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../../../..')).ProviderPath
if (-not [IO.Path]::IsPathRooted($BuildDirectory)) {
    $BuildDirectory = Join-Path $repoRoot $BuildDirectory
}
$cache = Get-Content -LiteralPath (Join-Path $BuildDirectory 'CMakeCache.txt')
foreach ($setting in @('FPGA_PLATFORM_SSE_320=OFF', 'SEMIHOSTING_ENABLED=OFF',
        'ML_FWK_TMP_MEM_BASE=0x73E00000', 'UART0_BASE=0x58203000')) {
    $key, $value = $setting.Split('=')
    if (-not ($cache -match "^${key}:[^=]+=$value`$")) {
        throw "Expected FVP test setting: $setting"
    }
}
$firmware = (Resolve-Path (Join-Path $BuildDirectory 'bin/mlek_minimind2_small.axf')).ProviderPath
$FvpExecutable = (Resolve-Path -LiteralPath $FvpExecutable).ProviderPath
$listeners = [Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
foreach ($port in $UartPort..($UartPort + 3)) {
    if ($listeners.Port -contains $port) {
        throw "TCP port $port is in use. Stop the other FVP or choose a different -UartPort."
    }
}
$arguments = @('-a', $firmware,
    '-C', 'mps4_board.subsystem.ethosu.num_macs=1024',
    '-C', 'mps4_board.subsystem.ethosu.extra_args=--fast',
    '-C', 'mps4_board.telnetterminal0.mode=raw',
    '-C', "mps4_board.telnetterminal0.start_port=$UartPort",
    '-C', "mps4_board.telnetterminal1.start_port=$($UartPort + 1)",
    '-C', "mps4_board.telnetterminal2.start_port=$($UartPort + 2)",
    '-C', "mps4_board.telnetterminal5.start_port=$($UartPort + 3)",
    '-C', 'mps4_board.telnetterminal0.start_telnet=0',
    '-C', 'mps4_board.telnetterminal1.start_telnet=0',
    '-C', 'mps4_board.telnetterminal2.start_telnet=0',
    '-C', 'mps4_board.telnetterminal5.start_telnet=0',
    '-C', 'mps4_board.uart0.shutdown_on_eot=0',
    '-C', 'mps4_board.uart0.out_file=-',
    '-C', 'mps4_board.visualisation.disable-visualisation=1',
    '-C', 'vis_hdlcd.disable_visualisation=1')
Write-Host 'Functional FVP --fast: timings are not FPGA performance measurements.'
Write-Host "Requested UART0 port $UartPort. Use the port printed by telnetterminal0."
& $FvpExecutable @arguments
if ($LASTEXITCODE -ne 0) { throw "FVP exited with code $LASTEXITCODE" }
