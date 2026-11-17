# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# Run a functional fast-mode FVP UART server. Stop it with Ctrl+C when finished. #>
[CmdletBinding()]
param(
    [string]$BuildDirectory = 'build/asr-uart/dedicated-256k-fvp',
    [string]$FvpExecutable = 'C:/Program Files/Arm/FVP_Corstone_SSE-320/models/Win64_VC2019/FVP_Corstone_SSE-320.exe',
    [ValidateRange(1, 65532)][int]$TcpPort = 5000
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).ProviderPath
if (-not [IO.Path]::IsPathRooted($BuildDirectory)) {
    $BuildDirectory = Join-Path $repoRoot $BuildDirectory
}
$cache = Get-Content -LiteralPath (Join-Path $BuildDirectory 'CMakeCache.txt')
foreach ($setting in @('asr_INPUT_MODE=UART', 'FPGA_PLATFORM_SSE_320=OFF',
        'SEMIHOSTING_ENABLED=OFF', 'ETHOS_U_NPU_TIMING_ADAPTER_ENABLED=OFF')) {
    $key, $value = $setting.Split('=')
    if (-not ($cache -match "^${key}:[^=]+=$value`$")) {
        throw "Expected FVP UART fast-mode setting: $setting"
    }
}
$firmware = (Resolve-Path (Join-Path $BuildDirectory 'bin/mlek_asr.axf')).ProviderPath
$FvpExecutable = (Resolve-Path -LiteralPath $FvpExecutable).ProviderPath
$arguments = @('-a', $firmware,
    '-C', 'mps4_board.subsystem.ethosu.num_macs=1024',
    '-C', 'mps4_board.subsystem.ethosu.extra_args=--fast',
    '-C', 'mps4_board.telnetterminal0.mode=raw',
    '-C', "mps4_board.telnetterminal0.start_port=$TcpPort",
    '-C', 'mps4_board.telnetterminal0.start_telnet=0',
    '-C', 'mps4_board.telnetterminal1.start_telnet=0',
    '-C', 'mps4_board.telnetterminal2.start_telnet=0',
    '-C', 'mps4_board.telnetterminal5.start_telnet=0',
    '-C', 'mps4_board.uart0.shutdown_on_eot=0',
    '-C', 'mps4_board.uart0.out_file=-',
    '-C', 'mps4_board.visualisation.disable-visualisation=1',
    '-C', 'vis_hdlcd.disable_visualisation=0')
Write-Host 'Functional fast mode: processing times are not FPGA performance measurements.'
Write-Host 'Send WAVs from another PowerShell with -TcpHost 127.0.0.1 and the UART0 port below.'
Write-Host "Requested starting port: $TcpPort. Use the port printed for telnetterminal0."
& $FvpExecutable @arguments
if ($LASTEXITCODE -ne 0) { throw "FVP exited with code $LASTEXITCODE" }
