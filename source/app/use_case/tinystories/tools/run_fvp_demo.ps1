# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [string]$BuildDirectory = 'build/tinystories42m/demo/fvp',
    [string]$FvpExecutable = 'C:/Program Files/Arm/FVP_Corstone_SSE-320/models/Win64_VC2019/FVP_Corstone_SSE-320.exe',
    [ValidateRange(1, 65532)][int]$TcpPort = 5000
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../../../..')).ProviderPath
if (-not [IO.Path]::IsPathRooted($BuildDirectory)) {
    $BuildDirectory = Join-Path $repoRoot $BuildDirectory
}
$cache = Get-Content -LiteralPath (Join-Path $BuildDirectory 'CMakeCache.txt')
foreach ($setting in @('tinystories_MODE=DEMO', 'FPGA_PLATFORM_SSE_320=OFF',
        'SEMIHOSTING_ENABLED=OFF', 'ETHOS_U_NPU_TIMING_ADAPTER_ENABLED=OFF',
        'ML_FWK_TMP_MEM_BASE=0x73E00000', 'UART0_BASE=0x58203000')) {
    $key, $value = $setting.Split('=')
    if (-not ($cache -match "^${key}:[^=]+=$value`$")) {
        throw "Expected FVP demo setting: $setting"
    }
}
$firmware = (Resolve-Path (Join-Path $BuildDirectory 'bin/mlek_tinystories.axf')).ProviderPath
$FvpExecutable = (Resolve-Path -LiteralPath $FvpExecutable).ProviderPath
$arguments = @('-a', $firmware,
    '-C', 'mps4_board.subsystem.ethosu.num_macs=1024',
    '-C', 'mps4_board.subsystem.ethosu.extra_args=--fast',
    '-C', 'mps4_board.telnetterminal0.mode=raw',
    '-C', "mps4_board.telnetterminal0.start_port=$TcpPort",
    '-C', "mps4_board.telnetterminal1.start_port=$($TcpPort + 1)",
    '-C', "mps4_board.telnetterminal2.start_port=$($TcpPort + 2)",
    '-C', "mps4_board.telnetterminal5.start_port=$($TcpPort + 3)",
    '-C', 'mps4_board.telnetterminal0.start_telnet=0',
    '-C', 'mps4_board.telnetterminal1.start_telnet=0',
    '-C', 'mps4_board.telnetterminal2.start_telnet=0',
    '-C', 'mps4_board.telnetterminal5.start_telnet=0',
    '-C', 'mps4_board.uart0.shutdown_on_eot=0',
    '-C', 'mps4_board.uart0.out_file=-',
    '-C', 'mps4_board.visualisation.disable-visualisation=1',
    '-C', 'vis_hdlcd.disable_visualisation=1')
Write-Host 'Functional FVP --fast: timings are not FPGA performance measurements.'
Write-Host "Connect a Raw TCP terminal to 127.0.0.1:$TcpPort, UTF-8, local echo OFF."
Write-Host 'Use the actual telnetterminal0 port printed by FVP. Wait for READY before each prompt.'
Write-Host 'This window is output-only. After READY, open another PowerShell and run:'
Write-Host "  & '$PSScriptRoot/connect_fvp_demo.ps1' -TcpPort $TcpPort"
& $FvpExecutable @arguments
if ($LASTEXITCODE -ne 0) { throw "FVP exited with code $LASTEXITCODE" }
