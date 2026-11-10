# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [ValidateSet('FPGA', 'FVP')][string]$Target = 'FPGA',
    [string]$CompilerBin = 'C:/Program Files/Arm/Development Studio 2026.0-1/sw/atfe-pro22.1.0/bin',
    [string]$BuildDirectory,
    [ValidateRange(1, 64)][int]$Jobs = 8
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../../../..')).ProviderPath
$cmake = Join-Path $repoRoot 'resources_downloaded/windows-tools/env/Scripts/cmake.exe'
$ninja = Join-Path $repoRoot 'resources_downloaded/windows-tools/env/Scripts/ninja.exe'
$pythonEnv = Join-Path $repoRoot 'resources_downloaded/windows-executorch/env'
if (-not $BuildDirectory) {
    $BuildDirectory = "build/tinystories42m/demo/$($Target.ToLowerInvariant())"
}
$fpgaEnabled = if ($Target -eq 'FPGA') { 'ON' } else { 'OFF' }
$uartBase = if ($Target -eq 'FPGA') { '0x59303000' } else { '0x58203000' }
if (-not (Test-Path (Join-Path $CompilerBin 'clang.exe'))) {
    throw "Arm compiler not found: $CompilerBin"
}
if (-not [IO.Path]::IsPathRooted($BuildDirectory)) {
    $BuildDirectory = Join-Path $repoRoot $BuildDirectory
}
$options = @(
    '-G', 'Ninja', "-DCMAKE_MAKE_PROGRAM=$ninja",
    "-DCMAKE_TOOLCHAIN_FILE=$repoRoot/scripts/cmake/toolchains/bare-metal-llvm.cmake",
    '-DCMAKE_BUILD_TYPE=Debug', '-DTARGET_PLATFORM=mps4', '-DTARGET_SUBSYSTEM=sse-320',
    "-DFPGA_PLATFORM_SSE_320=$fpgaEnabled", '-DML_FRAMEWORK=executorch',
    "-DMLEK_PYTHON_VENV=$pythonEnv", "-DRESOURCES_PATH=$repoRoot/resources_downloaded/windows-executorch",
    '-DETHOS_U_NPU_ENABLED=ON', '-DETHOS_U_NPU_ID=U85', '-DETHOS_U_NPU_CONFIG_ID=Z1024',
    '-DETHOS_U_NPU_MEMORY_MODE=Dedicated_Sram', '-DETHOS_U_NPU_CACHE_SIZE=1258304',
    '-DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=OFF', '-DUSE_CASE_BUILD=tinystories',
    '-Dtinystories_MODE=DEMO', '-Dtinystories_ACTIVATION_BUF_SZ=0x00800000',
    '-Dtinystories_CACHE_CLAMP_OPTIMIZED=ON', '-Dtinystories_CACHE_CLAMP_SPECIALIZED=ON',
    '-DUSE_SINGLE_INPUT=ON', "-DUART0_BASE=$uartBase", '-DUART0_BAUDRATE=115200',
    '-DML_FWK_TMP_MEM_BASE=0x73E00000', '-DML_FWK_TMP_MEM_SIZE=0x00200000',
    '-DMPS4_DISPLAY_ENABLED=OFF', '-DSEMIHOSTING_ENABLED=OFF', '-DINTERACTIVE_MODE=OFF',
    '-DCPU_PROFILE_ENABLED=ON'
)
$previousPath = $env:PATH
try {
    $env:PATH = "$CompilerBin;$previousPath"
    & $cmake -S $repoRoot -B $BuildDirectory @options
    if ($LASTEXITCODE -ne 0) { throw 'TinyStories demo configuration failed' }
    & $cmake --build $BuildDirectory --parallel $Jobs
    if ($LASTEXITCODE -ne 0) { throw 'TinyStories demo build failed' }
    Write-Output "Firmware: $BuildDirectory/bin/mlek_tinystories.axf"
} finally {
    $env:PATH = $previousPath
}
