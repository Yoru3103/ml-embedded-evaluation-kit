# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [ValidateSet('FPGA', 'FVP')][string]$Target = 'FPGA',
    [ValidateSet('DEMO', 'TEST')][string]$Mode = 'DEMO',
    [string]$CompilerBin = 'C:/Program Files/Arm/Development Studio 2026.0-1/sw/atfe-pro22.1.0/bin',
    [string]$BuildDirectory,
    [string]$ModelPath,
    [ValidateRange(1, 1024)][int]$Context = 128,
    [ValidateRange(1, 64)][int]$Jobs = 8,
    [switch]$TimingAdapter,
    [switch]$LogitDump
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../../../..')).ProviderPath
$cmake = Join-Path $repoRoot 'resources_downloaded/windows-tools/env/Scripts/cmake.exe'
$ninja = Join-Path $repoRoot 'resources_downloaded/windows-tools/env/Scripts/ninja.exe'
$pythonEnv = Join-Path $repoRoot 'resources_downloaded/windows-executorch/env'
if (-not $ModelPath) {
    $ModelPath = Join-Path $repoRoot ('resources_downloaded/tinystories42m/' +
        'npu_int16_kv_ctx128_1200/tinystories42m_u85_1024_ctx128_int16_kv_1200.pte')
}
if (-not (Test-Path -LiteralPath $ModelPath)) {
    throw "TinyStories PTE not found: $ModelPath"
}
if (-not $BuildDirectory) {
    $modeDir = if ($Mode -eq 'DEMO') { 'demo' } else { 'test' }
    $BuildDirectory = "build/tinystories42m/$modeDir/$($Target.ToLowerInvariant())"
}
$fpgaEnabled = if ($Target -eq 'FPGA') { 'ON' } else { 'OFF' }
$uartBase = if ($Target -eq 'FPGA') { '0x59303000' } else { '0x58203000' }
if (-not (Test-Path (Join-Path $CompilerBin 'clang.exe'))) {
    throw "Arm compiler not found: $CompilerBin"
}
if (-not [IO.Path]::IsPathRooted($BuildDirectory)) {
    $BuildDirectory = Join-Path $repoRoot $BuildDirectory
}
$modelPathForward = $ModelPath -replace '\\', '/'
$timingAdapterMode = if ($TimingAdapter) { 'ON' } else { 'OFF' }
$options = @(
    '-G', 'Ninja', "-DCMAKE_MAKE_PROGRAM=$ninja",
    "-DCMAKE_TOOLCHAIN_FILE=$repoRoot/scripts/cmake/toolchains/bare-metal-llvm.cmake",
    '-DCMAKE_BUILD_TYPE=Debug', '-DTARGET_PLATFORM=mps4', '-DTARGET_SUBSYSTEM=sse-320',
    "-DFPGA_PLATFORM_SSE_320=$fpgaEnabled", '-DML_FRAMEWORK=executorch',
    "-DMLEK_PYTHON_VENV=$pythonEnv", "-DRESOURCES_PATH=$repoRoot/resources_downloaded/windows-executorch",
    '-DETHOS_U_NPU_ENABLED=ON', '-DETHOS_U_NPU_ID=U85', '-DETHOS_U_NPU_CONFIG_ID=Z1024',
    '-DETHOS_U_NPU_MEMORY_MODE=Dedicated_Sram', '-DETHOS_U_NPU_CACHE_SIZE=1258304',
    "-DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=$timingAdapterMode", '-DUSE_CASE_BUILD=tinystories',
    "-Dtinystories_MODE=$Mode", "-Dtinystories_CONTEXT=$Context",
    "-Dtinystories_MODEL_PATH=$modelPathForward",
    '-Dtinystories_ACTIVATION_BUF_SZ=0x00700000',
    '-Dtinystories_CACHE_CLAMP_OPTIMIZED=ON', '-Dtinystories_CACHE_CLAMP_SPECIALIZED=ON',
    '-DUSE_SINGLE_INPUT=ON', "-DUART0_BASE=$uartBase", '-DUART0_BAUDRATE=115200',
    '-DML_FWK_TMP_MEM_BASE=0x73DC0000', '-DML_FWK_TMP_MEM_SIZE=0x00240000',
    '-DMPS4_DISPLAY_ENABLED=OFF', '-DSEMIHOSTING_ENABLED=OFF', '-DINTERACTIVE_MODE=OFF',
    '-DCPU_PROFILE_ENABLED=ON'
)
if ($LogitDump) {
    $options += '-Dtinystories_LOGIT_DUMP=ON'
}
$previousPath = $env:PATH
try {
    $env:PATH = "$CompilerBin;$previousPath"
    & $cmake -S $repoRoot -B $BuildDirectory @options
    if ($LASTEXITCODE -ne 0) { throw "TinyStories $Mode configuration failed" }
    & $cmake --build $BuildDirectory --parallel $Jobs
    if ($LASTEXITCODE -ne 0) { throw "TinyStories $Mode build failed" }
    Write-Output "Mode: $Mode; target: $Target; context: $Context; TA: $timingAdapterMode"
    Write-Output "PTE: $ModelPath"
    Write-Output "Firmware: $BuildDirectory/bin/mlek_tinystories.axf"
} finally {
    $env:PATH = $previousPath
}
