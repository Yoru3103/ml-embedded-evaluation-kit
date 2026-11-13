# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [ValidateSet('FPGA', 'FVP')][string]$Target = 'FVP',
    [string]$CompilerBin = 'C:/Program Files/Arm/Development Studio 2026.0-1/sw/atfe-pro22.1.0/bin',
    [string]$BuildDirectory,
    [string]$ModelPath,
    [string]$CacheOffsets,
    [ValidateSet(64, 128)][int]$ContextLength = 64,
    [switch]$TimingAdapter,
    [switch]$ProfileStages,
    [switch]$Int16Kv,
    [switch]$Demo,
    [switch]$FastDemo,
    [ValidateRange(1, 64)][int]$Jobs = 8
)
$ErrorActionPreference = 'Stop'
if ($FastDemo -and (-not $Demo -or $Target -ne 'FVP')) {
    throw '-FastDemo requires -Target FVP -Demo'
}
if ($ContextLength -eq 128 -and (-not $ModelPath -or -not $CacheOffsets -or -not $Int16Kv)) {
    throw 'The 128-token build requires -ModelPath, -CacheOffsets, and -Int16Kv from the 128-token PTE.'
}
$buildType = if ($FastDemo) { 'Release' } else { 'Debug' }
$activationBufferSize = if ($ContextLength -eq 128) { '0x001B0000' } else { '0x00180000' }
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../../../..')).ProviderPath
$cmake = Join-Path $repoRoot 'resources_downloaded/windows-tools/env/Scripts/cmake.exe'
$ninja = Join-Path $repoRoot 'resources_downloaded/windows-tools/env/Scripts/ninja.exe'
$pythonEnv = Join-Path $repoRoot 'resources_downloaded/windows-executorch/env'
if (-not $BuildDirectory) {
    $BuildDirectory = "build/minimind2_small/test/$($Target.ToLowerInvariant())"
}
$fpgaEnabled = if ($Target -eq 'FPGA') { 'ON' } else { 'OFF' }
$timingAdapterMode = if ($TimingAdapter) { 'ON' } else { 'OFF' }
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
    "-DCMAKE_BUILD_TYPE=$buildType", '-DTARGET_PLATFORM=mps4', '-DTARGET_SUBSYSTEM=sse-320',
    "-DFPGA_PLATFORM_SSE_320=$fpgaEnabled", '-DML_FRAMEWORK=executorch',
    "-DMLEK_PYTHON_VENV=$pythonEnv", "-DRESOURCES_PATH=$repoRoot/resources_downloaded/windows-executorch",
    '-DETHOS_U_NPU_ENABLED=ON', '-DETHOS_U_NPU_ID=U85', '-DETHOS_U_NPU_CONFIG_ID=Z1024',
    '-DETHOS_U_NPU_MEMORY_MODE=Dedicated_Sram', '-DETHOS_U_NPU_CACHE_SIZE=524288',
    "-DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=$timingAdapterMode", '-DUSE_CASE_BUILD=minimind2_small',
    "-Dminimind2_small_ACTIVATION_BUF_SZ=$activationBufferSize",
    "-Dminimind2_small_CONTEXT_LENGTH=$ContextLength",
    '-DUSE_SINGLE_INPUT=ON', "-DUART0_BASE=$uartBase", '-DUART0_BAUDRATE=115200',
    '-DML_FWK_TMP_MEM_BASE=0x73E00000', '-DML_FWK_TMP_MEM_SIZE=0x00200000',
    '-DMPS4_DISPLAY_ENABLED=OFF', '-DSEMIHOSTING_ENABLED=OFF', '-DINTERACTIVE_MODE=OFF',
    '-DCPU_PROFILE_ENABLED=ON'
)
if ($Demo) {
    $options += '-DMLEK_LOG_LEVEL=MLEK_LOG_LEVEL_WARN'
}
if ($ModelPath) {
    $options += "-Dminimind2_small_MODEL_PATH=$( (Resolve-Path -LiteralPath $ModelPath).ProviderPath )"
}
if ($CacheOffsets) {
    $options += "-Dminimind2_small_CACHE_OFFSETS=$CacheOffsets"
}
$profileMode = if ($ProfileStages) { 'ON' } else { 'OFF' }
$options += "-Dminimind2_small_PROFILE_STAGES=$profileMode"
$kvMode = if ($Int16Kv) { 'ON' } else { 'OFF' }
$options += "-Dminimind2_small_INT16_KV=$kvMode"
$demoMode = if ($Demo) { 'ON' } else { 'OFF' }
$options += "-Dminimind2_small_DEMO=$demoMode"
$previousPath = $env:PATH
try {
    $env:PATH = "$CompilerBin;$previousPath"
    & $cmake -S $repoRoot -B $BuildDirectory @options
    if ($LASTEXITCODE -ne 0) { throw 'MiniMind2 configuration failed' }
    & $cmake --build $BuildDirectory --parallel $Jobs
    if ($LASTEXITCODE -ne 0) { throw 'MiniMind2 build failed' }
    Write-Output "Target: $Target; TA: $timingAdapterMode; context: $ContextLength tokens"
    Write-Output "Firmware: $BuildDirectory/bin/mlek_minimind2_small.axf"
} finally {
    $env:PATH = $previousPath
}
