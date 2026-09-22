# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<#
.SYNOPSIS
Prepare native Windows ExecuTorch tools and cross-compile a supplied PTE model.
.DESCRIPTION
Run Setup once, then Configure or Build. Model export is a separate step.
Use GNU Arm Embedded (arm-none-eabi), or ATFE. AC6 is unsupported by ExecuTorch.
All generated files, caches and temporary files stay on D:.
#>
[CmdletBinding()]
param(
    [ValidateSet('Setup', 'Configure', 'Build', 'All')][string]$Action = 'Build',
    [ValidateSet('GNU', 'ATFE')][string]$Toolchain = 'GNU',
    [ValidateSet('FVP', 'FPGA')][string]$Target = 'FVP',
    [string]$UseCase = 'img_class',
    [string]$CompilerBin,
    [string]$ModelPath,
    [string]$InputPath,
    [string]$LabelsPath,
    [string]$BuildDirectory,
    [ValidateSet('Dedicated_Sram', 'Shared_Sram')][string]$MemoryMode = 'Dedicated_Sram',
    [ValidateRange(1, 128)][int]$Jobs = 4,
    [string[]]$CMakeArgs = @()
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).ProviderPath
if ($repoRoot -notmatch '^D:[\\/]') {
    throw 'Use a native Windows checkout on D:, for example D:\work\mlek-client.'
}
if ($UseCase -eq 'yolov8_detection') {
    throw 'YOLOv8 is reserved for reimplementation; its old Alif APIs were removed. Use img_class with a matching PTE.'
}
if ($UseCase -notmatch '^[a-z0-9_]+$') { throw 'Select one use case per invocation.' }
foreach ($argument in $CMakeArgs) {
    if ($argument -notmatch '^-D[A-Za-z0-9_]+(?::[A-Za-z]+)?=') {
        throw 'CMakeArgs accepts only -DNAME=value definitions.'
    }
    if ($argument -match '^-D(CMAKE_(TOOLCHAIN_FILE|C_COMPILER|CXX_COMPILER|MAKE_PROGRAM)|ML_FRAMEWORK|TARGET_PLATFORM|TARGET_SUBSYSTEM|FPGA_PLATFORM_SSE_320|MLEK_PYTHON_VENV|RESOURCES_PATH|USE_CASE_BUILD|ETHOS_U_NPU_MEMORY_MODE)(:|=)') {
        throw 'Use the named script parameters to change the compiler, framework, platform or environment.'
    }
}
$resourceDir = Join-Path $repoRoot 'resources_downloaded/windows-executorch'
$venvDir = Join-Path $resourceDir 'env'
$binDir = Join-Path $venvDir 'Scripts'
$python = Join-Path $binDir 'python.exe'
$cmake = Join-Path $binDir 'cmake.exe'
if (-not $BuildDirectory) {
    $BuildDirectory = Join-Path $repoRoot "build/windows-$Toolchain-et-$Target-$UseCase-$MemoryMode"
}
$BuildDirectory = [IO.Path]::GetFullPath($BuildDirectory)
if ($BuildDirectory -notmatch '^D:[\\/]') { throw 'BuildDirectory must be on D:.' }
$savedEnvironment = @{}
foreach ($name in @('PATH', 'TEMP', 'TMP', 'PIP_CACHE_DIR', 'PYTHONPATH')) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Executable failed with exit code $LASTEXITCODE." }
}
Push-Location $repoRoot
try {
    $env:TEMP = Join-Path $resourceDir 'tmp'
    $env:TMP = $env:TEMP
    $env:PIP_CACHE_DIR = Join-Path $resourceDir 'pip-cache'
    New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
    $env:PATH = "$binDir;$env:PATH"
    $env:PYTHONPATH = Join-Path $repoRoot 'scripts/py'
    $etSource = Join-Path $repoRoot 'dependencies/executorch'
    $versionFile = Join-Path $etSource 'version.txt'
    if (-not (Test-Path -LiteralPath $versionFile)) {
        throw 'Initialize dependencies first: git submodule update --init --recursive'
    }
    $version = (Get-Content -LiteralPath $versionFile -Raw).Trim()
    if ($version -ne '1.3.1') {
        throw "This Windows setup is pinned to ExecuTorch 1.3.1; source reports $version."
    }
    if ($Action -in @('Setup', 'All')) {
        if (-not (Test-Path -LiteralPath $python)) {
            Invoke-Checked 'py' @('-3.12', '-m', 'venv', $venvDir)
        }
        Invoke-Checked $python @('-m', 'pip', 'install',
            'cmake==3.30.5', 'ninja==1.11.1.3', 'executorch==1.3.1',
            'torch==2.12.0', 'torchao==0.17.0', '-e', (Join-Path $repoRoot 'scripts/py'))
        Invoke-Checked $python @('-m', 'pip', 'check')
    }
    if (-not (Test-Path -LiteralPath $python)) { throw 'Run -Action Setup first.' }
    Invoke-Checked $python @('-c',
        'import importlib.metadata as m; assert m.version("executorch") == "1.3.1"; assert m.version("torch").split("+")[0] == "2.12.0"; from executorch.exir._serialize import _deserialize_pte_binary; import torchgen')
    Invoke-Checked (Join-Path $binDir 'flatc.exe') @('--version')
    if ($Action -eq 'Setup') { return }
    foreach ($dependency in @('flatbuffers', 'flatcc', 'gflags', 'json')) {
        if (-not (Test-Path (Join-Path $etSource "third-party/$dependency/CMakeLists.txt"))) {
            throw "Missing ExecuTorch submodule $dependency. Run git submodule update --init --recursive."
        }
    }
    # ExecuTorch 1.3.1 uses the target's WIN32 when creating host flatc.
    # Carry a small reversible patch to reuse the Windows wheel executable.
    $hostPatch = Join-Path $PSScriptRoot 'patches/executorch-host-flatc.patch'
    $thirdPartyCmake = Join-Path $etSource 'third-party/CMakeLists.txt'
    if (Select-String -LiteralPath $thirdPartyCmake -Pattern 'MLEK_HOST_FLATC' -Quiet) {
        Invoke-Checked 'git' @('-C', $etSource, 'apply', '--reverse', '--check', $hostPatch)
    } else {
        Invoke-Checked 'git' @('-C', $etSource, 'apply', '--check', $hostPatch)
        Invoke-Checked 'git' @('-C', $etSource, 'apply', $hostPatch)
    }
    if (-not $ModelPath) { throw 'Supply -ModelPath with an exported, target-compatible .pte file.' }
    $ModelPath = (Resolve-Path -LiteralPath $ModelPath).ProviderPath.Replace('\', '/')
    if ([IO.Path]::GetExtension($ModelPath) -ine '.pte') {
        throw 'ModelPath must be .pte. A .pt checkpoint must be exported before compilation.'
    }
    if (-not $CompilerBin) {
        if ($Toolchain -eq 'GNU') {
            $CompilerBin = $env:MLEK_GNU_BIN
        } else {
            $CompilerBin = 'D:/Program Files/Arm/Development Studio 2026.0-1/sw/atfe-pro22.1.0/bin'
        }
    }
    if (-not $CompilerBin) { throw 'Set -CompilerBin or MLEK_GNU_BIN to the arm-none-eabi toolchain bin directory.' }
    $CompilerBin = (Resolve-Path -LiteralPath $CompilerBin).ProviderPath
    $compiler = if ($Toolchain -eq 'GNU') { 'arm-none-eabi-gcc.exe' } else { 'clang.exe' }
    $requiredTools = if ($Toolchain -eq 'GNU') {
        @($compiler, 'arm-none-eabi-g++.exe', 'arm-none-eabi-objcopy.exe')
    } else { @($compiler, 'llvm-ar.exe', 'llvm-objcopy.exe') }
    foreach ($tool in $requiredTools) {
        if (-not (Test-Path -LiteralPath (Join-Path $CompilerBin $tool))) {
            throw "Missing $tool in $CompilerBin"
        }
    }
    $env:PATH = "$CompilerBin;$env:PATH"
    $toolchainFile = if ($Toolchain -eq 'GNU') { 'bare-metal-gcc.cmake' } else { 'bare-metal-llvm.cmake' }
    $toolchainFile = (Join-Path $repoRoot "scripts/cmake/toolchains/$toolchainFile").Replace('\', '/')
    $cacheFile = Join-Path $BuildDirectory 'CMakeCache.txt'
    if (Test-Path -LiteralPath $cacheFile) {
        foreach ($pair in @(
            @('CMAKE_TOOLCHAIN_FILE', $toolchainFile),
            @('ML_FRAMEWORK', 'executorch'),
            @('USE_CASE_BUILD', $UseCase),
            @('ETHOS_U_NPU_MEMORY_MODE', $MemoryMode),
            @('FPGA_PLATFORM_SSE_320', $(if ($Target -eq 'FPGA') { 'ON' } else { 'OFF' }))
        )) {
            $entry = Select-String -LiteralPath $cacheFile -Pattern ("^" + $pair[0] + ":[^=]+=(.*)$")
            if ($entry -and $entry.Matches[0].Groups[1].Value.Replace('\', '/') -ine $pair[1]) {
                throw "Build directory has different $($pair[0]). Use a new -BuildDirectory."
            }
        }
        foreach ($file in Get-ChildItem (Join-Path $BuildDirectory 'CMakeFiles/*/CMakeCCompiler.cmake')) {
            $entry = Select-String -LiteralPath $file.FullName -Pattern '^set\(CMAKE_C_COMPILER "([^"]+)"\)'
            if ($entry -and $entry.Matches[0].Groups[1].Value.Replace('\', '/') -ine
                (Join-Path $CompilerBin $compiler).Replace('\', '/')) {
                throw 'Build directory belongs to another compiler installation.'
            }
        }
    }
    $fpga = if ($Target -eq 'FPGA') { 'ON' } else { 'OFF' }
    $argsList = @('-S', $repoRoot, '-B', $BuildDirectory, '-G', 'Ninja',
        "-DCMAKE_TOOLCHAIN_FILE=$toolchainFile", '-DCMAKE_BUILD_TYPE=Debug',
        '-DTARGET_PLATFORM=mps4', '-DTARGET_SUBSYSTEM=sse-320',
        "-DFPGA_PLATFORM_SSE_320=$fpga", '-DML_FRAMEWORK=executorch',
        "-DMLEK_PYTHON_VENV=$($venvDir.Replace('\', '/'))",
        "-DRESOURCES_PATH=$($resourceDir.Replace('\', '/'))",
        "-DETHOS_U_NPU_MEMORY_MODE=$MemoryMode", "-DUSE_CASE_BUILD=$UseCase",
        "-D${UseCase}_MODEL_PATH=$ModelPath", '-DUSE_SINGLE_INPUT=ON')
    if ($InputPath) {
        $argsList += "-D${UseCase}_FILE_PATH=$((Resolve-Path -LiteralPath $InputPath).ProviderPath.Replace('\', '/'))"
    }
    if ($LabelsPath) {
        $argsList += "-D${UseCase}_LABELS_TXT_FILE=$((Resolve-Path -LiteralPath $LabelsPath).ProviderPath.Replace('\', '/'))"
    }
    $argsList += $CMakeArgs
    Invoke-Checked $cmake $argsList
    if ($Action -ne 'Configure') {
        Invoke-Checked $cmake @('--build', $BuildDirectory, '--parallel', "$Jobs")
    }
} finally {
    foreach ($name in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
    }
    Pop-Location
}
