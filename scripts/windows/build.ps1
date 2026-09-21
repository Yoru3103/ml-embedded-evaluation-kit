# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<#
.SYNOPSIS
Prepare resources or build TFLM firmware on native Windows for Arm DS.
.DESCRIPTION
Setup requires Python 3.12 via the Windows py launcher. Configure and Build
reuse that environment. AC6 and ATfE use separate build directories.
#>
[CmdletBinding()]
param(
    [ValidateSet('AC6', 'ATFE')][string]$Toolchain = 'AC6',
    [ValidateSet('FPGA', 'FVP')][string]$Target = 'FPGA',
    [ValidateSet('Setup', 'Configure', 'Build', 'All')][string]$Action = 'Build',
    [string]$UseCase = 'img_class',
    [string]$CompilerBin,
    [string]$ModelPath,
    [string]$BuildDirectory,
    [ValidateRange(1, 128)][int]$Jobs = 4
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).ProviderPath
$resourceDir = Join-Path $repoRoot 'resources_downloaded/windows'
$venvPython = Join-Path $resourceDir 'env/Scripts/python.exe'
$originalPath = $env:PATH
$originalAc6 = $env:MLEK_AC6_BIN
$originalAtfe = $env:MLEK_ATFE_BIN
$originalTemp = $env:TEMP
$originalTmp = $env:TMP
$originalPipCache = $env:PIP_CACHE_DIR

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable failed with exit code $LASTEXITCODE."
    }
}

Push-Location $repoRoot
try {
    # Keep Python/pip temporary files beside the checkout, including on D:.
    $env:TEMP = Join-Path $resourceDir 'tmp'
    $env:TMP = $env:TEMP
    $env:PIP_CACHE_DIR = Join-Path $resourceDir 'pip-cache'
    New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
    if ($Action -in @('Setup', 'All')) {
        # A separate Windows environment never overwrites a WSL virtualenv.
        Invoke-Checked 'py' @('-3.12', '-m', 'venv', (Join-Path $resourceDir 'env'))
        Invoke-Checked $venvPython @('-m', 'pip', 'install', 'cmake==3.30.5', 'ninja==1.11.1.3')
        Invoke-Checked $venvPython @(
            'set_up_default_resources.py', '--ml-framework', 'tflm',
            '--use-case', $UseCase, '--downloads-dir', $resourceDir,
            '--additional-npu-config-name', 'ethos-u85-1024:Dedicated_Sram',
            '--parallel', "$Jobs"
        )
    }
    if ($Action -eq 'Setup') { return }
    if (-not (Test-Path -LiteralPath $venvPython)) {
        throw 'Windows Python environment missing. Run this script with -Action Setup first.'
    }
    if (-not $CompilerBin) {
        if ($Toolchain -eq 'AC6') {
            $CompilerBin = $env:MLEK_AC6_BIN
            if (-not $CompilerBin) {
                $CompilerBin = 'D:/Program Files/ArmCompilerforEmbedded6.24/bin'
            }
        } else {
            $CompilerBin = $env:MLEK_ATFE_BIN
            if (-not $CompilerBin) {
                $CompilerBin = 'D:/Program Files/Arm/Development Studio 2026.0-1/sw/atfe-pro22.1.0/bin'
            }
        }
    }
    $CompilerBin = (Resolve-Path -LiteralPath $CompilerBin).ProviderPath
    if ($Toolchain -eq 'AC6') {
        $requiredTools = @('armclang.exe', 'armasm.exe', 'armar.exe', 'armlink.exe', 'fromelf.exe')
        $env:MLEK_AC6_BIN = $CompilerBin
        $preset = if ($Target -eq 'FVP') { 'windows-ac6-fvp' } else { 'windows-ac6-mps4' }
    } else {
        $requiredTools = @('clang.exe', 'llvm-ar.exe', 'llvm-objcopy.exe')
        $env:MLEK_ATFE_BIN = $CompilerBin
        $preset = if ($Target -eq 'FVP') { 'windows-atfe-fvp' } else { 'windows-atfe-mps4' }
    }
    foreach ($tool in $requiredTools) {
        if (-not (Test-Path -LiteralPath (Join-Path $CompilerBin $tool))) {
            throw "Selected toolchain is missing $tool in $CompilerBin"
        }
    }
    $env:PATH = "$CompilerBin;$(Split-Path $venvPython);$originalPath"
    $cmakeExe = Join-Path (Split-Path $venvPython) 'cmake.exe'
    if (-not $BuildDirectory) {
        $BuildDirectory = Join-Path $repoRoot "build/$preset"
    }
    $cacheFile = Join-Path $BuildDirectory 'CMakeCache.txt'
    if (Test-Path -LiteralPath $cacheFile) {
        $toolchainName = if ($Toolchain -eq 'AC6') { 'armclang' } else { 'llvm' }
        $expectedToolchain = (Join-Path $repoRoot `
            "scripts/cmake/toolchains/bare-metal-$toolchainName.cmake").Replace('\', '/')
        $toolchainEntry = Select-String -LiteralPath $cacheFile `
            -Pattern '^CMAKE_TOOLCHAIN_FILE:FILEPATH=(.+)$'
        if ($toolchainEntry -and
            $toolchainEntry.Matches[0].Groups[1].Value.Replace('\', '/') -ine $expectedToolchain) {
            throw 'Build directory belongs to another compiler. Use a new -BuildDirectory.'
        }
        # These toolchains set the compiler as a normal CMake variable, so it
        # need not appear in CMakeCache.txt. Inspect CMake's compiler record.
        $compilerFiles = Get-ChildItem -Path `
            (Join-Path $BuildDirectory 'CMakeFiles/*/CMakeCCompiler.cmake') -ErrorAction SilentlyContinue
        foreach ($compilerFile in $compilerFiles) {
            $compilerEntry = Select-String -LiteralPath $compilerFile.FullName `
                -Pattern '^set\(CMAKE_C_COMPILER "([^"]+)"\)'
            if (-not $compilerEntry) { continue }
            $cachedCompiler = $compilerEntry.Matches[0].Groups[1].Value.Replace('\', '/')
            $selectedCompiler = (Join-Path $CompilerBin $requiredTools[0]).Replace('\', '/')
            if ($cachedCompiler -ine $selectedCompiler) {
                throw 'Build directory belongs to another compiler. Use a new -BuildDirectory.'
            }
        }
    }
    $configureArgs = @('--preset', $preset, '-B', $BuildDirectory, "-DUSE_CASE_BUILD=$UseCase")
    if ($ModelPath) {
        $modelFile = (Resolve-Path -LiteralPath $ModelPath).ProviderPath.Replace('\', '/')
        $configureArgs += "-D${UseCase}_MODEL_PATH=$modelFile"
    }
    # Always reconfigure: model C arrays are generated during configuration.
    Invoke-Checked $cmakeExe $configureArgs
    if ($Action -ne 'Configure') {
        Invoke-Checked $cmakeExe @('--build', $BuildDirectory, '--parallel', "$Jobs")
    }
} finally {
    $env:PATH = $originalPath
    $env:MLEK_AC6_BIN = $originalAc6
    $env:MLEK_ATFE_BIN = $originalAtfe
    $env:TEMP = $originalTemp
    $env:TMP = $originalTmp
    $env:PIP_CACHE_DIR = $originalPipCache
    Pop-Location
}
