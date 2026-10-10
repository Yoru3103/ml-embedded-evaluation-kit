# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<#
.SYNOPSIS
Prepare resources or build TFLM firmware on native Windows for Arm DS.
.DESCRIPTION
Setup requires Python 3.12 via the Windows py launcher. Configure and Build
reuse that environment. CMake and Ninja are shared with the ExecuTorch script.
AC6 and ATfE use separate build directories.
#>
[CmdletBinding()]
param(
    [ValidateSet('AC6', 'ATFE')][string]$Toolchain = 'AC6',
    [ValidateSet('FPGA', 'FVP')][string]$Target = 'FPGA',
    [ValidateSet('Setup', 'Configure', 'Build', 'All')][string]$Action = 'Build',
    [string]$UseCase = 'img_class',
    [string]$CompilerBin,
    [string]$ModelPath,
    [string]$InputPath,
    [string]$LabelsPath,
    [ValidateRange(1, 4096)][int]$ImageSize = 0,
    [ValidateRange(1, 1000)][int]$NumClasses = 0,
    [string]$ActivationBufferSize,
    [string]$BuildDirectory,
    [ValidateRange(1, 128)][int]$Jobs = 4,
    [string[]]$CMakeArgs = @()
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).ProviderPath
if ($UseCase -notmatch '^[a-z0-9_]+$') { throw 'Select one use case per invocation.' }
foreach ($argument in $CMakeArgs) {
    if ($argument -notmatch '^-D[A-Za-z0-9_]+(?::[A-Za-z]+)?=') {
        throw 'CMakeArgs accepts only -DNAME=value definitions.'
    }
    if ($argument -match '^-D(CMAKE_(TOOLCHAIN_FILE|C_COMPILER|CXX_COMPILER|MAKE_PROGRAM)|ML_FRAMEWORK|TARGET_PLATFORM|TARGET_SUBSYSTEM|FPGA_PLATFORM_SSE_320|MLEK_PYTHON_VENV|RESOURCES_PATH|USE_CASE_BUILD|ETHOS_U_NPU_MEMORY_MODE)(:|=)') {
        throw 'Use the named script parameters to change the compiler, framework, platform or environment.'
    }
}
$resourceDir = Join-Path $repoRoot 'resources_downloaded/windows'
$venvDir = Join-Path $resourceDir 'env'
$venvPython = Join-Path $resourceDir 'env/Scripts/python.exe'
$toolsVenvDir = Join-Path $repoRoot 'resources_downloaded/windows-tools/env'
$venvHelper = Join-Path $PSScriptRoot 'windows-venv.ps1'
. $venvHelper
$originalPath = $env:PATH
$originalAc6 = $env:MLEK_AC6_BIN
$originalAtfe = $env:MLEK_ATFE_BIN
$originalTemp = $env:TEMP
$originalTmp = $env:TMP
$originalPipCache = $env:PIP_CACHE_DIR
$originalPythonHome = $env:PYTHONHOME
$originalVirtualEnv = $env:VIRTUAL_ENV
$originalPipConfig = $env:PIP_CONFIG_FILE
$originalSslKeylogFile = $env:SSLKEYLOGFILE

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
    $env:PYTHONHOME = $null
    $env:VIRTUAL_ENV = $null
    $env:PIP_CONFIG_FILE = $null
    # Some managed Windows installations set this to a non-writable path.
    # Python's ensurepip inherits it and may fail while creating the venv.
    $env:SSLKEYLOGFILE = $null
    New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
    if ($Action -in @('Setup', 'All')) {
        $buildTools = Ensure-WindowsBuildTools -ToolsVenvDir $toolsVenvDir
        $env:PATH = "$($buildTools.BinDir);$originalPath"
        # This is the independent TFLM/TFLite environment.
        Ensure-WindowsVenv -VenvDir $venvDir | Out-Null
        if ($UseCase -in @('yolov8_detection', 'yolov8_pose')) {
            # YOLOv8 uses caller-supplied model, image and labels rather than
            # an entry in use_case_resources.json.
            Invoke-Checked $venvPython @('-m', 'pip', 'install', '-e', 'scripts/py[tflite]')
        } else {
            Invoke-Checked $venvPython @(
                'set_up_default_resources.py', '--ml-framework', 'tflm',
                '--use-case', $UseCase, '--downloads-dir', $resourceDir,
                '--additional-npu-config-name', 'ethos-u85-1024:Dedicated_Sram',
                '--parallel', "$Jobs"
            )
        }
        # CMake and Ninja live only in the shared Windows tools environment.
        Invoke-Checked $venvPython @('-m', 'pip', 'uninstall', '-y', 'cmake', 'ninja')
    }
    $buildTools = Get-WindowsBuildTools -ToolsVenvDir $toolsVenvDir
    if ($Action -eq 'Setup') { return }
    if (-not (Test-WindowsVenv -Python $venvPython)) {
        throw 'Windows Python environment missing. Run this script with -Action Setup first.'
    }
    $CompilerBin = Resolve-WindowsToolchainBin `
        -Toolchain $Toolchain `
        -CompilerBin $CompilerBin
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
    $env:PATH = "$CompilerBin;$($buildTools.BinDir);$(Split-Path $venvPython);$originalPath"
    $cmakeExe = $buildTools.CMake
    if (-not $BuildDirectory) {
        $buildName = if ($UseCase -eq 'img_class') { $preset } else { "$preset-$UseCase" }
        $BuildDirectory = Join-Path $repoRoot "build/$buildName"
    } elseif (-not [IO.Path]::IsPathRooted($BuildDirectory)) {
        $BuildDirectory = Join-Path $repoRoot $BuildDirectory
    }
    $BuildDirectory = [IO.Path]::GetFullPath($BuildDirectory)
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
    $ninjaPath = $buildTools.Ninja.Replace('\', '/')
    $configureArgs = @(
        '--preset', $preset,
        '-B', $BuildDirectory,
        '-U', 'SRAM_*', '-U', 'EXT_*',
        '-DCPU_PROFILE_ENABLED=ON',
        "-DCMAKE_MAKE_PROGRAM=$ninjaPath",
        "-DUSE_CASE_BUILD=$UseCase"
    )
    if (-not $ModelPath -and $UseCase -eq 'yolov8_pose') {
        $ModelPath = Join-Path $repoRoot `
            'resources_downloaded/yolov8_pose/model_int8_vela_u85_1024_Dedicated_Sram_256k.tflite'
    }
    if ($ModelPath) {
        $modelFile = (Resolve-Path -LiteralPath $ModelPath).ProviderPath.Replace('\', '/')
        $configureArgs += "-D${UseCase}_MODEL_PATH=$modelFile"
    }
    if ($InputPath) {
        $inputFile = (Resolve-Path -LiteralPath $InputPath).ProviderPath.Replace('\', '/')
        $configureArgs += "-D${UseCase}_FILE_PATH=$inputFile"
    }
    if ($LabelsPath) {
        $labelsFile = (Resolve-Path -LiteralPath $LabelsPath).ProviderPath.Replace('\', '/')
        $configureArgs += "-D${UseCase}_LABELS_YAML_FILE=$labelsFile"
    }
    if ($ImageSize -gt 0) {
        $configureArgs += "-D${UseCase}_IMAGE_SIZE=$ImageSize"
    }
    if ($NumClasses -gt 0) {
        $configureArgs += "-D${UseCase}_NUM_CLASSES=$NumClasses"
    }
    if ($ActivationBufferSize) {
        $configureArgs += "-D${UseCase}_ACTIVATION_BUF_SZ=$ActivationBufferSize"
    }
    if ($UseCase -eq 'yolov8_pose') {
        $configureArgs += @('-DETHOS_U_NPU_CACHE_SIZE=262144',
            '-DTA_CONFIG_FILE=ta_config_u85_sys_dram_1024')
    }
    $configureArgs += $CMakeArgs
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
    $env:PYTHONHOME = $originalPythonHome
    $env:VIRTUAL_ENV = $originalVirtualEnv
    $env:PIP_CONFIG_FILE = $originalPipConfig
    $env:SSLKEYLOGFILE = $originalSslKeylogFile
    Pop-Location
}
