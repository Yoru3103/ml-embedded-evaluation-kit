# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

<# Prepare a local Corstone-320 FVP with the SDL event filter for KEY0 and HDLCD. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$FvpExecutable,
    [string]$SdlDll,
    [string]$CompilerBin,
    [string]$HostCompiler,
    [string]$OutputDirectory = 'resources_downloaded/windows/corstone-320-key0'
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).ProviderPath
. (Join-Path $PSScriptRoot 'windows-venv.ps1')

function Invoke-Key0Tool {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$(Split-Path -Leaf $Executable) failed with exit code $LASTEXITCODE."
    }
}

function Get-Key0SdlExports {
    param([string]$Library)
    $peInfo = Invoke-Key0Tool $readObject @('--file-headers', '--coff-exports', $Library)
    if (-not ($peInfo -match '^Format: COFF-x86-64$') -or
        -not ($peInfo -match '^\s+IMAGE_FILE_DLL\b')) {
        throw "Expected a Windows x64 SDL2 DLL at $Library."
    }
    foreach ($line in $peInfo) {
        if ($line -cmatch '^\s+Name: (SDL_[A-Za-z0-9_]+)$') { $Matches[1] }
    }
}

$FvpExecutable = (Resolve-Path -LiteralPath $FvpExecutable).ProviderPath
$modelName = 'FVP_Corstone_SSE-320.exe'
if ((Split-Path -Leaf $FvpExecutable) -ine $modelName) {
    throw "Select the installed Windows Corstone-320 $modelName."
}
$vendorModels = Split-Path -Parent $FvpExecutable
$vendorRoot = (Resolve-Path -LiteralPath (Join-Path $vendorModels '../..')).ProviderPath
$vendorPython = Join-Path $vendorRoot 'python'
$vendorStdlib = Join-Path $vendorPython 'Lib/encodings/__init__.py'
$vendorSdl = Join-Path $vendorModels 'SDL2.dll'
if (-not (Test-Path -LiteralPath (Join-Path $vendorPython 'python39.dll') -PathType Leaf) -or
    -not (Test-Path -LiteralPath $vendorStdlib -PathType Leaf) -or
    -not (Test-Path -LiteralPath $vendorSdl -PathType Leaf)) {
    throw "Select the installed Corstone-320 FVP with its Python 3.9 runtime and SDL2.dll."
}

$CompilerBin = Resolve-WindowsToolchainBin -Toolchain ATFE -CompilerBin $CompilerBin
$linker = Join-Path $CompilerBin 'lld-link.exe'
$readObject = Join-Path $CompilerBin 'llvm-readobj.exe'
foreach ($tool in @($linker, $readObject)) {
    if (-not (Test-Path -LiteralPath $tool -PathType Leaf)) {
        throw "An ATFE Windows tool is missing: $tool. Supply -CompilerBin."
    }
}
if (-not $SdlDll) { $SdlDll = Join-Path $CompilerBin '../../models/bin/SDL2.dll' }
if (-not (Test-Path -LiteralPath $SdlDll -PathType Leaf)) {
    throw 'Supply -SdlDll with the x64 SDL2.dll from Arm Development Studio sw/models/bin.'
}
$SdlDll = (Resolve-Path -LiteralPath $SdlDll).ProviderPath

if (-not [IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory = Join-Path $repoRoot $OutputDirectory
}
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory).TrimEnd('\', '/')
if ($OutputDirectory -ieq $vendorRoot -or
    $OutputDirectory.StartsWith("$vendorRoot\", [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Choose an output directory outside the installed FVP runtime.'
}
$localModels = Join-Path $OutputDirectory 'models/Win64_VC2019'
$localFvp = Join-Path $localModels $modelName
$localSdl = Join-Path $localModels 'SDL2.dll'
$localBackingSdl = Join-Path $localModels 'SDL2_arm.dll'

if (Test-Path -LiteralPath $OutputDirectory) {
    $localPython = Join-Path $OutputDirectory 'python/python39.dll'
    $localStdlib = Join-Path $OutputDirectory 'python/Lib/encodings/__init__.py'
    $requiredFiles = @($localFvp, $localSdl, $localBackingSdl, $localPython, $localStdlib)
    $missingFiles = $requiredFiles | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) }
    if (-not $missingFiles) {
        $fvpHash = (Get-FileHash -LiteralPath $FvpExecutable).Hash
        $sdlHash = (Get-FileHash -LiteralPath $SdlDll).Hash
        $imports = Invoke-Key0Tool $readObject @('--coff-imports', $localSdl)
        if ((Get-FileHash -LiteralPath $localFvp).Hash -eq $fvpHash -and
            (Get-FileHash -LiteralPath $localBackingSdl).Hash -eq $sdlHash -and
            $imports -match '^\s+Name: SDL2_arm\.dll$') {
            Write-Output $localFvp
            return
        }
    }
    throw "Output already exists at $OutputDirectory. Choose a new -OutputDirectory."
}

$replacementExports = @(Get-Key0SdlExports $SdlDll)
$requiredExports = @('SDL_CreateWindow', 'SDL_GetWindowFromID', 'SDL_GetWindowTitle',
    'SDL_SetEventFilter')
$missingExports = @(foreach ($name in ($requiredExports + @(Get-Key0SdlExports $vendorSdl))) {
    if ($name -cnotin $replacementExports) { $name }
})
if ($missingExports) {
    throw "The selected SDL2.dll is missing exports: $($missingExports -join ', ')."
}

if ($HostCompiler) {
    $HostCompiler = (Resolve-Path -LiteralPath $HostCompiler).ProviderPath
} else {
    $msvc = Get-Command -Name 'cl.exe' -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($msvc) {
        $HostCompiler = $msvc.Source
    } else {
        $clangCompilers = Get-Command -Name 'clang.exe' -CommandType Application -All `
            -ErrorAction SilentlyContinue
        foreach ($compiler in $clangCompilers) {
            if ((Invoke-Key0Tool $compiler.Source @('--print-targets')) -match '^\s+x86-64\s+-') {
                $HostCompiler = $compiler.Source
                break
            }
        }
    }
    if (-not $HostCompiler) {
        throw ('Use the Visual Studio x64 Native Tools prompt or supply -HostCompiler ' +
            'with a Windows LLVM clang.exe that targets x86-64.')
    }
}

$parent = Split-Path -Parent $OutputDirectory
New-Item -ItemType Directory -Path $parent -Force | Out-Null
$staging = Join-Path $parent ('.key0-' + [Guid]::NewGuid().ToString('N'))
try {
    $stageModels = Join-Path $staging 'models/Win64_VC2019'
    $scratch = Join-Path $staging 'compile'
    New-Item -ItemType Directory -Path $stageModels, $scratch -Force | Out-Null
    Get-ChildItem -LiteralPath $vendorModels -Force | Copy-Item -Destination $stageModels -Recurse
    Copy-Item -LiteralPath $vendorPython -Destination $staging -Recurse
    $proxy = Join-Path $stageModels 'SDL2.dll'
    Remove-Item -LiteralPath $proxy -Force
    Copy-Item -LiteralPath $SdlDll -Destination (Join-Path $stageModels 'SDL2_arm.dll')

    # Generate the import library for the four SDL calls made by the filter.
    $backingDef = Join-Path $scratch 'sdl-imports.def'
    $backingLib = Join-Path $scratch 'sdl-imports.lib'
    $backingStub = Join-Path $scratch 'sdl-imports.dll'
    @('LIBRARY SDL2_arm.dll', 'EXPORTS') + $requiredExports |
        Set-Content -LiteralPath $backingDef -Encoding Ascii
    Invoke-Key0Tool $linker @('/dll', '/noentry', '/machine:x64', '/nodefaultlib',
        "/def:$backingDef", "/out:$backingStub", "/implib:$backingLib") | Out-Null

    $forwardDef = Join-Path $scratch 'sdl-forwards.def'
    $forwards = foreach ($name in $replacementExports) {
        if ($name -ceq 'SDL_CreateWindow') { "$name=CreateFvpWindow" }
        else { "$name=SDL2_arm.$name" }
    }
    @('LIBRARY SDL2.dll', 'EXPORTS') + $forwards |
        Set-Content -LiteralPath $forwardDef -Encoding Ascii
    $source = Join-Path $PSScriptRoot '../fvp-key0-sdl-filter.c'
    $object = Join-Path $scratch 'sdl-filter.obj'
    $proxyLib = Join-Path $scratch 'sdl-filter.lib'
    $compileArgs = if ((Split-Path -Leaf $HostCompiler) -in @('cl.exe', 'clang-cl.exe')) {
        @('/nologo', '/O2', '/GS-', '/Zl', '/c', "/Fo$object", $source)
    } else {
        @('--target=x86_64-pc-windows-msvc', '-O2', '-ffreestanding',
            '-fno-stack-protector', '-c', $source, '-o', $object)
    }
    Invoke-Key0Tool $HostCompiler $compileArgs | Out-Null
    Invoke-Key0Tool $linker @('/dll', '/noentry', '/machine:x64', '/nodefaultlib',
        "/def:$forwardDef", "/out:$proxy", "/implib:$proxyLib", $object, $backingLib) | Out-Null

    Remove-Item -LiteralPath $scratch -Recurse -Force
    [IO.Directory]::Move($staging, $OutputDirectory)
    Write-Output $localFvp
} finally {
    if (Test-Path -LiteralPath $staging) { Remove-Item -LiteralPath $staging -Recurse -Force }
}
