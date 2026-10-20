# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# Run with -CMake pointing to a host CMake executable. No Pester dependency. #>
[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$CMake)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '../windows-tflm-configure.ps1')
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).ProviderPath
$testRoot = Join-Path $repoRoot ('build/tflm-configure-tests/' + [Guid]::NewGuid().ToString('N'))
$project = Join-Path $testRoot 'project'
New-Item -ItemType Directory -Path $project -Force | Out-Null
@'
cmake_minimum_required(VERSION 3.21)
project(WindowsExtractionRecovery NONE)
file(APPEND "${CMAKE_BINARY_DIR}/attempts.txt" "configure\n")
if(MODE STREQUAL "unrelated")
    message(FATAL_ERROR "A model validation error must remain fatal")
endif()
set(name tensorflow-flatbuffers)
set(source "${CMAKE_BINARY_DIR}/_deps/ex-${name}-populate1234/archive")
set(destination "${CMAKE_BINARY_DIR}/_deps/${name}-src")
if(MODE STREQUAL "outside")
    set(source "${CMAKE_BINARY_DIR}/outside/archive")
endif()
if(NOT EXISTS "${destination}/include/probe.h")
    file(MAKE_DIRECTORY "${source}/include")
    file(WRITE "${source}/include/probe.h" "extracted contents")
    set(stamp "${CMAKE_BINARY_DIR}/_deps/${name}-subbuild/${name}-populate-prefix/src/${name}-populate-stamp")
    file(MAKE_DIRECTORY "${stamp}")
    file(WRITE "${stamp}/extract-${name}-populate.cmake" "# extraction completed")
    message(FATAL_ERROR "file RENAME failed to rename\n\n ${source}\n\nto\n\n ${destination}\n\nbecause: Access is denied.")
endif()
if(NOT "${FETCHCONTENT_SOURCE_DIR_TENSORFLOW-FLATBUFFERS}" STREQUAL "${destination}")
    message(FATAL_ERROR "The source override must use this build's extracted files")
endif()
'@ | Set-Content -LiteralPath (Join-Path $project 'CMakeLists.txt') -Encoding utf8

foreach ($mode in @('recover', 'unrelated', 'outside')) {
    $build = Join-Path $testRoot $mode
    $failed = $false
    try {
        Invoke-WindowsTflmConfigure -Executable $CMake -BuildDirectory $build `
            -Arguments @('-S', $project, '-B', $build, "-DMODE=$mode")
    } catch {
        $failed = $true
        if ($mode -eq 'recover') { throw }
    }
    $attempts = @(Get-Content -LiteralPath (Join-Path $build 'attempts.txt')).Count
    if ($mode -eq 'recover') {
        $header = Join-Path $build '_deps/tensorflow-flatbuffers-src/include/probe.h'
        if ($failed -or $attempts -ne 2 -or (Get-Content $header) -ne 'extracted contents') {
            throw 'A verified extraction should be recovered with one retry and preserved contents.'
        }
    } elseif (-not $failed -or $attempts -ne 1) {
        throw "$mode must fail immediately without retrying."
    }
    if ($mode -eq 'outside' -and
        -not (Test-Path -LiteralPath (Join-Path $build 'outside/archive/include/probe.h'))) {
        throw 'An unexpected extraction path must not be moved.'
    }
    Write-Host "PASS: $mode"
}
exit 0
