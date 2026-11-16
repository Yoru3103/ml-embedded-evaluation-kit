# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<#
.SYNOPSIS
Build and optionally package the fixed-window WeKWS PTE use case.
.EXAMPLE
./scripts/windows/build-kws-zh.ps1 -Package
#>
[CmdletBinding()]
param(
    [ValidateSet('Dedicated_Sram', 'Shared_Sram')]
    [string[]]$MemoryModes = @('Dedicated_Sram', 'Shared_Sram'),
    [ValidateSet('FPGA', 'FVP')][string]$Target = 'FPGA',
    [string]$ModelsRoot = 'resources_downloaded/kws_zh',
    [string]$InputPath = 'resources/kws_zh/samples',
    [string]$BuildRoot = 'build/kws_zh',
    [string]$BuildTag = (Get-Date -Format 'yyyyMMdd'),
    [string]$PackageRoot = 'build/board-test-packages/kws_zh',
    [string]$CompilerBin,
    [ValidateRange(1, 4096)][int]$CacheSizeKiB = 256,
    [ValidateRange(1, 128)][int]$Jobs = 4,
    [switch]$DumpOutput,
    [switch]$Package
)
$ErrorActionPreference = 'Stop'
if ($Package -and $Target -ne 'FPGA') { throw '-Package requires -Target FPGA.' }
if ($BuildTag -notmatch '^[a-zA-Z0-9_-]+$') { throw 'BuildTag accepts letters, digits, _ and -.' }
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).ProviderPath
Push-Location $repoRoot
try {
    foreach ($mode in $MemoryModes) {
        $dedicated = $mode -eq 'Dedicated_Sram'
        $variant = if ($dedicated) { "dedicated-$($CacheSizeKiB)k" } else { 'shared' }
        $directory = Join-Path $BuildRoot "pte-$variant-$($Target.ToLowerInvariant())-$BuildTag"
        $modelDirectory = if ($dedicated) { "pte_$($mode)_$($CacheSizeKiB)k" } else { "pte_$mode" }
        $model = Join-Path $ModelsRoot "$modelDirectory/wekws_dstcn_u85.pte"
        # Dedicated scratch uses AXI EXT; Shared scratch uses AXI SRAM.
        $temporaryBase = if ($dedicated) { '0x72000000' } else { '0x31100000' }
        $options = @{
            Framework = 'pte'; Toolchain = 'ATFE'; Target = $Target; Action = 'Build'
            UseCase = 'kws_zh'; MemoryMode = $mode; BuildType = 'Release'; PmuGroup = 'A'
            ModelPath = $model; InputPath = $InputPath; BuildDirectory = $directory
            CompilerBin = $CompilerBin; Jobs = $Jobs
            CMakeArgs = @(
                '-DUSE_SINGLE_INPUT=OFF',
                "-DML_FWK_TMP_MEM_BASE=$temporaryBase",
                '-DML_FWK_TMP_MEM_SIZE=0x00300000',
                '-DHDLCD_FRAME_BUFFER_BASE_ADDRESS=0x74000000',
                "-Dkws_zh_DUMP_OUTPUT=$(if ($DumpOutput) { 'ON' } else { 'OFF' })")
        }
        if ($Target -eq 'FPGA') {
            $options.TimingAdapterConfig = 'ta_config_u85_sys_dram_1024'
        }
        if ($mode -eq 'Dedicated_Sram') {
            $options.CacheSizeKiB = $CacheSizeKiB
        } else {
            $options.CMakeArgs += '-DETHOS_U_NPU_CACHE_SIZE=0'
        }
        & (Join-Path $PSScriptRoot 'build.ps1') @options
        if ($Package) {
            . (Join-Path $PSScriptRoot 'windows-venv.ps1')
            $compiler = Resolve-WindowsToolchainBin -Toolchain ATFE -CompilerBin $CompilerBin
            $output = Join-Path $PackageRoot "pte-$variant-$BuildTag"
            New-Item -ItemType Directory -Force -Path (Join-Path $output 'validation') |
                Out-Null
            Copy-Item -LiteralPath 'docs/use_cases/kws_zh.md' `
                -Destination (Join-Path $output 'KWS_ZH_GUIDE.md')
            if ($dedicated -and $CacheSizeKiB -eq 256) {
                Copy-Item -LiteralPath 'docs/use_cases/kws_zh_dedicated256.md' `
                    -Destination (Join-Path $output 'DEDICATED_256K_FIX.md')
            }
            $modelParent = Split-Path -Parent $model
            foreach ($name in @('compilation.json', 'out_per-layer.csv')) {
                $source = Join-Path $modelParent $name
                if (Test-Path -LiteralPath $source) {
                    Copy-Item -LiteralPath $source -Destination (Join-Path $output 'validation')
                }
            }
            Copy-Item -LiteralPath 'resources/kws_zh/samples/manifest.json' `
                -Destination (Join-Path $output 'validation/samples-manifest.json')
            $fvpDirectory = Join-Path $BuildRoot "pte-$variant-fvp-$BuildTag"
            foreach ($name in @('fvp.log', 'fvp_comparison.json')) {
                $source = Join-Path $fvpDirectory $name
                if (Test-Path -LiteralPath $source) {
                    Copy-Item -LiteralPath $source -Destination (Join-Path $output 'validation')
                }
            }
            $frontendReport = Join-Path $BuildRoot 'frontend-validation/fbank_comparison.json'
            if (Test-Path -LiteralPath $frontendReport) {
                Copy-Item -LiteralPath $frontendReport -Destination (Join-Path $output 'validation')
            }
            & (Join-Path $PSScriptRoot 'package-fpga.ps1') -BuildDirectory $directory `
                -CompilerBin $compiler -OutputDirectory $output
        }
    }
} finally {
    Pop-Location
}
