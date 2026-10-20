# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# Shared naming, options and provenance for TFLM and ExecuTorch firmware. #>

function Get-WindowsFirmwareBuildDirectory {
    param(
        [string]$RepoRoot, [string]$Framework, [string]$Toolchain, [string]$Target,
        [string]$UseCase, [string]$MemoryMode, [int]$CacheSizeKiB,
        [string]$BuildType, [string]$PmuGroup
    )
    $memory = if ($MemoryMode) { $MemoryMode.ToLowerInvariant() } else { 'default-memory' }
    $cache = if ($MemoryMode -in @('Shared_Sram', 'Sram_Only')) { 'no-cache' }
             elseif ($CacheSizeKiB) { "${CacheSizeKiB}k" } else { 'default-cache' }
    $pmu = if ($PmuGroup) { $PmuGroup.ToLowerInvariant() } else { 'default-pmu' }
    $relative = 'build/{0}/{1}/{2}/{3}/{4}/{5}-{6}-{7}' -f `
        $UseCase, $Framework.ToLowerInvariant(), $Toolchain.ToLowerInvariant(), `
        $Target.ToLowerInvariant(), $memory, $cache, $BuildType.ToLowerInvariant(), $pmu
    return Join-Path $RepoRoot $relative
}

function Get-WindowsFirmwareCMakeArgs {
    param([string]$BuildType, [int]$CacheSizeKiB, [string]$TimingAdapterConfig, [string]$PmuGroup)
    $definitions = @("-DCMAKE_BUILD_TYPE=$BuildType", '-DUSE_SINGLE_INPUT=ON')
    if ($CacheSizeKiB) { $definitions += "-DETHOS_U_NPU_CACHE_SIZE=$($CacheSizeKiB * 1024)" }
    if ($TimingAdapterConfig) { $definitions += "-DTA_CONFIG_FILE=$TimingAdapterConfig" }
    if ($PmuGroup) { $definitions += "-DETHOS_U_NPU_PMU_GROUP=$PmuGroup" }
    return $definitions
}

function Write-WindowsFirmwareBuildInfo {
    param(
        [string]$BuildDirectory, [string]$Framework, [string]$UseCase,
        [string]$CMakeExecutable, [string[]]$ConfigureArgs,
        [string]$ModelPath, [string]$Status, [int]$Jobs
    )
    New-Item -ItemType Directory -Path $BuildDirectory -Force | Out-Null
    $configuration = [ordered]@{}
    $cachePath = Join-Path $BuildDirectory 'CMakeCache.txt'
    if (Test-Path -LiteralPath $cachePath) {
        foreach ($line in Get-Content -LiteralPath $cachePath) {
            if ($line -match '^([A-Za-z0-9_]+):[^=]+=(.*)$') {
                $name, $value = $Matches[1], $Matches[2]
                if ($name -match '^(CMAKE_BUILD_TYPE|ML_FRAMEWORK|ML_FWK_TMP_MEM_(BASE|SIZE)|FB_BASE|TARGET_PLATFORM|TARGET_SUBSYSTEM|FPGA_PLATFORM_SSE_320|ETHOS_U_NPU_.*|TA_CONFIG_FILE|TA_SRAM_BASE|TA_EXT_BASE|INTERACTIVE_MODE|SEMIHOSTING_ENABLED|CPU_PROFILE_ENABLED|USE_SINGLE_INPUT)$' -or
                    $name.StartsWith("${UseCase}_")) {
                    $configuration[$name] = $value
                }
            }
        }
    }
    $modelHash = if ($ModelPath -and (Test-Path -LiteralPath $ModelPath)) {
        $ModelPath = (Resolve-Path -LiteralPath $ModelPath).ProviderPath.Replace('\', '/')
        (Get-FileHash -LiteralPath $ModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    } else { $null }
    $info = [ordered]@{
        status = $Status
        recorded_at_utc = [DateTimeOffset]::UtcNow.ToString('o')
        framework = $Framework
        use_case = $UseCase
        model_path = $ModelPath
        model_sha256 = $modelHash
        configuration = $configuration
        configure_command = @($CMakeExecutable) + $ConfigureArgs
        build_command = @($CMakeExecutable, '--build', $BuildDirectory, '--parallel', "$Jobs")
    }
    $info | ConvertTo-Json -Depth 5 | Set-Content `
        -LiteralPath (Join-Path $BuildDirectory 'build-info.json') -Encoding utf8
}
