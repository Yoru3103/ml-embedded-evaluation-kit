# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

function Test-WindowsVenv {
    param([Parameter(Mandatory = $true)][string]$Python)

    if (-not (Test-Path -LiteralPath $Python)) {
        return $false
    }

    $savedEnvironment = @{}
    foreach ($name in @(
            'PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'PIP_CONFIG_FILE', 'SSLKEYLOGFILE')) {
        $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        [Environment]::SetEnvironmentVariable($name, $null, 'Process')
    }
    try {
        $null = & $Python -c 'import pip' 2>$null
        $success = $LASTEXITCODE -eq 0
    } finally {
        foreach ($name in $savedEnvironment.Keys) {
            [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
        }
    }
    return $success
}

function Ensure-WindowsVenv {
    param([Parameter(Mandatory = $true)][string]$VenvDir)

    $python = Join-Path $VenvDir 'Scripts/python.exe'
    if (Test-WindowsVenv -Python $python) {
        return $python
    }

    $venvConfig = Join-Path $VenvDir 'pyvenv.cfg'
    if ((Test-Path -LiteralPath $VenvDir) -and
        -not (Test-Path -LiteralPath $venvConfig)) {
        throw "The existing directory is not a Python virtual environment: $VenvDir"
    }

    $savedEnvironment = @{}
    foreach ($name in @(
            'PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'PIP_CONFIG_FILE', 'SSLKEYLOGFILE')) {
        $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        [Environment]::SetEnvironmentVariable($name, $null, 'Process')
    }

    try {
        $arguments = @('-3.12', '-m', 'venv')
        if (Test-Path -LiteralPath $VenvDir) {
            $arguments += '--clear'
        }
        $arguments += $VenvDir
        $venvOutput = & py @arguments 2>&1
        if ($LASTEXITCODE -ne 0) {
            $details = ($venvOutput | Out-String).Trim()
            throw "py -3.12 -m venv failed with exit code $LASTEXITCODE. $details"
        }
    } finally {
        foreach ($name in $savedEnvironment.Keys) {
            [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
        }
    }

    if (-not (Test-WindowsVenv -Python $python)) {
        throw "The Python virtual environment was created without a usable pip: $VenvDir"
    }
    return $python
}

function Get-WindowsBuildTools {
    param([Parameter(Mandatory = $true)][string]$ToolsVenvDir)

    $binDir = Join-Path $ToolsVenvDir 'Scripts'
    $cmake = Join-Path $binDir 'cmake.exe'
    $ninja = Join-Path $binDir 'ninja.exe'

    foreach ($tool in @($cmake, $ninja)) {
        if (-not (Test-Path -LiteralPath $tool)) {
            throw "Shared Windows build tool not found: $tool. Run either build script with -Action Setup."
        }
    }

    & $cmake --version *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Shared CMake is not usable: $cmake. Run either build script with -Action Setup."
    }
    & $ninja --version *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Shared Ninja is not usable: $ninja. Run either build script with -Action Setup."
    }

    return [PSCustomObject]@{
        BinDir = $binDir
        CMake = $cmake
        Ninja = $ninja
    }
}

function Ensure-WindowsBuildTools {
    param([Parameter(Mandatory = $true)][string]$ToolsVenvDir)

    $python = Ensure-WindowsVenv -VenvDir $ToolsVenvDir
    & $python -m pip install --upgrade cmake ninja 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install the shared latest CMake and Ninja into $ToolsVenvDir."
    }

    return (Get-WindowsBuildTools -ToolsVenvDir $ToolsVenvDir)
}

function Resolve-WindowsToolchainBin {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet('AC6', 'ATFE', 'GNU')]
        [string]$Toolchain,
        [string]$CompilerBin
    )

    $compilerName = switch ($Toolchain) {
        'AC6' { 'armclang.exe' }
        'ATFE' { 'clang.exe' }
        'GNU' { 'arm-none-eabi-gcc.exe' }
    }
    $environmentName = switch ($Toolchain) {
        'AC6' { 'MLEK_AC6_BIN' }
        'ATFE' { 'MLEK_ATFE_BIN' }
        'GNU' { 'MLEK_GNU_BIN' }
    }

    $candidateBin = $CompilerBin
    if (-not $candidateBin) {
        $candidateBin = [Environment]::GetEnvironmentVariable($environmentName, 'Process')
    }
    if ($candidateBin) {
        $resolvedBin = (Resolve-Path -LiteralPath $candidateBin).ProviderPath
        if (-not (Test-Path -LiteralPath (Join-Path $resolvedBin $compilerName))) {
            throw "$Toolchain compiler $compilerName was not found in $resolvedBin."
        }
        return $resolvedBin
    }

    $pathCompiler = Get-Command $compilerName -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($pathCompiler) {
        return Split-Path -Parent $pathCompiler.Source
    }

    $patterns = switch ($Toolchain) {
        'AC6' {
            @(
                'Program Files/ArmCompilerforEmbedded*/bin/armclang.exe',
                'Program Files/Arm/Development Studio */sw/ARMCompiler*/bin/armclang.exe'
            )
        }
        'ATFE' {
            @('Program Files/Arm/Development Studio */sw/atfe-*/bin/clang.exe')
        }
        'GNU' {
            @(
                'Program Files/Arm GNU Toolchain */bin/arm-none-eabi-gcc.exe',
                'Program Files (x86)/Arm GNU Toolchain */bin/arm-none-eabi-gcc.exe'
            )
        }
    }

    $discovered = foreach ($drive in Get-PSDrive -PSProvider FileSystem) {
        foreach ($pattern in $patterns) {
            Get-ChildItem -Path (Join-Path $drive.Root $pattern) -File -ErrorAction SilentlyContinue
        }
    }
    $selected = $discovered | Sort-Object FullName -Descending | Select-Object -First 1
    if ($selected) {
        return $selected.DirectoryName
    }

    throw "Unable to locate the $Toolchain toolchain. Supply -CompilerBin or set $environmentName."
}
