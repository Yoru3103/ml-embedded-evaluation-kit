# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

function Resolve-WindowsFvpExecutable {
    param([string]$Executable)

    if (-not $Executable) { $Executable = $env:FVP_PATH }
    if ($Executable) {
        return (Resolve-Path -LiteralPath $Executable -ErrorAction Stop).ProviderPath
    }

    $name = 'FVP_Corstone_SSE-320.exe'
    $command = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($command) { return $command.Path }

    $model = "Arm/FVP_Corstone_SSE-320/models/Win64_VC2019/$name"
    $candidate = Join-Path $env:ProgramFiles $model
    if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    foreach ($drive in Get-PSDrive -PSProvider FileSystem) {
        $candidate = Join-Path $drive.Root "Program Files/$model"
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    throw 'Corstone-320 FVP was not found. Set FVP_PATH or pass -FvpExecutable.'
}
