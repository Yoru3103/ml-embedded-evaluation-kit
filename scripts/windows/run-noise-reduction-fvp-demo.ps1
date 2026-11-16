# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
<# Run the SSE-320 RNNoise demo and collect the WAV files in a new output directory. #>
[CmdletBinding()]
param(
    [string]$BuildDirectory = 'build/fvp-noise-reduction',
    [string]$OutputDirectory,
    [string]$FvpExecutable = 'C:/Program Files/Arm/FVP_Corstone_SSE-320/models/Win64_VC2019/FVP_Corstone_SSE-320.exe',
    [switch]$Headless,
    [switch]$Quiet
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).ProviderPath
Push-Location $repoRoot
try {
    $buildPath = (Resolve-Path -LiteralPath $BuildDirectory).ProviderPath
    $fvp = (Resolve-Path -LiteralPath $FvpExecutable).ProviderPath
    $firmware = (Resolve-Path -LiteralPath (Join-Path $buildPath 'bin/mlek_noise_reduction.axf')).ProviderPath
    $cache = Get-Content -LiteralPath (Join-Path $buildPath 'CMakeCache.txt')
    foreach ($setting in @('TARGET_PLATFORM=mps4', 'TARGET_SUBSYSTEM=sse-320',
            'SEMIHOSTING_ENABLED=ON', 'noise_reduction_SAVE_WAV=ON')) {
        $key, $value = $setting.Split('=')
        if (-not ($cache -match "^${key}:[^=]+=$value`$")) {
            throw "Expected build setting $setting. Run build-noise-reduction-fvp-demo.ps1."
        }
    }
    if (-not $OutputDirectory) {
        $OutputDirectory = Join-Path $buildPath ('wav/' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
    }
    $destination = (New-Item -ItemType Directory -Path $OutputDirectory).FullName
} finally {
    Pop-Location
}

$fvpArgs = @('-a', $firmware, '-C', 'mps4_board.subsystem.ethosu.num_macs=1024',
    '-C', 'mps4_board.subsystem.ethosu.extra_args=--fast',
    '-C', 'mps4_board.subsystem.cpu0.semihosting-enable=1',
    '-C', 'mps4_board.subsystem.cpu0.semihosting-stack_base=0',
    '-C', 'mps4_board.subsystem.cpu0.semihosting-stack_limit=0',
    '-C', 'mps4_board.visualisation.disable-visualisation=1',
    '-C', "vis_hdlcd.disable_visualisation=$(if ($Headless) {1} else {0})")
foreach ($uart in @(0, 1, 2, 5)) {
    $fvpArgs += @('-C', "mps4_board.telnetterminal$uart.start_telnet=0")
}

Write-Host "FVP output directory: $destination"
Push-Location $destination
try {
    $log = Join-Path $destination 'fvp.log'
    if ($Quiet) {
        & $fvp @fvpArgs *> $log
    } else {
        & $fvp @fvpArgs | Tee-Object -FilePath $log
    }
    $fvpExitCode = $LASTEXITCODE
    if ($fvpExitCode -ne 0) { throw "FVP exited with code $fvpExitCode. See $log" }
    if (-not (Select-String -LiteralPath $log -SimpleMatch 'Main loop terminated successfully.')) {
        throw "Noise reduction did not report successful completion. See $log"
    }
    $outputs = @(Get-ChildItem -LiteralPath $destination -Filter 'denoised_*.wav' -File)
    if ($outputs.Count -eq 0) { throw "FVP produced no WAV files. See $log" }
    $outputs | ForEach-Object { Write-Host $_.FullName }
} finally {
    Pop-Location
}
