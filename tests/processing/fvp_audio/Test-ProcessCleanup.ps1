# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# Small software acceptance: cancellation state and cleanup of owned dummy processes. #>
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).ProviderPath
. (Join-Path $repoRoot 'scripts/windows/internal/fvp-audio-process-tools.ps1')
$python = Join-Path $repoRoot 'resources_downloaded/windows-executorch/env/Scripts/python.exe'
$children = @()
try {
    [Mlek.FvpAudioConsole]::Start()
    if ([Mlek.FvpAudioConsole]::Requested) { throw 'Cancellation state was not reset.' }
    [Mlek.FvpAudioConsole]::RequestStop()
    if (-not (Test-FvpAudioStopRequested)) { throw 'Cancellation was not observed.' }
    foreach ($index in 1..2) {
        $children += Start-Process -FilePath $python -ArgumentList @('-c',
            '"import time; time.sleep(60)"') -WindowStyle Hidden -PassThru
    }
    foreach ($child in $children) {
        $childId = $child.Id
        Stop-FvpAudioProcess $child
        if (Get-Process -Id $childId -ErrorAction SilentlyContinue) {
            throw "Dummy process $childId survived cleanup."
        }
    }
    $children = @()
    Write-Output 'Console cancellation and owned-process cleanup: PASS'
} finally {
    foreach ($child in $children) { Stop-FvpAudioProcess $child }
    [Mlek.FvpAudioConsole]::Stop()
}
