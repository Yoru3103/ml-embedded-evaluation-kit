# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<#
.SYNOPSIS
Launch a native Windows Corstone-320 FVP using a build or demonstration profile.
.EXAMPLE
.\run-fvp.ps1 -ListProfiles
.EXAMPLE
.\run-fvp.ps1 -Profile Vision -Demo img-class -Mode Static -Headless -SmokeTimeoutSeconds 300
#>
[CmdletBinding()]
param(
    [string]$Profile = 'Firmware',
    [switch]$ListProfiles,
    [switch]$ProfileHelp
)
dynamicparam {
    . (Join-Path $PSScriptRoot 'internal/profile-tools.ps1')
    $profiles = (Import-PowerShellDataFile (Join-Path $PSScriptRoot 'internal/profiles.psd1')).Run
    if (-not $ListProfiles) {
        $selected = if ($PSBoundParameters.ContainsKey('Profile')) {
            $PSBoundParameters['Profile']
        } else { 'Firmware' }
        $command = Get-MlekProfileCommand -Profiles $profiles -Name $selected
        if (-not $ProfileHelp) {
            $exclude = if ($selected -eq 'PicoFace') { @('Action') } else { @() }
            Get-MlekProfileParameters -Command $command -Exclude $exclude
        }
    }
}
end {
    if ($ListProfiles) { $profiles.Keys | Sort-Object; return }
    if ($ProfileHelp) { Get-Help $command.Path -Full; return }
    foreach ($name in @('Profile', 'ListProfiles', 'ProfileHelp')) {
        $null = $PSBoundParameters.Remove($name)
    }
    if ($selected -eq 'PicoFace') { $PSBoundParameters['Action'] = 'Run' }
    if ($selected -notin @('Firmware', 'Vision', 'VisionBenchmark') -and
        $command.Parameters.ContainsKey('FvpExecutable') -and
        -not $PSBoundParameters.ContainsKey('FvpExecutable') -and
        -not $PSBoundParameters['PlanOnly']) {
        . (Join-Path $PSScriptRoot 'internal/fvp-tools.ps1')
        $PSBoundParameters['FvpExecutable'] = Resolve-WindowsFvpExecutable
    }
    & $command.Path @PSBoundParameters
}
