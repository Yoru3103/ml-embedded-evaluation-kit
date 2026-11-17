# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<#
.SYNOPSIS
Build native Windows MLEK firmware or prepare resources for a build profile.
.EXAMPLE
.\build.ps1 -ListProfiles
.EXAMPLE
.\build.ps1 -Profile Vision -Demo img-class -Mode Static -Action All
#>
[CmdletBinding()]
param(
    [string]$Profile = 'Firmware',
    [switch]$ListProfiles,
    [switch]$ProfileHelp
)
dynamicparam {
    . (Join-Path $PSScriptRoot 'internal/profile-tools.ps1')
    $profiles = (Import-PowerShellDataFile (Join-Path $PSScriptRoot 'internal/profiles.psd1')).Build
    if (-not $ListProfiles) {
        $selected = if ($PSBoundParameters.ContainsKey('Profile')) {
            $PSBoundParameters['Profile']
        } else { 'Firmware' }
        $command = Get-MlekProfileCommand -Profiles $profiles -Name $selected
        if (-not $ProfileHelp) { Get-MlekProfileParameters -Command $command }
    }
}
end {
    if ($ListProfiles) { $profiles.Keys | Sort-Object; return }
    if ($ProfileHelp) { Get-Help $command.Path -Full; return }
    foreach ($name in @('Profile', 'ListProfiles', 'ProfileHelp')) {
        $null = $PSBoundParameters.Remove($name)
    }
    & $command.Path @PSBoundParameters
}
