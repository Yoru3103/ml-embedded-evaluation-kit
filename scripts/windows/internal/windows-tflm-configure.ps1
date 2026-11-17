# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

function Invoke-WindowsTflmConfigure {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$BuildDirectory
    )

    # CMake has already verified and extracted the archive when this rename fails.
    # Recover only these build-local TFLM sources, never another build's cache.
    $dependencyRoot = [IO.Path]::GetFullPath((Join-Path $BuildDirectory '_deps'))
    $recovered = @{}
    while ($true) {
        $originalPreference = $ErrorActionPreference
        try {
            # Windows PowerShell 5.1 turns redirected native stderr into error
            # records. CMake also writes ordinary progress there; use its exit
            # code to determine failure rather than stopping on that output.
            $ErrorActionPreference = 'Continue'
            $output = @(& $Executable @Arguments 2>&1)
            $configureExit = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $originalPreference
        }
        # Empty native stderr records stringify as a type name in PowerShell
        # 5.1. Preserve their actual text so multiline CMake diagnostics parse.
        $output = @($output | ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) {
                $_.Exception.Message
            } else {
                "$_"
            }
        })
        $output | ForEach-Object { Write-Host "$_" }
        if ($configureExit -eq 0) { return }

        $message = ($output | ForEach-Object { "$_" }) -join "`n"
        $failure = [regex]::Match($message,
            'file RENAME failed to rename\s+(?<source>[^\r\n]+)\s+to\s+(?<destination>[^\r\n]+)\s+because:')
        if (-not $failure.Success) {
            throw "$Executable failed with exit code $configureExit."
        }
        $source = [IO.Path]::GetFullPath($failure.Groups['source'].Value.Trim())
        $destination = [IO.Path]::GetFullPath($failure.Groups['destination'].Value.Trim())
        $dependency = [regex]::Match([IO.Path]::GetFileName($destination),
            '^(?<name>tensorflow-(?:flatbuffers|gemlowp|ruy))-src$')
        if (-not $dependency.Success -or
            [IO.Path]::GetDirectoryName($destination) -ine $dependencyRoot) {
            throw "CMake rename failed outside the supported build-local TFLM dependencies."
        }
        $name = $dependency.Groups['name'].Value
        $sourcePrefix = $dependencyRoot.TrimEnd('\') + '\'
        if (-not $source.StartsWith($sourcePrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "The failed extraction source is outside this build's dependency directory."
        }
        $relativeSource = $source.Substring($sourcePrefix.Length).Replace('\', '/')
        $expectedSource = '^ex-' + [regex]::Escape($name) + '-populate[0-9]+(?:/[^/]+)?$'
        if ($relativeSource -notmatch $expectedSource -or $recovered.ContainsKey($name) -or
            -not (Test-Path -LiteralPath $source -PathType Container) -or
            (Test-Path -LiteralPath $destination)) {
            throw "Cannot safely recover the extracted $name directory; see the CMake error above."
        }

        $extractScript = Join-Path $dependencyRoot (
            "$name-subbuild/$name-populate-prefix/src/$name-populate-stamp/extract-$name-populate.cmake")
        if (-not (Test-Path -LiteralPath $extractScript -PathType Leaf)) {
            throw "The generated extraction script for $name is missing."
        }
        Write-Host "Recovering the extracted $name source after CMake's Windows rename failure."
        for ($attempt = 1; $attempt -le 5; $attempt++) {
            try {
                Move-Item -LiteralPath $source -Destination $destination -ErrorAction Stop
                break
            } catch {
                if ($attempt -eq 5) { throw }
                Start-Sleep -Seconds 1
            }
        }
        # Use FetchContent's supported source override. It is scoped to this build
        # and disappears with the build directory; a clean build downloads normally.
        $sourceOption = "-DFETCHCONTENT_SOURCE_DIR_$($name.ToUpperInvariant())="
        $Arguments = @($Arguments | Where-Object { -not $_.StartsWith($sourceOption) })
        $Arguments += $sourceOption + $destination.Replace('\', '/')
        $recovered[$name] = $true
    }
}
