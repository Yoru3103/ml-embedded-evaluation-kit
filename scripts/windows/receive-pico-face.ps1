# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# Receive the stored FPGA image without a debugger or any new inference. #>
[CmdletBinding()]
param(
    [string]$Port,
    [string]$Output,
    [switch]$ListPorts,
    [switch]$WaitReady,
    [string]$Capture,
    [ValidateRange(1, 600)][int]$TimeoutSeconds = 90
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).ProviderPath
$imageRoot = Join-Path $repoRoot 'build/logs/pico_face_logs/out'
if ($Output) {
    # Keep legacy -Output out/face inside the dedicated image directory too.
    $relative = $Output.Replace('\', '/').TrimEnd('/')
    if ($relative.StartsWith('build/logs/pico_face_logs/out/',
            [StringComparison]::OrdinalIgnoreCase)) {
        $relative = $relative.Substring('build/logs/pico_face_logs/out/'.Length)
    } elseif ($relative.StartsWith('out/', [StringComparison]::OrdinalIgnoreCase)) {
        $relative = $relative.Substring(4)
    }
    $Output = if ([IO.Path]::IsPathRooted($relative)) { [IO.Path]::GetFullPath($relative) }
              else { [IO.Path]::GetFullPath((Join-Path $imageRoot $relative)) }
    if (-not $Output.StartsWith($imageRoot + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase)) {
        throw "Image output must stay inside $imageRoot"
    }
}
$python = Join-Path $repoRoot 'resources_downloaded/windows-executorch/env/Scripts/python.exe'
$savedPythonPath = $env:PYTHONPATH
Push-Location $repoRoot
try {
    $env:PYTHONPATH = Join-Path $repoRoot 'scripts/py'
    if ($ListPorts) {
        [System.IO.Ports.SerialPort]::GetPortNames() | Sort-Object
        return
    }
    if (-not $Capture -and $Port) {
        # Native .NET serial I/O avoids an extra pyserial dependency on Windows.
        $serial = [System.IO.Ports.SerialPort]::new($Port, 115200,
            [System.IO.Ports.Parity]::None, 8, [System.IO.Ports.StopBits]::One)
        $serial.Handshake = [System.IO.Ports.Handshake]::None
        $serial.DtrEnable = $false
        $serial.RtsEnable = $false
        $serial.ReadTimeout = 500
        $serial.WriteTimeout = 5000
        $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
        function Read-SerialBytes([int]$Count) {
            $buffer = [byte[]]::new($Count)
            $received = 0
            while ($received -lt $Count) {
                if ([DateTime]::UtcNow -ge $deadline) { throw 'Serial receive timed out.' }
                try { $received += $serial.Read($buffer, $received, $Count - $received) }
                catch [System.TimeoutException] { }
            }
            return ,$buffer
        }
        function Find-SerialMarker([string]$Marker) {
            $bytes = [Text.Encoding]::ASCII.GetBytes($Marker)
            $matched = 0
            while ($matched -lt $bytes.Length) {
                $value = (Read-SerialBytes 1)[0]
                if ($value -eq $bytes[$matched]) { $matched++ }
                elseif ($value -eq $bytes[0]) { $matched = 1 }
                else { $matched = 0 }
            }
        }
        try {
            Write-Host 'Close the serial terminal before receiving the image.'
            $serial.Open()
            $serial.DiscardInBuffer()
            if ($WaitReady) {
                Write-Host 'Waiting for SERIAL_IMAGE_READY; reset the board now.'
                Find-SerialMarker 'SERIAL_IMAGE_READY'
            }
            $serial.Write('I')
            Find-SerialMarker 'PFIMAGE1'
            $header = [byte[]]([Text.Encoding]::ASCII.GetBytes('PFIMAGE1') +
                (Read-SerialBytes 48))
            $size = [BitConverter]::ToUInt32($header, 24)
            if ($size -ne 49152) { throw "Unexpected RGB length: $size" }
            $rgb = Read-SerialBytes ([int]$size)
            if (-not $Output) {
                $seed = [BitConverter]::ToUInt32($header, 28)
                $steps = [BitConverter]::ToUInt32($header, 32)
                $stamp = [DateTime]::Now.ToString('yyyyMMdd_HHmmss_fff')
                $Output = Join-Path $imageRoot "face_seed${seed}_steps${steps}_$stamp"
            }
            $prefix = [IO.Path]::GetFullPath($Output)
            $Capture = [IO.Path]::ChangeExtension($prefix, '.pfframe')
            [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Capture)) | Out-Null
            [IO.File]::WriteAllBytes($Capture, [byte[]]($header + $rgb))
        } finally {
            if ($serial.IsOpen) { $serial.Close() }
            $serial.Dispose()
        }
    }
    $arguments = @('-m', 'mlek_tools.pico_face.receive', '--timeout', "$TimeoutSeconds")
    if ($Output) { $arguments += @('--output', $Output) }
    if ($Capture) { $arguments += @('--capture', $Capture) }
    else { throw 'Select -Port COMx, -ListPorts or -Capture <binary UART capture>.' }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Image receive/CRC verification failed.' }
} finally {
    $env:PYTHONPATH = $savedPythonPath
    Pop-Location
}
