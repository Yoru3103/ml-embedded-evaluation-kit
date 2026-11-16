# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<#
.SYNOPSIS
Upload one WAV to the dedicated Conformer UART demo and print one result.
.DESCRIPTION
Close MobaXterm on the same COM port first. Upload and printing are excluded
from the board's end-to-end timing. Optional -Denoise requires Python with numpy
and soundfile; the normal upload requires no Python or external serial package.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$WavPath,
    [string]$Port,
    [string]$TcpHost,
    [ValidateRange(1, 65535)][int]$TcpPort = 5000,
    [ValidateSet(115200, 460800, 921600)][int]$BaudRate = 115200,
    [ValidateRange(1, 86400)][int]$TimeoutSeconds = 60,
    [switch]$Denoise,
    [ValidateRange(-100, -20)][double]$NoiseThresholdDbfs = -40,
    [ValidateSet('auto', 'head', 'tail')][string]$NoiseSource = 'auto',
    [ValidateRange(0.2, 5)][double]$EdgeSeconds = 1,
    [ValidateRange(0, 12)][double]$MaxReductionDb = 6,
    [string]$DenoisePython,
    [switch]$ValidateOnly
)
$ErrorActionPreference = 'Stop'
$wallTimer = [Diagnostics.Stopwatch]::StartNew()
. (Join-Path $PSScriptRoot 'asr-uart-tools.ps1')
$path = (Resolve-Path -LiteralPath $WavPath).ProviderPath
$audio = [Mlek.AsrAudio]::Load($path)
if (-not $ValidateOnly -and ((-not $Port -and -not $TcpHost) -or ($Port -and $TcpHost))) {
    throw 'Specify exactly one of -Port (FPGA) or -TcpHost (FVP).'
}
$denoiseMs = 0.0
$denoiseReport = $null
if ($Denoise) {
    . (Join-Path $PSScriptRoot 'asr-denoise-tools.ps1')
    $denoiseTimer = [Diagnostics.Stopwatch]::StartNew()
    $processed = Invoke-AsrDenoise -Path $path -ThresholdDbfs $NoiseThresholdDbfs `
        -Source $NoiseSource -EdgeSeconds $EdgeSeconds -ReductionDb $MaxReductionDb `
        -Python $DenoisePython
    $audio = $processed.Audio
    $denoiseReport = $processed.Report
    $denoiseMs = $denoiseTimer.Elapsed.TotalMilliseconds
}
if ($ValidateOnly) {
    $validation = [ordered]@{ File=$path; Samples=$audio.Samples; Bytes=$audio.Pcm.Length; Crc32=$audio.Crc }
    if ($Denoise) {
        $validation.DenoiseApplied = $denoiseReport.applied
        $validation.DenoiseHostMs = $denoiseMs
        $validation.NoiseRegions = $denoiseReport.regions
    }
    [pscustomobject]$validation
    return
}
$requestId = [uint32](Get-Random -Minimum 1 -Maximum 2147483647)
if ($TcpHost) {
    . (Join-Path $PSScriptRoot 'asr-tcp-transport.ps1')
    $serial = [Mlek.AsrTcpTransport]::new($TcpHost, $TcpPort, 5000)
} else {
    $serial = [IO.Ports.SerialPort]::new($Port, $BaudRate, [IO.Ports.Parity]::None,
        8, [IO.Ports.StopBits]::One)
    $serial.Handshake = [IO.Ports.Handshake]::None
    $serial.DtrEnable = $false
    $serial.RtsEnable = $false
    $serial.Encoding = [Text.UTF8Encoding]::new($false)
    $serial.NewLine = "`n"
    $serial.ReadTimeout = 500
    $serial.WriteTimeout = 5000
}
try {
    $serial.Open()
    . (Join-Path $PSScriptRoot 'asr-uart-transfer.ps1')
    $transaction = Invoke-AsrUartTransfer -Serial $serial -Audio $audio `
        -RequestId $requestId -TimeoutSeconds $TimeoutSeconds
    $response = $transaction.Reply
    if ($response -notmatch "^ASR_RESULT $requestId ([0-9]+) ([0-9]+) (.*)$") {
        throw 'Malformed target result.'
    }
    $npuMs = ([double]$Matches[1] / 1000).ToString('F3', [Globalization.CultureInfo]::InvariantCulture)
    $totalMs = ([double]$Matches[2] / 1000).ToString('F3', [Globalization.CultureInfo]::InvariantCulture)
    $text = $Matches[3]
    $wallTimer.Stop()
    Write-Output "File: $([IO.Path]::GetFileName($path))"
    Write-Output "Text: $text"
    Write-Output "NPU.Driver: $npuMs ms"
    Write-Output "End-to-end: $totalMs ms"
    $uploadMs = $transaction.UploadMs.ToString('F3', [Globalization.CultureInfo]::InvariantCulture)
    $requestMs = $transaction.RequestMs.ToString('F3', [Globalization.CultureInfo]::InvariantCulture)
    $transport = if ($TcpHost) { 'TCP' } else { 'UART' }
    Write-Output "Upload.${transport}: $uploadMs ms"
    Write-Output "Request.Wall: $requestMs ms"
    if ($Denoise) {
        $state = if ($denoiseReport.applied) { 'applied' } else { 'skipped (no quiet edge region)' }
        Write-Output "Denoise: $state"
        Write-Output "Denoise.Host: $($denoiseMs.ToString('F3', [Globalization.CultureInfo]::InvariantCulture)) ms"
        Write-Output "Total.Wall: $($wallTimer.Elapsed.TotalMilliseconds.ToString('F3', [Globalization.CultureInfo]::InvariantCulture)) ms"
    }
} finally {
    if ($serial.IsOpen) { $serial.Close() }
    $serial.Dispose()
}
