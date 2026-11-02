# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# Upload a WAV or image to a resident FPGA file demo; TCP supports FVP validation. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet('kws_zh','asr','hand-pose','qrcode')][string]$UseCase,
    [Parameter(Mandatory=$true)][string]$InputPath,
    [string]$Port,
    [string]$TcpHost,
    [ValidateRange(1,65535)][int]$TcpPort = 5000,
    [ValidateSet(115200,460800,921600)][int]$BaudRate = 115200,
    [ValidateRange(1,86400)][int]$TimeoutSeconds = 180,
    [ValidateSet('square','letterbox')][string]$Framing = 'square',
    [string]$Python = 'resources_downloaded/windows-executorch/env/Scripts/python.exe',
    [switch]$ValidateOnly
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).ProviderPath
if (-not $ValidateOnly -and ((-not $Port -and -not $TcpHost) -or ($Port -and $TcpHost))) {
    throw 'Specify exactly one of -Port (FPGA) or -TcpHost (FVP).'
}
if ($UseCase -eq 'asr') {
    $arguments = @{WavPath=$InputPath; BaudRate=$BaudRate; TimeoutSeconds=$TimeoutSeconds;
        ValidateOnly=$ValidateOnly}
    if ($Port) { $arguments.Port=$Port }
    if ($TcpHost) { $arguments.TcpHost=$TcpHost; $arguments.TcpPort=$TcpPort }
    & (Join-Path $PSScriptRoot 'send-asr-audio.ps1') @arguments
    return
}
. (Join-Path $PSScriptRoot 'asr-uart-tools.ps1')
$requestId = [uint32](Get-Random -Minimum 1 -Maximum 2147483647)
$inputFile = (Resolve-Path -LiteralPath $InputPath).ProviderPath
if ($UseCase -eq 'kws_zh') {
    $audio = [Mlek.AsrAudio]::Load($inputFile)
    $magic='KWD1'; $width=[uint32]16000; $height=$audio.Samples; $payload=$audio.Pcm
} else {
    $tempDirectory = Join-Path $repoRoot "build/file-demo-inputs/$requestId"
    New-Item -ItemType Directory -Force $tempDirectory | Out-Null
    $output = Join-Path $tempDirectory 'input.bin'
    $preview = Join-Path $tempDirectory 'input-preview.png'
    if (-not [IO.Path]::IsPathRooted($Python)) { $Python=Join-Path $repoRoot $Python }
    if ($UseCase -eq 'hand-pose') {
        & $Python (Join-Path $repoRoot 'scripts/py/prepare_hand_pose_demo.py') `
            --input $inputFile --output $output --preview $preview --framing $Framing
        $magic='HPD1'; $width=[uint32]224; $height=[uint32]224
    } else {
        & $Python (Join-Path $repoRoot 'scripts/py/prepare_qrcode_demo.py') `
            --input $inputFile --output $output --preview $preview --width 256 --height 256
        $magic='QRD1'; $width=[uint32]256; $height=[uint32]256
    }
    if ($LASTEXITCODE -ne 0) { throw 'Image preparation failed.' }
    $packet = [IO.File]::ReadAllBytes($output)
    $payload = New-Object byte[] ($packet.Length-24)
    [Array]::Copy($packet,24,$payload,0,$payload.Length)
    Write-Host "Model input preview: $preview"
}
$memory = [IO.MemoryStream]::new()
$writer = [IO.BinaryWriter]::new($memory)
$writer.Write([Text.Encoding]::ASCII.GetBytes($magic))
$writer.Write($requestId); $writer.Write([uint32]$width); $writer.Write([uint32]$height)
$writer.Write([uint32]$payload.Length); $writer.Write([Mlek.AsrAudio]::Crc32($payload))
$writer.Flush(); $header=$memory.ToArray(); $writer.Dispose(); $memory.Dispose()
if ($ValidateOnly) {
    [pscustomobject]@{UseCase=$UseCase; Magic=$magic; Width=$width; Height=$height;
        Bytes=$payload.Length; HeaderBytes=$header.Length}
    return
}
if ($TcpHost) {
    . (Join-Path $PSScriptRoot 'asr-tcp-transport.ps1')
    $transport = [Mlek.AsrTcpTransport]::new($TcpHost,$TcpPort,5000)
} else {
    $transport = [IO.Ports.SerialPort]::new($Port,$BaudRate,[IO.Ports.Parity]::None,
        8,[IO.Ports.StopBits]::One)
    $transport.Handshake=[IO.Ports.Handshake]::None
    $transport.DtrEnable=$false; $transport.RtsEnable=$false
    $transport.ReadTimeout=500; $transport.WriteTimeout=5000
    $transport.NewLine="`n"
}
function Read-FileReply([string]$Pattern) {
    $timer=[Diagnostics.Stopwatch]::StartNew()
    while ($timer.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        try { $line=$transport.ReadLine().TrimEnd("`r") }
        catch [TimeoutException] { continue }
        if ($line -match "^FILE_ERROR $requestId ") { throw $line }
        if ($line -match $Pattern) { return $line }
        if ($line -and $line -notmatch '^FILE_ACK ') { Write-Host $line }
    }
    throw 'Upload/result timed out. Reset the board after an interrupted partial upload.'
}
try {
    $transport.Open(); $transport.DiscardInBuffer()
    $hello=$null
    $timer=[Diagnostics.Stopwatch]::StartNew()
    while ($timer.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        $transport.Write('?')
        try { $line=$transport.ReadLine().TrimEnd("`r") }
        catch [TimeoutException] { continue }
        if ($line -match "^FILE_HELLO $magic ([0-9]+) ([0-9]+) ([0-9]+) 512$") {
            if ([uint32]$Matches[1] -ne $width -or $payload.Length -gt [uint32]$Matches[3] -or
                ($UseCase -ne 'kws_zh' -and [uint32]$Matches[2] -ne $height)) {
                throw 'Input dimensions/capacity do not match target firmware.'
            }
            $hello=$line; break
        }
    }
    if (-not $hello) { throw "No compatible $UseCase file firmware responded." }
    $transport.Write($header,0,$header.Length)
    $null=Read-FileReply "^FILE_READY $requestId$"
    for ($offset=0; $offset -lt $payload.Length;) {
        $count=[Math]::Min(512,$payload.Length-$offset)
        $transport.Write($payload,$offset,$count); $offset+=$count
        $null=Read-FileReply "^FILE_ACK $requestId $offset$"
    }
    Read-FileReply "^FILE_RESULT $requestId [0-9]+$" | Write-Output
} finally {
    $transport.Dispose()
}
