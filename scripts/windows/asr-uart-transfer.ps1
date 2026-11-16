# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# Protocol transaction, accepting a serial-compatible object for host tests. #>
function Invoke-AsrUartTransfer {
    param([object]$Serial, [Mlek.AsrAudio]$Audio, [uint32]$RequestId, [int]$TimeoutSeconds)
    function Read-AsrResponse([string]$Pattern) {
        $timer = [Diagnostics.Stopwatch]::StartNew()
        while ($timer.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
            try { $line = $serial.ReadLine().TrimEnd("`r") }
            catch [TimeoutException] { continue }
            if ($line -match "^ASR_ERROR $requestId (.+)$") { throw "Target rejected request: $($Matches[1])" }
            if ($line -match $Pattern) { return $line }
        }
        throw 'Target response timed out. Inference is not automatically retried; check the connection and UART-mode firmware.'
    }
    $serial.DiscardInBuffer()
    $hello = $null
    $timer = [Diagnostics.Stopwatch]::StartNew()
    while ($timer.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        $serial.Write('?')
        try {
            $line = $serial.ReadLine().TrimEnd("`r")
            if ($line -match '^ASR_HELLO 1 16000 ([0-9]+) 512$') {
                if ($audio.Samples -gt [uint32]$Matches[1]) {
                    throw 'WAV exceeds target capacity; update the firmware or use a shorter file.'
                }
                $hello = $line; break
            }
        } catch [TimeoutException] { }
        Start-Sleep -Milliseconds 100
    }
    if (-not $hello) { throw 'No compatible Conformer UART firmware responded.' }
    $header = $audio.Header($requestId)
    $requestTimer = [Diagnostics.Stopwatch]::StartNew()
    $uploadMs = 0.0
    $serial.Write($header, 0, $header.Length)
    $response = Read-AsrResponse "^ASR_(READY|RESULT) $requestId(?: |$)"
    if ($response -eq "ASR_READY $requestId") {
        for ($offset = 0; $offset -lt $audio.Pcm.Length;) {
            $count = [Math]::Min(512, $audio.Pcm.Length - $offset)
            $serial.Write($audio.Pcm, $offset, $count)
            $offset += $count
            $null = Read-AsrResponse "^ASR_ACK $requestId $offset`$"
        }
        $uploadMs = $requestTimer.Elapsed.TotalMilliseconds
        $response = Read-AsrResponse "^ASR_RESULT $requestId "
    }
    return [pscustomobject]@{
        Reply=$response; UploadMs=$uploadMs; RequestMs=$requestTimer.Elapsed.TotalMilliseconds
    }
}
