# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# Host-only WAV and protocol tests; no COM port or test framework required. #>
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).ProviderPath
. (Join-Path $repoRoot 'scripts/windows/asr-uart-tools.ps1')
. (Join-Path $repoRoot 'scripts/windows/asr-uart-transfer.ps1')
function Assert-True([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}
$testRoot = Join-Path $repoRoot 'build/validation/asr-uart-sender'
New-Item -ItemType Directory -Force -Path $testRoot | Out-Null
function New-TestWav([int]$Rate=16000, [int]$Channels=1, [int]$Bits=16, [int]$Bytes=1030) {
    $stream = [IO.MemoryStream]::new()
    $writer = [IO.BinaryWriter]::new($stream)
    $writer.Write([Text.Encoding]::ASCII.GetBytes('RIFF'))
    $writer.Write([uint32](48 + $Bytes))
    $writer.Write([Text.Encoding]::ASCII.GetBytes('WAVEJUNK'))
    $writer.Write([uint32]3)
    $writer.Write([byte[]]@(1,2,3,0))
    $writer.Write([Text.Encoding]::ASCII.GetBytes('fmt '))
    $writer.Write([uint32]16)
    $writer.Write([uint16]1); $writer.Write([uint16]$Channels)
    $writer.Write([uint32]$Rate); $writer.Write([uint32]($Rate*$Channels*($Bits/8)))
    $writer.Write([uint16]($Channels*($Bits/8))); $writer.Write([uint16]$Bits)
    $writer.Write([Text.Encoding]::ASCII.GetBytes('data'))
    $writer.Write([uint32]$Bytes)
    $pcm = [byte[]]::new($Bytes)
    for ($i=0; $i -lt $Bytes; $i++) { $pcm[$i] = [byte]($i % 256) }
    $writer.Write($pcm); $writer.Flush()
    $path = Join-Path $testRoot "$Rate-$Channels-$Bits-$Bytes.wav"
    [IO.File]::WriteAllBytes($path, $stream.ToArray())
    $writer.Dispose(); $stream.Dispose()
    return $path
}
class AsrFakeSerial {
    [Collections.Generic.Queue[string]]$Replies = [Collections.Generic.Queue[string]]::new()
    [Collections.Generic.List[byte]]$Received = [Collections.Generic.List[byte]]::new()
    [uint32]$RequestId
    [uint32]$Samples
    [uint32]$Crc
    [bool]$RejectCrc = $false
    [bool]$Replay = $false
    [bool]$DropAck = $false
    [uint32]$Capacity = 480000
    [void]DiscardInBuffer() { $this.Replies.Clear() }
    [void]Write([string]$Text) {
        if ($Text -ne '?') { throw 'Unexpected handshake' }
        $this.Replies.Enqueue('startup diagnostic')
        $this.Replies.Enqueue("ASR_HELLO 1 16000 $($this.Capacity) 512")
    }
    [void]Write([byte[]]$Data, [int]$Offset, [int]$Count) {
        if ($Count -eq 16 -and [Text.Encoding]::ASCII.GetString($Data, $Offset, 4) -eq 'ASR1') {
            $this.RequestId = [BitConverter]::ToUInt32($Data,4)
            $this.Samples = [BitConverter]::ToUInt32($Data,8)
            $this.Crc = [BitConverter]::ToUInt32($Data,12)
            if ($this.Replay) {
                $this.Replies.Enqueue("ASR_RESULT $($this.RequestId) 321000 654000 hello world")
            } else { $this.Replies.Enqueue("ASR_READY $($this.RequestId)") }
            return
        }
        if ($Count -gt 512) { throw 'Block is too large' }
        for ($i=$Offset; $i -lt $Offset+$Count; ++$i) { $this.Received.Add($Data[$i]) }
        if ($this.DropAck) { return }
        $this.Replies.Enqueue("ASR_ACK $($this.RequestId) $($this.Received.Count)")
        if ($this.Received.Count -eq $this.Samples*2) {
            if ($this.RejectCrc) { $this.Replies.Enqueue("ASR_ERROR $($this.RequestId) CRC") }
            else { $this.Replies.Enqueue("ASR_RESULT $($this.RequestId) 321000 654000 hello world") }
        }
    }
    [string]ReadLine() {
        if ($this.Replies.Count -eq 0) { throw [TimeoutException]::new() }
        return $this.Replies.Dequeue()
    }
}

Assert-True ([Mlek.AsrAudio]::Crc32([Text.Encoding]::ASCII.GetBytes('123456789')) -eq 3421780262) 'CRC standard vector failed'
$valid = New-TestWav
$audio = [Mlek.AsrAudio]::Load($valid)
Assert-True ($audio.Samples -eq 515 -and $audio.Pcm[13] -eq 13) 'RIFF chunks or raw CR changed'
foreach ($argsSet in @(@{Rate=48000},@{Channels=2},@{Bits=32},@{Bytes=1022},@{Bytes=960002})) {
    $rejected=$false
    try { $null = [Mlek.AsrAudio]::Load((New-TestWav @argsSet)) } catch { $rejected=$true }
    Assert-True $rejected 'Invalid WAV accepted'
}
$longAudio = [Mlek.AsrAudio]::Load((New-TestWav -Bytes 960000))
Assert-True ($longAudio.Samples -eq 480000) '30-second WAV must be accepted'
$oldTarget = [AsrFakeSerial]::new(); $oldTarget.Capacity=112352
$rejected=$false
try { $null = Invoke-AsrUartTransfer $oldTarget $longAudio 43 2 } catch { $rejected=$true }
Assert-True ($rejected -and $oldTarget.Received.Count -eq 0 -and $oldTarget.Samples -eq 0) `
    'Long upload was not rejected before sending to old firmware'
$fake = [AsrFakeSerial]::new()
$result = Invoke-AsrUartTransfer $fake $audio 42 2
Assert-True ($result.Reply -eq 'ASR_RESULT 42 321000 654000 hello world') 'Result did not match request'
Assert-True ($result.UploadMs -gt 0 -and $result.RequestMs -ge $result.UploadMs) 'Transfer timing is inconsistent'
Assert-True ($fake.Received.Count -eq 1030 -and $fake.Received[13] -eq 13) 'Partial last block or PCM corruption'
Assert-True ([Mlek.AsrAudio]::Crc32($fake.Received.ToArray()) -eq $audio.Crc) 'PCM altered in transit'
$fake = [AsrFakeSerial]::new(); $fake.Replay=$true
$result = Invoke-AsrUartTransfer $fake $audio 42 2
Assert-True ($fake.Received.Count -eq 0 -and $result.Reply -match '^ASR_RESULT 42 ' -and
    $result.UploadMs -eq 0) 'Replay uploaded or reran request'
$fake = [AsrFakeSerial]::new(); $fake.RejectCrc=$true
$rejected=$false
try { $null = Invoke-AsrUartTransfer $fake $audio 42 2 } catch { $rejected=$true }
Assert-True $rejected 'CRC error was not reported'
$fake = [AsrFakeSerial]::new(); $fake.DropAck=$true
$rejected=$false
try { $null = Invoke-AsrUartTransfer $fake $audio 42 1 } catch { $rejected=$true }
Assert-True ($rejected -and $fake.Received.Count -eq 512) 'Timeout retried or continued the upload'
Write-Output 'ASR WAV validation, framing, CRC, ACK, replay and timeout tests: PASS'
