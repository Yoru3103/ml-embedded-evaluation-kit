# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# Exercise raw bytes, partial-line timeouts and disconnects over a real loopback socket. #>
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).ProviderPath
. (Join-Path $repoRoot 'scripts/windows/asr-tcp-transport.ps1')
$testRoot = Join-Path $repoRoot 'build/validation/asr-uart-sender'
New-Item -ItemType Directory -Force -Path $testRoot | Out-Null
$ready = Join-Path $testRoot "tcp-$([guid]::NewGuid()).port"
$job = Start-Job -ArgumentList $ready -ScriptBlock {
    param($Ready)
    $ErrorActionPreference = 'Stop'
    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
    $peer = $null
    try {
        $listener.Start()
        [IO.File]::WriteAllText($Ready, [string]$listener.LocalEndpoint.Port)
        $peer = $listener.AcceptTcpClient()
        $stream = $peer.GetStream()
        $stream.ReadTimeout = 10000
        for ($i=0; $i -lt 256; ++$i) {
            if ($stream.ReadByte() -ne $i) { throw "Raw byte $i changed" }
        }
        $part = [Text.Encoding]::ASCII.GetBytes('ASR_')
        $stream.Write($part, 0, $part.Length)
        Start-Sleep -Milliseconds 1200
        $part = [Text.Encoding]::ASCII.GetBytes("HELLO 1 16000 112352 512`n")
        $stream.Write($part, 0, $part.Length)
        Start-Sleep -Milliseconds 300
        'SERVER_PASS'
    } finally {
        if ($peer) { $peer.Dispose() }
        $listener.Stop()
    }
}
$transport = $null
try {
    $timer = [Diagnostics.Stopwatch]::StartNew()
    while (-not (Test-Path -LiteralPath $ready)) {
        if ($timer.Elapsed.TotalSeconds -gt 20 -or $job.State -eq 'Failed') {
            throw 'Loopback server did not start.'
        }
        Start-Sleep -Milliseconds 50
    }
    $portNumber = [int][IO.File]::ReadAllText($ready)
    $transport = [Mlek.AsrTcpTransport]::new('127.0.0.1', $portNumber, 5000)
    $transport.Open()
    $transport.DiscardInBuffer()
    $bytes = [byte[]](0..255)
    $transport.Write($bytes, 0, $bytes.Length)
    $timedOut = $false
    $line = ''
    $timer.Restart()
    while (-not $line -and $timer.Elapsed.TotalSeconds -lt 10) {
        try { $line = $transport.ReadLine() }
        catch [TimeoutException] { $timedOut = $true }
    }
    if (-not $timedOut -or $line -ne 'ASR_HELLO 1 16000 112352 512') {
        throw 'Partial line was lost across a read timeout.'
    }
    $closed = $false
    try { $null = $transport.ReadLine() } catch [IO.EndOfStreamException] { $closed = $true }
    if (-not $closed) { throw 'Peer disconnect was not reported.' }
    $null = Wait-Job $job -Timeout 10
    $serverResult = Receive-Job $job -ErrorAction Stop
    if ($serverResult -ne 'SERVER_PASS') { throw 'Server verification failed.' }
    Write-Output 'ASR TCP raw bytes, partial-line timeout and disconnect tests: PASS'
} finally {
    if ($transport) { $transport.Dispose() }
    Stop-Job $job
    Remove-Job $job
    Remove-Item -LiteralPath $ready -ErrorAction SilentlyContinue
}
