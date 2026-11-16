# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Port,
    [ValidateRange(5, 3600)][int]$TimeoutSeconds = 120,
    [switch]$Measure
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../../../../..')).ProviderPath
$python = Join-Path $root 'resources_downloaded/windows-executorch/env/Scripts/python.exe'
$tokenizer = Join-Path $root 'resources_downloaded/minimind2_small/hf_tokenizer'
$encoder = Join-Path $PSScriptRoot 'fvp_console.py'
$OutputEncoding = [Text.UTF8Encoding]::new($false)
$previousConsoleEncoding = [Console]::OutputEncoding
$serial = [IO.Ports.SerialPort]::new($Port, 115200, [IO.Ports.Parity]::None,
    8, [IO.Ports.StopBits]::One)
$serial.Handshake = [IO.Ports.Handshake]::None
$serial.DtrEnable = $false
$serial.RtsEnable = $false
$serial.NewLine = "`n"
$serial.WriteTimeout = 5000

function Read-Frame([string]$Pattern, [int]$Seconds) {
    $watch = [Diagnostics.Stopwatch]::StartNew()
    while ($watch.Elapsed.TotalSeconds -lt $Seconds) {
        $serial.ReadTimeout = [Math]::Max(1, [int]($Seconds * 1000 - $watch.ElapsedMilliseconds))
        $line = $serial.ReadLine().TrimEnd("`r")
        if ($line -match $Pattern) { return $line }
    }
    throw "Timed out on $Port waiting for MiniMind firmware."
}

function Connect-MiniMind {
    $pattern = '^MINIMIND_DEMO HELLO context=(?<context>\d+) max_prompt=(?<maxPrompt>\d+)$'
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $hello = $null
    while ($watch.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        $serial.Write('?')
        try {
            $hello = Read-Frame $pattern 2
            break
        } catch [TimeoutException] {
            # Retry only the single-byte probe while the board finishes its previous request.
        }
    }
    if ($null -eq $hello) { throw [TimeoutException]::new('No MiniMind HELLO received.') }
    $frame = [regex]::Match($hello, $pattern)
    $contextLength = [int]$frame.Groups['context'].Value
    $maxPrompt = [int]$frame.Groups['maxPrompt'].Value
    if ($contextLength -notin @(64, 128) -or $maxPrompt -ne ($contextLength - 1)) {
        throw "Unsupported MiniMind HELLO frame on ${Port}: $hello"
    }
    return $contextLength
}

try {
    [Console]::OutputEncoding = $OutputEncoding
    $stage = 'firmware handshake'
    $serial.Open()
    $serial.DiscardInBuffer()
    Write-Host "Connecting to MiniMind on $Port (115200 8N1)"
    $contextLength = Connect-MiniMind
    while ($true) {
        $prompt = Read-Host 'You >'
        if ($prompt -eq '/quit') { break }
        if ([string]::IsNullOrWhiteSpace($prompt)) { continue }
        $tokens = $prompt | & $python -B $encoder --tokenizer $tokenizer --encode-stdin `
            --context-length $contextLength
        if ($LASTEXITCODE -ne 0 -or $tokens -notmatch '^\d+( \d+)*$') {
            throw 'MiniMind prompt encoding failed.'
        }
        $stage = 'prompt acknowledgement'
        $requestWatch = [Diagnostics.Stopwatch]::StartNew()
        $acceptMs = $null
        $firstTextMs = $null
        $boardTiming = $null
        $profileLines = @()
        $serial.WriteLine($tokens)
        $promptCount = ($tokens -split ' ').Count
        Write-Host "Sent $promptCount prompt tokens (context $contextLength). Waiting for reply..."
        $decoder = [Text.Encoding]::UTF8.GetDecoder()
        while ($true) {
            $header = Read-Frame '^(MINIMIND_DEMO (ACCEPT|START|CHUNK|TIMING|DONE|ERROR)|MINIMIND_(BREAKDOWN|PROFILE|QDQ|CLAMP)) ' $TimeoutSeconds
            if ($header -match '^MINIMIND_(BREAKDOWN|PROFILE|QDQ|CLAMP) ') {
                if ($Measure) { $profileLines += $header }
                continue
            }
            if ($header.StartsWith('MINIMIND_DEMO TIMING ')) {
                $boardTiming = $header.Substring('MINIMIND_DEMO TIMING '.Length)
                continue
            }
            if ($header -match '^MINIMIND_DEMO ACCEPT prompt_tokens=(\d+)$') {
                if ([int]$Matches[1] -ne $promptCount) {
                    throw "Board accepted $($Matches[1]) prompt tokens, expected $promptCount."
                }
                $stage = 'prompt processing'
                $acceptMs = $requestWatch.Elapsed.TotalMilliseconds
                Write-Host 'Board accepted prompt. Processing...'
                continue
            }
            if ($header -match '^MINIMIND_DEMO START ') {
                $stage = 'streaming reply'
                Write-Host 'MiniMind > ' -NoNewline
                continue
            }
            if ($header -match '^MINIMIND_DEMO CHUNK (\d+) (\d+)$') {
                $length = [int]$Matches[2]
                if ($length -gt 4096) { throw "Invalid token payload length: $length" }
                $payload = [byte[]]::new($length)
                $serial.ReadTimeout = 5000
                for ($offset = 0; $offset -lt $length;) {
                    $offset += $serial.Read($payload, $offset, $length - $offset)
                }
                $chars = [char[]]::new($length + 2)
                $charCount = $decoder.GetChars($payload, 0, $length, $chars, 0, $false)
                if ($charCount -gt 0) {
                    if ($null -eq $firstTextMs) {
                        $firstTextMs = $requestWatch.Elapsed.TotalMilliseconds
                    }
                    [Console]::Write([string]::new($chars, 0, $charCount))
                }
                continue
            }
            if ($header -match '^MINIMIND_DEMO DONE tokens=(\d+) reason=(\w+)$') {
                $replyTokens = [int]$Matches[1]
                $totalMs = $requestWatch.Elapsed.TotalMilliseconds
                $chars = [char[]]::new(4)
                $charCount = $decoder.GetChars([byte[]]::new(0), 0, 0, $chars, 0, $true)
                if ($charCount -gt 0) { [Console]::Write([string]::new($chars, 0, $charCount)) }
                Write-Host "`n[$($Matches[2])]"
                if ($Measure) {
                    $profileLines | ForEach-Object { Write-Host $_ }
                    if ($null -ne $boardTiming) { Write-Host "[board] $boardTiming" }
                    $first = if ($null -eq $firstTextMs) { 'n/a' } else { '{0:F1}' -f $firstTextMs }
                    Write-Host ('[host] accept_ms={0:F1} first_text_ms={1} total_ms={2:F1} reply_tokens={3}' -f
                        $acceptMs, $first, $totalMs, $replyTokens)
                }
                break
            }
            if ($header -match '^MINIMIND_DEMO ERROR ') {
                Write-Host $header
                break
            }
        }
        $stage = 'READY after reply'
        $null = Read-Frame '^MINIMIND_DEMO READY context=\d+ max_prompt=\d+$' 15
    }
} catch [TimeoutException] {
    throw "MiniMind serial timeout on $Port during $stage. Check the board UART log."
} finally {
    $serial.Dispose()
    [Console]::OutputEncoding = $previousConsoleEncoding
}
