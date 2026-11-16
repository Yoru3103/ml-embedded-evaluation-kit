# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# Run the host server and custom VSI0 pull-DMA audio demo. Ctrl+C stops both. #>
[CmdletBinding(DefaultParameterSetName='Wav')]
param(
    [ValidateSet('kws_zh', 'asr', 'asr_zh')][string]$UseCase = 'kws_zh',
    [Parameter(Mandatory=$true, ParameterSetName='Wav')][string]$Wav,
    [Parameter(Mandatory=$true, ParameterSetName='Microphone')][switch]$Microphone,
    [Parameter(ParameterSetName='Microphone')][int]$Device,
    [Parameter(ParameterSetName='Microphone')][ValidateRange(-90, -1)][double]$VadDbfs,
    [Parameter(ParameterSetName='Microphone')][switch]$LevelMeter,
    [Parameter(ParameterSetName='Microphone')][switch]$AutoVad,
    [Parameter(ParameterSetName='Wav')][switch]$Realtime,
    [ValidateRange(1, 65535)][int]$Port = 6010,
    [ValidateRange(20, 60000)][int]$QueueMs,
    [ValidateRange(0.01, 86400)][double]$Duration,
    [string]$BuildDirectory,
    [string]$Python,
    [string]$FvpPythonHome,
    [string]$FvpPythonExtensions,
    [switch]$Headless,
    [ValidateRange(0, 86400)][int]$SmokeTimeoutSeconds = 0,
    [string]$FvpExecutable = 'C:/Program Files/Arm/FVP_Corstone_SSE-320/models/Win64_VC2019/FVP_Corstone_SSE-320.exe'
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).ProviderPath
. (Join-Path $PSScriptRoot 'fvp-audio-process-tools.ps1')
if (-not $PSBoundParameters.ContainsKey('QueueMs')) {
    $QueueMs = if ($UseCase -eq 'kws_zh') { 5000 } else { 30000 }
}
function Resolve-DemoPath {
    param([string]$Path)
    if (-not [IO.Path]::IsPathRooted($Path)) { $Path = Join-Path $repoRoot $Path }
    return (Resolve-Path -LiteralPath $Path).ProviderPath
}
if (-not $BuildDirectory) { $BuildDirectory = "build/fvp-audio/$UseCase" }
$BuildDirectory = Resolve-DemoPath $BuildDirectory
$cacheBytes = if ($UseCase -eq 'asr_zh') {786432} else {262144}
$cache = Get-Content -LiteralPath (Join-Path $BuildDirectory 'CMakeCache.txt')
foreach ($setting in @("${UseCase}_INPUT_MODE=VSI", 'TARGET_PLATFORM=mps4',
        'TARGET_SUBSYSTEM=sse-320', 'ETHOS_U_NPU_ID=U85', 'ETHOS_U_NPU_CONFIG_ID=Z1024',
        'ETHOS_U_NPU_MEMORY_MODE=Dedicated_Sram', "ETHOS_U_NPU_CACHE_SIZE=$cacheBytes",
        'FPGA_PLATFORM_SSE_320=OFF', 'SEMIHOSTING_ENABLED=OFF', 'MPS4_DISPLAY_ENABLED=ON',
        'ETHOS_U_NPU_TIMING_ADAPTER_ENABLED=OFF')) {
    $key, $value = $setting.Split('=')
    if (-not ($cache -match "^${key}:[^=]+=$value`$")) {
        throw "Expected audio demo build setting: $setting"
    }
}
$firmware = Resolve-DemoPath (Join-Path $BuildDirectory "bin/mlek_$UseCase.axf")
$FvpExecutable = Resolve-DemoPath $FvpExecutable
if (-not $FvpPythonHome) {
    $FvpPythonHome = Join-Path (Split-Path $FvpExecutable) '../../python'
}
$FvpPythonHome = Resolve-DemoPath $FvpPythonHome
if (-not (Test-Path -LiteralPath (Join-Path $FvpPythonHome 'Lib/encodings'))) {
    throw "FVP Python standard library missing: $FvpPythonHome/Lib/encodings"
}
if (-not $FvpPythonExtensions) {
    foreach ($candidate in @((Join-Path $env:LOCALAPPDATA 'Programs/Python/Python39/DLLs'),
            (Join-Path $repoRoot 'resources_downloaded/fvp-python39'))) {
        if (Test-Path -LiteralPath (Join-Path $candidate '_socket.pyd')) {
            $FvpPythonExtensions = $candidate
            break
        }
    }
}
if (-not $FvpPythonExtensions) {
    throw 'Supply -FvpPythonExtensions pointing to x64 CPython 3.9 extensions including _socket.pyd.'
}
$FvpPythonExtensions = Resolve-DemoPath $FvpPythonExtensions
if (-not $Python) { $Python = 'resources_downloaded/windows-executorch/env/Scripts/python.exe' }
$Python = Resolve-DemoPath $Python
$hostScript = Resolve-DemoPath 'scripts/py/fvp_audio_demo.py'
$bridge = Resolve-DemoPath 'scripts/py/fvp_audio_demo_vsi'
$hostArguments = @($hostScript, '--port', "$Port", '--queue-ms', "$QueueMs")
if ($Microphone) {
    $hostArguments += '--microphone'
    if ($AutoVad -and $UseCase -ne 'asr_zh') { throw '-AutoVad applies only to Chinese ASR.' }
    if ($UseCase -eq 'asr_zh') {
        $hostArguments += $(if ($AutoVad) { '--auto-utterances' } else { '--record-utterance' })
    }
    if ($UseCase -eq 'asr' -and $PSBoundParameters.ContainsKey('VadDbfs')) {
        $hostArguments += @('--vad-dbfs', $VadDbfs.ToString([Globalization.CultureInfo]::InvariantCulture))
    } elseif ($PSBoundParameters.ContainsKey('VadDbfs')) {
        throw '-VadDbfs applies only to ASR microphone capture.'
    }
    if ($LevelMeter) { $hostArguments += '--level-meter' }
    if ($PSBoundParameters.ContainsKey('Device')) { $hostArguments += @('--device', "$Device") }
} else { $hostArguments += @('--wav', (Resolve-DemoPath $Wav)) }
if ($Realtime) { $hostArguments += '--realtime' }
if ($PSBoundParameters.ContainsKey('Duration')) {
    $hostArguments += @('--duration', $Duration.ToString([Globalization.CultureInfo]::InvariantCulture))
}
$fvpArguments = @('-a', $firmware, '-C', "mps4_board.v_path=$bridge",
    '-C', 'mps4_board.subsystem.ethosu.num_macs=1024',
    '-C', 'mps4_board.subsystem.ethosu.extra_args=--fast',
    '-C', 'mps4_board.subsystem.cpu0.semihosting-enable=0',
    '-C', 'mps4_board.visualisation.disable-visualisation=1',
    '-C', "vis_hdlcd.disable_visualisation=$(if ($Headless) {1} else {0})",
    '-C', 'mps4_board.uart0.out_file=-', '-C', 'mps4_board.uart0.unbuffered_output=1',
    '-C', 'mps4_board.uart0.shutdown_on_eot=0')
foreach ($uart in @(0, 1, 2, 5)) {
    $fvpArguments += @('-C', "mps4_board.telnetterminal$uart.start_telnet=0")
}
$savedEnvironment = @{}
foreach ($name in @('FVP_AUDIO_DEMO_PORT', 'PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'PATH')) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
$server = $null
$fvp = $null
$reader = $null
Push-Location $repoRoot
try {
    [Mlek.FvpAudioConsole]::Start()
    $env:FVP_AUDIO_DEMO_PORT = "$Port"
    $env:PYTHONHOME = $null
    $env:PYTHONPATH = $null
    $env:VIRTUAL_ENV = $null
    # Start-Process joins arguments on Windows; quote paths containing spaces.
    $quoted = $hostArguments | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' }
    $server = Start-Process -FilePath $Python -ArgumentList $quoted -PassThru `
        -WindowStyle Hidden -WorkingDirectory $repoRoot `
        -RedirectStandardOutput (Join-Path $BuildDirectory 'audio-host.log') `
        -RedirectStandardError (Join-Path $BuildDirectory 'audio-host.err.log')
    # Retain the native handle before exit so Windows PowerShell can read ExitCode.
    $null = $server.Handle
    $hostLog = Join-Path $BuildDirectory 'audio-host.log'
    $deadline = [DateTime]::UtcNow.AddSeconds(10)
    while ($true) {
        if (Test-FvpAudioStopRequested) { return }
        $server.Refresh()
        if ($server.HasExited) {
            throw "Audio host exited; see $BuildDirectory/audio-host.err.log"
        }
        if ((Test-Path -LiteralPath $hostLog) -and
            (Select-String -LiteralPath $hostLog -Pattern '^Audio listening ' -Quiet)) { break }
        if ([DateTime]::UtcNow -ge $deadline) {
            throw "Audio host was not ready within 10 seconds; see $hostLog"
        }
        Start-Sleep -Milliseconds 100
    }
    # The host uses its own venv; only the FVP receives its bundled Python 3.9 home.
    $env:PYTHONHOME = $FvpPythonHome
    $env:PYTHONPATH = "$FvpPythonExtensions;$(Join-Path $FvpPythonHome 'Lib')"
    $env:PATH = "$FvpPythonExtensions;$FvpPythonHome;$env:PATH"
    Write-Host "Custom VSI0 audio adapter on port $Port; host logs in $BuildDirectory."
    Write-Host 'Functional fast mode: FVP times are not FPGA performance measurements.'
    $fvpLog = Join-Path $BuildDirectory 'audio-fvp.log'
    $quotedFvp = $fvpArguments | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' }
    $fvp = Start-Process -FilePath $FvpExecutable -ArgumentList $quotedFvp -PassThru `
        -WindowStyle Hidden -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $fvpLog `
        -RedirectStandardError (Join-Path $BuildDirectory 'audio-fvp.err.log')
    $null = $fvp.Handle
    $stream = [IO.File]::Open($fvpLog, [IO.FileMode]::Open, [IO.FileAccess]::Read,
        [IO.FileShare]::ReadWrite)
    $reader = [IO.StreamReader]::new($stream, [Text.Encoding]::UTF8)
    $deadline = [DateTime]::UtcNow.AddSeconds($SmokeTimeoutSeconds)
    $outputTail = ''
    $completed = $false
    $lastMeter = ''
    $nextMeter = [DateTime]::UtcNow
    Write-Host 'Press Ctrl+C or Q in this terminal to stop the demo.'
    while (-not (Test-FvpAudioStopRequested)) {
        if ($LevelMeter -and [DateTime]::UtcNow -ge $nextMeter) {
            $meterLine = Get-Content -LiteralPath (Join-Path $BuildDirectory 'audio-host.log') `
                -Tail 1 -ErrorAction SilentlyContinue
            if ($meterLine -and $meterLine -ne $lastMeter -and $meterLine -like 'MIC RMS=*') {
                Write-Host $meterLine
                $lastMeter = $meterLine
            }
            $nextMeter = [DateTime]::UtcNow.AddSeconds(1)
        }
        $chunk = $reader.ReadToEnd()
        if ($chunk) {
            Write-Host -NoNewline $chunk
            $outputTail += $chunk
            if ($outputTail.Length -gt 8192) {
                $outputTail = $outputTail.Substring($outputTail.Length - 8192)
            }
        }
        $fvp.Refresh()
        $server.Refresh()
        if (-not $completed -and
            $outputTail -match 'AUDIO_EOS sample=\d+|ASR_VSI_EOS \d+ success=1|ASR_ZH_VSI_EOS success=1') {
            $completed = $true
            if ($SmokeTimeoutSeconds -eq 0) {
                Write-Host "`nAudio complete. LCD result remains visible; press Q or Ctrl+C to exit."
            }
        }
        if ($outputTail -match 'ASR_VSI_ERROR|ASR_ZH_VSI_ERROR|ASR_ZH completed success=0|VSI audio input failed|VSI audio bridge missing') {
            throw "FVP audio error; see $fvpLog"
        }
        if ($server.HasExited -and $server.ExitCode -ne 0) {
            throw "Audio host failed; see $BuildDirectory/audio-host.err.log"
        }
        if ($SmokeTimeoutSeconds -gt 0) {
            if ($completed) { break }
            if ($fvp.HasExited) { throw "FVP exited before audio EOS; see $fvpLog" }
            if ([DateTime]::UtcNow -ge $deadline) {
                throw "FVP audio EOS timeout after $SmokeTimeoutSeconds seconds; see $fvpLog"
            }
        } elseif ($fvp.HasExited) {
            if ($fvp.ExitCode -ne 0) { throw "FVP exited with code $($fvp.ExitCode)." }
            break
        }
        Start-Sleep -Milliseconds 100
    }
} finally {
    if ($reader) { $reader.Dispose() }
    Stop-FvpAudioProcess $server
    Stop-FvpAudioProcess $fvp
    [Mlek.FvpAudioConsole]::Stop()
    foreach ($name in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
    }
    Pop-Location
}
