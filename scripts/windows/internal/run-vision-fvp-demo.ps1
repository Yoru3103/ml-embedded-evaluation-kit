# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet('gesture', 'coco', 'img-class')][string]$Demo,
    [ValidateSet('VSI', 'Camera', 'Static')][string]$Mode = 'VSI',
    [string]$BuildDirectory,
    [string]$ImagePath,
    [ValidateRange(0, 32)][int]$CameraIndex = 0,
    [string]$Python = 'resources_downloaded/windows/env/Scripts/python.exe',
    [string]$FvpExecutable = $env:FVP_PATH,
    [string]$FvpPythonExtensions,
    [ValidateRange(1024, 65535)][int]$VsiPort = 6000,
    [switch]$CaptureFromPreview,
    [switch]$Fast,
    [switch]$Headless,
    [ValidateRange(0, 86400)][int]$SmokeTimeoutSeconds = 0
)
$ErrorActionPreference = 'Stop'
if ($Mode -eq 'Camera' -and $Demo -ne 'img-class') {
    throw 'Semihosted Camera mode is available for img-class. Select VSI for YOLO camera input.'
}
if ($Mode -eq 'Static' -and $ImagePath) {
    throw 'For Static mode, select the image with -InputPath when building the firmware.'
}
if ($CaptureFromPreview -and ($Mode -ne 'VSI' -or $Headless -or $ImagePath)) {
    throw '-CaptureFromPreview requires VSI mode with a live camera preview.'
}
if (-not $BuildDirectory) {
    $variant = if ($CaptureFromPreview) { 'vsi-auto' } else { $Mode.ToLowerInvariant() }
    if ($Fast) { $variant += '-fast' }
    $BuildDirectory = "build/fvp-$Demo/$variant"
}
. (Join-Path $PSScriptRoot 'fvp-tools.ps1')

if ($Mode -eq 'Camera') {
    $options = @{
        Demo='img-class'; BuildDirectory=$BuildDirectory; Python=$Python
        FvpExecutable=(Resolve-WindowsFvpExecutable -Executable $FvpExecutable); Headless=$Headless; SmokeTimeoutSeconds=$SmokeTimeoutSeconds
        Fast=$Fast
    }
    if ($ImagePath) { $options.ImagePath = $ImagePath }
    else {
        $options.Camera = $true
        $options.CameraIndex = $CameraIndex
    }
    & (Join-Path $PSScriptRoot 'run-qrcode-fvp-demo.ps1') @options
    return
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).ProviderPath
. (Join-Path $PSScriptRoot 'fvp-audio-process-tools.ps1')

function Resolve-VisionPath {
    param([string]$Path)
    if (-not [IO.Path]::IsPathRooted($Path)) { $Path = Join-Path $repoRoot $Path }
    return (Resolve-Path -LiteralPath $Path).ProviderPath
}

$BuildDirectory = Resolve-VisionPath $BuildDirectory
$useCase = if ($Demo -eq 'img-class') { 'img_class' } else { 'yolov8_detection' }
$firmware = Resolve-VisionPath (Join-Path $BuildDirectory "bin/mlek_$useCase.axf")
$vsi = if ($Mode -eq 'VSI') { 'ON' } else { 'OFF' }
$cache = Get-Content -LiteralPath (Join-Path $BuildDirectory 'CMakeCache.txt')
foreach ($setting in @('TARGET_PLATFORM=mps4', 'TARGET_SUBSYSTEM=sse-320',
        'FPGA_PLATFORM_SSE_320=OFF', 'MPS4_DISPLAY_ENABLED=ON', 'SEMIHOSTING_ENABLED=OFF',
        'ETHOS_U_NPU_CONFIG_ID=Z1024', "FVP_VSI_ENABLED=$vsi")) {
    $key, $value = $setting.Split('=')
    $expected = if ($value -eq 'OFF') { '(OFF|FALSE|0)' } else { [regex]::Escape($value) }
    if (-not ($cache -match "^${key}:[^=]+=$expected`$")) { throw "Expected firmware setting: $setting" }
}
$button = $Mode -eq 'VSI' -and ($cache -match "^${useCase}_BUTTON_CAPTURE:[^=]+=ON$")
$fastFirmware = $cache -match '^ETHOS_U_NPU_TIMING_ADAPTER_ENABLED:[^=]+=(OFF|FALSE|0)$'
if ($Fast -and -not $fastFirmware) {
    throw '-Fast requires firmware built with the timing adapter disabled.'
}
if ($CaptureFromPreview -and $button) {
    throw '-CaptureFromPreview requires a VSI firmware built with -AutoCapture.'
}
if ($button -and -not $PSBoundParameters.ContainsKey('FvpExecutable') -and -not $env:FVP_PATH) {
    $preparedFvp = Join-Path $repoRoot `
        'resources_downloaded/windows/corstone-320-key0/models/Win64_VC2019/FVP_Corstone_SSE-320.exe'
    if (Test-Path -LiteralPath $preparedFvp -PathType Leaf) { $FvpExecutable = $preparedFvp }
}
$FvpExecutable = Resolve-WindowsFvpExecutable -Executable $FvpExecutable

$fvpArguments = @('-a', $firmware, '-C', 'mps4_board.subsystem.ethosu.num_macs=1024',
    '-C', 'mps4_board.subsystem.cpu0.semihosting-enable=0',
    '-C', "mps4_board.visualisation.disable-visualisation=$(if ($button) {0} else {1})",
    '-C', "vis_hdlcd.disable_visualisation=$(if ($Headless) {1} else {0})",
    '-C', 'mps4_board.uart0.out_file=-', '-C', 'mps4_board.uart0.unbuffered_output=1',
    '-C', 'mps4_board.uart0.shutdown_on_eot=1')
foreach ($uart in @(0, 1, 2, 5)) {
    $fvpArguments += @('-C', "mps4_board.telnetterminal$uart.start_telnet=0")
}
if ($fastFirmware) {
    $fvpArguments += @('-C', 'mps4_board.subsystem.ethosu.extra_args=--fast')
}
if ($button) {
    $fvpArguments += @('-C', "mps4_board.visualisation.window_title=MLEK $Demo - KEY0")
}
$savedEnvironment = @{}
foreach ($name in @('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'PATH', 'VSI_VIDEO_HOST',
        'VSI_VIDEO_PORT',
        'SDL_MOUSE_FOCUS_CLICKTHROUGH')) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
$server = $null
$fvp = $null
$reader = $null
$lock = $null
try {
    $lock = [IO.File]::Open((Join-Path $BuildDirectory 'demo.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    [Mlek.FvpAudioConsole]::Start()
    $env:PYTHONHOME = $null
    $env:PYTHONPATH = $null
    $env:VIRTUAL_ENV = $null
    if ($Mode -eq 'VSI') {
        $Python = Resolve-VisionPath $Python
        $vsiPath = Resolve-VisionPath 'scripts/vsi/python'
        $pythonHome = Resolve-VisionPath (Join-Path (Split-Path $FvpExecutable) '../../python')
        if (-not $FvpPythonExtensions) {
            $FvpPythonExtensions = Join-Path $env:LOCALAPPDATA 'Programs/Python/Python39/DLLs'
        }
        $FvpPythonExtensions = Resolve-VisionPath $FvpPythonExtensions
        foreach ($extension in @('_socket.pyd', '_multiprocessing.pyd', 'select.pyd')) {
            if (-not (Test-Path -LiteralPath (Join-Path $FvpPythonExtensions $extension))) {
                throw "FVP needs Python 3.9 extensions. Missing $extension in $FvpPythonExtensions"
            }
        }
        $hostArguments = @((Join-Path $vsiPath 'vsi_video_server.py'),
            '--camera-index', "$CameraIndex", '--port', "$VsiPort")
        if ($ImagePath) { $hostArguments += @('--image', (Resolve-VisionPath $ImagePath)) }
        elseif (-not $Headless) { $hostArguments += '--preview' }
        if ($CaptureFromPreview) { $hostArguments += '--capture-on-space' }
        $hostLog = Join-Path $BuildDirectory 'vsi-video.log'
        $quotedHost = $hostArguments | ForEach-Object { '"' + $_ + '"' }
        $server = Start-Process -FilePath $Python -ArgumentList $quotedHost -PassThru `
            -NoNewWindow -WorkingDirectory $repoRoot -RedirectStandardOutput $hostLog `
            -RedirectStandardError (Join-Path $BuildDirectory 'vsi-video.err.log')
        $null = $server.Handle
        $deadline = [DateTime]::UtcNow.AddSeconds(10)
        while (-not (Select-String -LiteralPath $hostLog -Pattern '^VSI video listening ' -Quiet)) {
            if (Test-FvpAudioStopRequested) { return }
            if ($server.HasExited -or [DateTime]::UtcNow -ge $deadline) {
                throw "VSI video server failed to start. Inspect $BuildDirectory/vsi-video.err.log"
            }
            Start-Sleep -Milliseconds 100
        }
        $env:VSI_VIDEO_HOST = '127.0.0.1'
        $env:VSI_VIDEO_PORT = "$VsiPort"
        $env:PYTHONHOME = $pythonHome
        $env:PYTHONPATH = "$FvpPythonExtensions;$(Join-Path $pythonHome 'Lib')"
        $env:PATH = "$FvpPythonExtensions;$pythonHome;$env:PATH"
        $fvpArguments += @('-C', "mps4_board.v_path=$vsiPath")
        if ($button) {
            Write-Host "Wait for the firmware prompt, then click KEY0 in 'MLEK $Demo - KEY0'."
        } elseif ($CaptureFromPreview) {
            Write-Host 'Press Space in the VSI camera preview to capture and infer one frame.'
        }
    }
    $log = Join-Path $BuildDirectory "$Demo-fvp.log"
    $quotedFvp = $fvpArguments | ForEach-Object { '"' + $_ + '"' }
    if ($button) {
        $env:SDL_MOUSE_FOCUS_CLICKTHROUGH = '1'
    }
    # The FVP board panel uses BMP assets distributed beside its executable.
    $fvp = Start-Process -FilePath $FvpExecutable -ArgumentList $quotedFvp -PassThru `
        -NoNewWindow -WorkingDirectory (Split-Path -Parent $FvpExecutable) -RedirectStandardOutput $log `
        -RedirectStandardError (Join-Path $BuildDirectory "$Demo-fvp.err.log")
    $null = $fvp.Handle
    $stream = [IO.File]::Open($log, 'Open', 'Read', 'ReadWrite')
    $reader = [IO.StreamReader]::new($stream, [Text.Encoding]::UTF8)
    $deadline = [DateTime]::UtcNow.AddSeconds($SmokeTimeoutSeconds)
    $tail = ''
    Write-Host 'Press Q or Ctrl+C in this terminal to stop.'
    while (-not (Test-FvpAudioStopRequested)) {
        $chunk = $reader.ReadToEnd()
        if ($chunk) {
            Write-Host -NoNewline $chunk
            $tail += $chunk
            if ($tail.Length -gt 8192) { $tail = $tail.Substring($tail.Length - 8192) }
        }
        if ($tail -match 'terminated with failure|Camera error|Failed to (initialise|configure|start) camera|Failed to initialise (model|platform)|Inference failed|Post-processing failed') {
            throw "Firmware failed. Inspect $log"
        }
        if ($SmokeTimeoutSeconds -and $tail -match 'Final results:|Detections: \d+' -and
            $tail -match 'DURATION:') {
            break
        }
        if ($fvp.HasExited) {
            if ($fvp.ExitCode -or $SmokeTimeoutSeconds) {
                throw "FVP exited with code $($fvp.ExitCode). Inspect $log and $Demo-fvp.err.log"
            }
            break
        }
        if ($server -and $server.HasExited) {
            if ($server.ExitCode -or $SmokeTimeoutSeconds) {
                throw "VSI video server exited. Inspect $BuildDirectory/vsi-video.err.log"
            }
            break
        }
        if ($SmokeTimeoutSeconds -and [DateTime]::UtcNow -ge $deadline) {
            throw "Timed out waiting for inference. Inspect $log"
        }
        Start-Sleep -Milliseconds 100
    }
} finally {
    if ($reader) { $reader.Dispose() }
    Stop-FvpAudioProcess $fvp
    Stop-FvpAudioProcess $server
    if ($lock) { $lock.Dispose() }
    [Mlek.FvpAudioConsole]::Stop()
    foreach ($name in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
    }
}
