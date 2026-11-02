# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# Start the resident detector; result remains visible until another image is sent. #>
[CmdletBinding(DefaultParameterSetName='Image')]
param(
    [Parameter(Mandatory=$true, ParameterSetName='Image')][string]$ImagePath,
    [Parameter(Mandatory=$true, ParameterSetName='Camera')][switch]$Camera,
    [Parameter(ParameterSetName='Camera')][ValidateRange(0, 32)][int]$CameraIndex = 0,
    [Parameter(ParameterSetName='Camera')][ValidateSet('auto', 'dshow', 'msmf')]
    [string]$CameraBackend = 'dshow',
    [ValidateSet('qrcode', 'hand-pose')][string]$Demo = 'qrcode',
    [ValidateSet('square', 'letterbox')][string]$CaptureFraming = 'letterbox',
    [string]$BuildDirectory = 'build/fvp-qrcode/dedicated-384k',
    [string]$Python = 'resources_downloaded/windows-executorch/env/Scripts/python.exe',
    [string]$FvpExecutable = 'C:/Program Files/Arm/FVP_Corstone_SSE-320/models/Win64_VC2019/FVP_Corstone_SSE-320.exe',
    [switch]$Headless,
    [ValidateRange(0, 86400)][int]$SmokeTimeoutSeconds = 0
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).ProviderPath
. (Join-Path $PSScriptRoot 'fvp-audio-process-tools.ps1')
function Resolve-QrRunPath {
    param([string]$Path)
    if (-not [IO.Path]::IsPathRooted($Path)) { $Path = Join-Path $repoRoot $Path }
    return (Resolve-Path -LiteralPath $Path).ProviderPath
}
$BuildDirectory = Resolve-QrRunPath $BuildDirectory
$FvpExecutable = Resolve-QrRunPath $FvpExecutable
$useCase = if ($Demo -eq 'hand-pose') { 'yolov8_pose' } else { 'qrcode' }
$marker = if ($Demo -eq 'hand-pose') { 'HAND_POSE' } else { 'QRCODE' }
$cache = Get-Content -LiteralPath (Join-Path $BuildDirectory 'CMakeCache.txt')
foreach ($setting in @("${useCase}_INPUT_MODE=SEMIHOST", 'SEMIHOSTING_ENABLED=ON',
        'TARGET_PLATFORM=mps4', 'TARGET_SUBSYSTEM=sse-320', 'FPGA_PLATFORM_SSE_320=OFF',
        'MPS4_DISPLAY_ENABLED=ON', 'ETHOS_U_NPU_ID=U85', 'ETHOS_U_NPU_CONFIG_ID=Z1024',
        'ETHOS_U_NPU_TIMING_ADAPTER_ENABLED=OFF')) {
    $key, $value = $setting.Split('=')
    if (-not ($cache -match "^${key}:[^=]+=$value`$")) { throw "Expected $Demo setting: $setting" }
}
$firmware = Resolve-QrRunPath (Join-Path $BuildDirectory "bin/mlek_$useCase.axf")
if ($Camera -and $SmokeTimeoutSeconds -gt 0) {
    throw '-SmokeTimeoutSeconds applies only to image-file mode.'
}
$Python = Resolve-QrRunPath $Python
$lock = $null
$fvp = $null
$cameraProcess = $null
$reader = $null
$previousPythonHome = $env:PYTHONHOME
$previousPythonPath = $env:PYTHONPATH
try {
    # One running FVP owns this request directory. Another build can run independently.
    $lock = [IO.File]::Open((Join-Path $BuildDirectory 'demo.lock'), [IO.FileMode]::OpenOrCreate,
        [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    [Mlek.FvpAudioConsole]::Start()
    if ($Camera) {
        $inputDirectory = Join-Path $BuildDirectory 'demo-input'
        New-Item -ItemType Directory -Force -Path $inputDirectory | Out-Null
        # Wait for an explicit capture; never replay a previous run's image.
        $requestFile = Join-Path $inputDirectory "$Demo-input.bin"
        if (Test-Path -LiteralPath $requestFile) { Remove-Item -LiteralPath $requestFile }
    } else {
        $sender = if ($Demo -eq 'hand-pose') { 'send-hand-pose-image.ps1' } else { 'send-qrcode-image.ps1' }
        $sendOptions = @{ ImagePath=$ImagePath; BuildDirectory=$BuildDirectory; Python=$Python }
        if ($Demo -eq 'hand-pose') { $sendOptions.Framing = $CaptureFraming }
        & (Join-Path $PSScriptRoot $sender) @sendOptions
    }
    $arguments = @('-a', $firmware, '-C', 'mps4_board.subsystem.ethosu.num_macs=1024',
        '-C', 'mps4_board.subsystem.ethosu.extra_args=--fast',
        '-C', 'mps4_board.subsystem.cpu0.semihosting-enable=1',
        '-C', 'mps4_board.visualisation.disable-visualisation=1',
        '-C', "vis_hdlcd.disable_visualisation=$(if ($Headless) {1} else {0})",
        '-C', 'mps4_board.uart0.shutdown_on_eot=0')
    foreach ($uart in @(0, 1, 2, 5)) {
        $arguments += @('-C', "mps4_board.telnetterminal$uart.start_telnet=0")
    }
    # Image input uses host files, so no VSI/Python extension setup is required in FVP.
    $env:PYTHONHOME = $null
    $env:PYTHONPATH = $null
    $quoted = $arguments | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' }
    $log = Join-Path $BuildDirectory "$Demo-fvp.log"
    $fvp = Start-Process -FilePath $FvpExecutable -ArgumentList $quoted -PassThru `
        -WindowStyle Hidden -WorkingDirectory (Join-Path $BuildDirectory 'demo-input') `
        -RedirectStandardOutput $log `
        -RedirectStandardError (Join-Path $BuildDirectory "$Demo-fvp.err.log")
    $null = $fvp.Handle
    $stream = [IO.File]::Open($log, [IO.FileMode]::Open, [IO.FileAccess]::Read,
        [IO.FileShare]::ReadWrite)
    $reader = [IO.StreamReader]::new($stream, [Text.Encoding]::UTF8)
    if ($Camera) {
        $cameraArguments = @((Join-Path $repoRoot 'scripts/py/qrcode_camera_demo.py'),
            '--build-directory', $BuildDirectory, '--camera-index', "$CameraIndex",
            '--backend', $CameraBackend, '--demo', $Demo, '--framing', $CaptureFraming)
        $quotedCamera = $cameraArguments | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' }
        $cameraProcess = Start-Process -FilePath $Python -ArgumentList $quotedCamera -PassThru `
            -WindowStyle Hidden -WorkingDirectory $repoRoot `
            -RedirectStandardOutput (Join-Path $BuildDirectory "$Demo-camera.log") `
            -RedirectStandardError (Join-Path $BuildDirectory "$Demo-camera.err.log")
        $null = $cameraProcess.Handle
        Write-Host 'Focus the camera preview: SPACE captures, Q / ESC or closing it stops the demo.'
    }
    $tail = ''
    $deadline = [DateTime]::UtcNow.AddSeconds($SmokeTimeoutSeconds)
    if ($Demo -eq 'qrcode') { Write-Host 'QR/barcode detection only; no encoded-content decoding.' }
    else { Write-Host 'Hand pose: RGB image, 21 keypoints and 20 skeleton edges.' }
    Write-Host 'Press Ctrl+C or Q in this terminal to stop.'
    if (-not $Camera) { Write-Host 'Send another image from a second terminal.' }
    while (-not (Test-FvpAudioStopRequested)) {
        $chunk = $reader.ReadToEnd()
        if ($chunk) {
            Write-Host -NoNewline $chunk
            $tail += $chunk
            if ($tail.Length -gt 8192) { $tail = $tail.Substring($tail.Length - 8192) }
        }
        if ($tail -match "${marker}_ERROR|terminated with failure|Failed to initialise|QR-code .*failed|QR-code .*requires|QR-code .*mismatch|QR-code .*fit") {
            throw "$Demo inference failed; inspect $log"
        }
        $fvp.Refresh()
        if ($cameraProcess) {
            $cameraProcess.Refresh()
            if ($cameraProcess.HasExited) {
                if ($cameraProcess.ExitCode -ne 0) {
                    throw "Camera failed; inspect $BuildDirectory/$Demo-camera.log and $Demo-camera.err.log"
                }
                break
            }
        }
        if ($SmokeTimeoutSeconds -gt 0) {
            if ($tail -match "${marker}_RESULT id=\d+ count=\d+") { break }
            if ([DateTime]::UtcNow -ge $deadline) { throw "$Demo result timeout; inspect $log" }
            if ($fvp.HasExited) { throw "FVP exited before $Demo result; inspect $log" }
        } elseif ($fvp.HasExited) {
            if ($fvp.ExitCode -ne 0) { throw "FVP exited with code $($fvp.ExitCode)" }
            break
        }
        Start-Sleep -Milliseconds 100
    }
} finally {
    if ($reader) { $reader.Dispose() }
    Stop-FvpAudioProcess $cameraProcess
    Stop-FvpAudioProcess $fvp
    if ($lock) { $lock.Dispose() }
    [Mlek.FvpAudioConsole]::Stop()
    $env:PYTHONHOME = $previousPythonHome
    $env:PYTHONPATH = $previousPythonPath
}
