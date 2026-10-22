# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# Export and build pico-face for Corstone-320 FVP or FPGA; Run supports FVP only. #>
[CmdletBinding()]
param(
    [ValidateSet('Export', 'Build', 'Run', 'Package', 'All')][string]$Action = 'Build',
    [ValidateSet('ATFE', 'GNU')][string]$Toolchain = 'ATFE',
    [ValidateSet('FVP', 'FPGA')][string]$Target = 'FVP',
    [ValidateSet('Dedicated_Sram', 'Shared_Sram')][string]$MemoryMode = 'Dedicated_Sram',
    [string]$CompilerBin,
    [string]$ExportPython,
    [string]$Checkpoints,
    [string]$BuildDirectory,
    [string]$FvpExecutable,
    [switch]$UartImage,
    [int]$Jobs = 4,
    [int]$Seed = 3,
    [ValidateSet(1, 2, 4, 8)][int]$Steps = 4,
    [ValidateRange(0, 4)][int]$ClassId = 1,
    [ValidateRange(0, 100)][double]$Guidance = 4.0
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).ProviderPath
if ($UartImage -and $Target -ne 'FPGA') {
    throw '-UartImage requires -Target FPGA.'
}
$sourceClone = Join-Path $repoRoot '../arm-pico-face/CMSIS-Executorch'
if ($Target -eq 'FPGA' -and $Action -in @('Run', 'All')) {
    throw 'Use Export, Build or Package for FPGA. This script does not flash or run a board.'
}
if ($Target -eq 'FVP' -and $Action -eq 'Package') {
    throw 'Package creates FPGA board files. Select -Target FPGA.'
}
if (-not $FvpExecutable) {
    $FvpExecutable = Join-Path $sourceClone `
        '.tools/fvp-11.32.23/windows_x86/bin/FVP_Corstone_SSE-320.exe'
}
$shared = $MemoryMode -eq 'Shared_Sram'
$variant = if ($shared) { 'shared' } else { 'dedicated-384k' }
$cacheTag = if ($shared) { 'no-cache' } else { '384k' }
$cacheBytes = if ($shared) { 0 } else { 393216 }
$modelDir = Join-Path $repoRoot "resources_downloaded/pico_face/u85-1024-$variant"
if (-not $ExportPython) { $ExportPython = Join-Path $sourceClone '.venv/Scripts/python.exe' }
if (-not $Checkpoints) { $Checkpoints = Join-Path $sourceClone 'model/pico_faces' }
if (-not $BuildDirectory) {
    $relative = 'build/pico_face/pte/{0}/{1}/{2}/{3}-release-default-pmu' -f `
        $Toolchain.ToLowerInvariant(), $Target.ToLowerInvariant(), `
        $MemoryMode.ToLowerInvariant(), $cacheTag
    $BuildDirectory = Join-Path $repoRoot $relative
} elseif (-not [IO.Path]::IsPathRooted($BuildDirectory)) {
    $BuildDirectory = Join-Path $repoRoot $BuildDirectory
}
$BuildDirectory = [IO.Path]::GetFullPath($BuildDirectory)
$logRoot = Join-Path $repoRoot 'build/logs/pico_face_logs'
$buildLogDirectory = Join-Path $logRoot 'build'
$fvpLogDirectory = Join-Path $logRoot 'fvp'
New-Item -ItemType Directory -Path $buildLogDirectory, $fvpLogDirectory -Force | Out-Null
$logTag = '{0}-{1}-seed{2}-steps{3}-{4}' -f $Target.ToLowerInvariant(), $variant, `
    $Seed, $Steps, [DateTime]::Now.ToString('yyyyMMdd_HHmmss_fff')
$savedPythonPath = $env:PYTHONPATH
$savedUnbuffered = $env:PYTHONUNBUFFERED
Push-Location $repoRoot
try {
    $env:PYTHONPATH = Join-Path $repoRoot 'scripts/py'
    $env:PYTHONUNBUFFERED = '1'
    if ($Action -in @('Export', 'All')) {
        if (-not (Test-Path -LiteralPath $ExportPython)) {
            throw 'Supply -ExportPython pointing to the prepared Arm backend Python environment.'
        }
        & $ExportPython -m mlek_tools.pico_face.export --checkpoints $Checkpoints `
            --output $modelDir --vela-config (Join-Path $repoRoot 'scripts/vela/default_vela.ini') `
            --memory-mode $MemoryMode 2>&1 |
            Tee-Object -FilePath (Join-Path $buildLogDirectory "export-$logTag.log")
        if ($LASTEXITCODE -ne 0) { throw 'Pico-face export failed.' }
    }
    if ($Action -in @('Build', 'All', 'Package')) {
        $manifest = Get-Content (Join-Path $modelDir 'manifest.json') -Raw | ConvertFrom-Json
        $pte = Join-Path $modelDir 'pico_face.pte'
        $params = Join-Path $modelDir 'PicoFaceParams.hpp'
        if ($manifest.target -ne 'ethos-u85-1024' -or
            $manifest.memory_mode -ne $MemoryMode -or $manifest.cache_bytes -ne $cacheBytes -or
            (Get-FileHash $pte -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifest.pte_sha256 -or
            (Get-FileHash $params -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifest.params_sha256) {
            throw 'Model configuration or model/conditioning hashes do not match.'
        }
        $semihosting = if ($Target -eq 'FVP') { 'ON' } else { 'OFF' }
        $uartImageOption = if ($UartImage) { 'ON' } else { 'OFF' }
        $tempBase = if ($shared) { '0x31000000' }
                    elseif ($Target -eq 'FPGA') { '0x72000000' } else { '0x94000000' }
        $tempSize = if ($shared) { '0x00200000' } else { '0x00400000' }
        if ($shared -and (-not $manifest.scratch_bytes -or
            $manifest.scratch_bytes.dit_step -gt 2097152 -or
            $manifest.scratch_bytes.decode -gt 2097152)) {
            throw 'Shared model scratch is missing or exceeds the 2 MiB SRAM budget.'
        }
        $buildOptions = @{
            Action='Build'; Toolchain=$Toolchain; Target=$Target; UseCase='pico_face';
            ModelPath=$pte; BuildDirectory=$BuildDirectory; BuildType='Release';
            CacheSizeKiB=($cacheBytes / 1024); TimingAdapterConfig='ta_config_u85_sys_dram_1024';
            MemoryMode=$MemoryMode; Jobs=$Jobs;
            CMakeArgs=@('-DMPS4_DISPLAY_ENABLED=OFF', "-DSEMIHOSTING_ENABLED=$semihosting",
                "-DML_FWK_TMP_MEM_BASE=$tempBase", "-DML_FWK_TMP_MEM_SIZE=$tempSize",
                "-DETHOS_U_NPU_CACHE_SIZE=$cacheBytes", "-Dpico_face_SAVE_IMAGE=$semihosting",
                "-Dpico_face_UART_IMAGE=$uartImageOption",
                "-Dpico_face_SEED=$Seed", "-Dpico_face_STEPS=$Steps", "-Dpico_face_CLASS_ID=$ClassId",
                "-Dpico_face_GUIDANCE=$($Guidance.ToString([Globalization.CultureInfo]::InvariantCulture))")
        }
        if ($CompilerBin) { $buildOptions.CompilerBin = $CompilerBin }
        if ($Action -ne 'Package') {
            $buildLog = Join-Path $buildLogDirectory "build-$logTag.log"
            Write-Host "Build log: $buildLog"
            & (Join-Path $PSScriptRoot 'build-executorch.ps1') @buildOptions *>&1 |
                Tee-Object -FilePath $buildLog
        }
        if ($Target -eq 'FPGA') {
            $firmwareInfo = Get-Content (Join-Path $BuildDirectory 'build-info.json') -Raw |
                ConvertFrom-Json
            $configuration = $firmwareInfo.configuration
            if ($firmwareInfo.status -ne 'built' -or
                $firmwareInfo.model_sha256 -ne $manifest.pte_sha256 -or
                $configuration.FPGA_PLATFORM_SSE_320 -ne 'ON' -or
                $configuration.ETHOS_U_NPU_MEMORY_MODE -ne $MemoryMode -or
                $configuration.ETHOS_U_NPU_CACHE_SIZE -ne "$cacheBytes" -or
                $configuration.SEMIHOSTING_ENABLED -ne 'OFF' -or
                $configuration.pico_face_SAVE_IMAGE -ne 'OFF') {
                throw 'Build provenance does not match the requested FPGA package.'
            }
            $tempBase = $configuration.ML_FWK_TMP_MEM_BASE
            $tempSize = $configuration.ML_FWK_TMP_MEM_SIZE
            $Seed = $configuration.pico_face_SEED
            $Steps = $configuration.pico_face_STEPS
            $ClassId = $configuration.pico_face_CLASS_ID
            $Guidance = $configuration.pico_face_GUIDANCE
            $package = Join-Path $BuildDirectory 'board-package'
            $software = Join-Path $package 'SOFTWARE'
            New-Item -ItemType Directory -Path $software -Force | Out-Null
            $sectors = Join-Path $BuildDirectory 'bin/sectors/pico_face'
            foreach ($name in @('boot.bin', 'bram.bin', 'ddr.bin')) {
                $binary = Join-Path $sectors $name
                if (-not (Test-Path -LiteralPath $binary) -or
                    (Get-Item -LiteralPath $binary).Length -eq 0) {
                    throw "Missing or empty FPGA binary: $binary"
                }
                Copy-Item -LiteralPath $binary -Destination (Join-Path $software $name) -Force
            }
            Copy-Item -LiteralPath (Join-Path $BuildDirectory 'bin/sectors/images.txt') `
                -Destination (Join-Path $package 'images.txt') -Force
            Copy-Item -LiteralPath (Join-Path $BuildDirectory 'bin/mlek_pico_face.axf'), `
                (Join-Path $BuildDirectory 'bin/mlek_pico_face.map'), `
                (Join-Path $BuildDirectory 'build-info.json') -Destination $package -Force
            Copy-Item -LiteralPath (Join-Path $modelDir 'manifest.json') `
                -Destination (Join-Path $package 'model-manifest.json') -Force
            @"
Pico-face / Corstone-320 FPGA / U85 Z1024 / $MemoryMode
NPU cache bytes: $cacheBytes
Method pool: DDR, 4 MiB capacity. Temporary pool: $tempBase, $tempSize bytes.
UART: 115200 8N1, no flow control. No semihosting and no display.
UART image service: $($configuration.pico_face_UART_IMAGE). When ON, send I after
SERIAL_IMAGE_READY to retrieve the stored binary frame; Q exits the service.
Seed=$Seed Steps=$Steps Class=$ClassId Guidance=$Guidance

Copy the three files in SOFTWARE to the board's SOFTWARE directory.
Install images.txt in the active Corstone-320 image configuration directory,
as for the previously validated MLEK use cases (do not overwrite the FPGA bitstream).
For the documented MPS4 revision B/FI101 example this is MB/HBI0376B/FI101/images.txt;
use the directory matching the board's actual configuration.

The image file preserves the standard MLEK addresses:
boot.bin: 0x11000000, bram.bin: 0x12000000, ddr.bin: 0x70000000.
Expected successful UART marker: PICO_FACE_PASS.
RGB888 data address and CRC are printed on UART; read 49152 bytes with a debugger.
This package is built and inspected only; physical board inference is not validated.
"@ | Set-Content -LiteralPath (Join-Path $package 'README.txt') -Encoding utf8
            $hashes = foreach ($file in Get-ChildItem -LiteralPath $package -File -Recurse |
                Where-Object Name -ne 'SHA256SUMS.txt' | Sort-Object FullName) {
                $relativeName = $file.FullName.Substring($package.Length + 1).Replace('\', '/')
                '{0}  {1}' -f (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant(), $relativeName
            }
            $hashes | Set-Content -LiteralPath (Join-Path $package 'SHA256SUMS.txt') -Encoding ascii
            $uartTag = if ($configuration.pico_face_UART_IMAGE -eq 'ON') { '-uart' } else { '' }
            $sampleTag = if ($Seed -ne 3 -or $Steps -ne 4 -or $ClassId -ne 1 -or $Guidance -ne 4) {
                "-seed$Seed-steps$Steps-class$ClassId-guidance$Guidance"
            } else { '' }
            $archive = Join-Path $BuildDirectory "pico-face-fpga-$variant$sampleTag$uartTag.zip"
            Compress-Archive -Path (Join-Path $package '*') -DestinationPath $archive -Force
            Write-Host "FPGA package: $package"
            Write-Host "FPGA archive: $archive"
        }
    }
    if ($Action -in @('Run', 'All')) {
        $firmware = Join-Path $BuildDirectory 'bin/mlek_pico_face.axf'
        if (-not (Test-Path -LiteralPath $firmware)) { throw "Missing $firmware" }
        if (-not (Test-Path -LiteralPath $FvpExecutable)) { throw "Missing $FvpExecutable" }
        Push-Location $BuildDirectory
        try {
            Remove-Item -LiteralPath 'pico_face.rgb' -Force -ErrorAction SilentlyContinue
            $arguments = @('-a', $firmware,
                '-C', 'mps4_board.subsystem.ethosu.num_macs=1024',
                '-C', 'mps4_board.subsystem.cpu0.semihosting-enable=1',
                '-C', 'mps4_board.subsystem.cpu0.semihosting-stack_base=0',
                '-C', 'mps4_board.subsystem.cpu0.semihosting-heap_limit=0',
                '-C', 'mps4_board.telnetterminal0.start_telnet=0',
                '-C', 'mps4_board.uart0.out_file=-',
                '-C', 'mps4_board.uart0.shutdown_on_eot=1',
                '-C', 'mps4_board.visualisation.disable-visualisation=1',
                '-C', 'vis_hdlcd.disable_visualisation=1', '--simlimit', '120', '--stat')
            $runConfiguration = (Get-Content (Join-Path $BuildDirectory 'build-info.json') `
                -Raw | ConvertFrom-Json).configuration
            $runTag = '{0}-seed{1}-steps{2}-{3}' -f $variant, `
                $runConfiguration.pico_face_SEED, $runConfiguration.pico_face_STEPS, `
                [DateTime]::Now.ToString('yyyyMMdd_HHmmss_fff')
            $fvpLog = Join-Path $fvpLogDirectory "fvp-$runTag.log"
            Write-Host "FVP log: $fvpLog"
            & $FvpExecutable @arguments 2>&1 | Tee-Object -FilePath $fvpLog
            if ($LASTEXITCODE -ne 0) { throw "FVP failed: $LASTEXITCODE" }
            if (-not (Select-String -LiteralPath $fvpLog -Pattern 'PICO_FACE_PASS' -Quiet) -or
                -not (Test-Path pico_face.rgb) -or (Get-Item pico_face.rgb).Length -ne 49152) {
                throw "FVP did not produce a complete passing RGB result; see $fvpLog."
            }
            & (Join-Path $repoRoot 'resources_downloaded/windows-executorch/env/Scripts/python.exe') `
                -c "from PIL import Image; from pathlib import Path; Image.frombytes('RGB',(128,128),Path('pico_face.rgb').read_bytes()).save('pico_face.png')"
            if ($LASTEXITCODE -ne 0) { throw 'RGB conversion failed.' }
            Write-Host "Image: $(Join-Path $BuildDirectory 'pico_face.png')"
        } finally { Pop-Location }
    }
} finally {
    $env:PYTHONPATH = $savedPythonPath
    $env:PYTHONUNBUFFERED = $savedUnbuffered
    Pop-Location
}
