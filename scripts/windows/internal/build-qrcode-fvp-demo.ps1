# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
<# Build the resident grayscale QR/barcode detector with semihosted image requests. #>
[CmdletBinding()]
param(
    [string]$ModelPath = 'resources_downloaded/yolo_QRcode/usr_quant_nhwc_angle3d/pte/best_bifpn_gray_nhwc_angle3d_u85_1024_384KiB.pte',
    [string]$BuildDirectory = 'build/fvp-qrcode/dedicated-384k',
    [string]$CompilerBin,
    [ValidateSet('Configure', 'Build')][string]$Action = 'Build',
    [ValidateRange(32, 1024)][int]$Width = 256,
    [ValidateRange(32, 1024)][int]$Height = 256,
    [ValidateRange(1, 4096)][int]$CacheSizeKiB = 384,
    [ValidateRange(1, 128)][int]$Jobs = 4
)
$ErrorActionPreference = 'Stop'
if ($Width % 32 -or $Height % 32) { throw 'Width and Height must be multiples of 32.' }
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).ProviderPath
$downscale = [Math]::Max(1, [Math]::Max([Math]::Ceiling($Width / 160.0),
    [Math]::Ceiling($Height / 192.0)))
$options = @{
    Framework='pte'; Toolchain='ATFE'; Target='FVP'; Action=$Action; UseCase='qrcode'
    ModelPath=$ModelPath; BuildDirectory=$BuildDirectory; CompilerBin=$CompilerBin
    MemoryMode='Dedicated_Sram'; CacheSizeKiB=$CacheSizeKiB; BuildType='Release'; Jobs=$Jobs
    ActivationBufferSize='0x00400000'; InputPath='resources/qrcode/smaples/01.bmp'
    CMakeArgs=@('-Dqrcode_INPUT_MODE=SEMIHOST', '-DSEMIHOSTING_ENABLED=ON',
        '-DMPS4_DISPLAY_ENABLED=ON', '-DFVP_VSI_ENABLED=OFF', '-DINTERACTIVE_MODE=OFF',
        '-DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=OFF',
        '-Dqrcode_ET_METHOD_MEMORY_IN_DDR=ON', '-Dqrcode_MAX_DETECTIONS=5',
        "-Dqrcode_IMAGE_WIDTH=$Width", "-Dqrcode_IMAGE_HEIGHT=$Height",
        "-Dqrcode_DISPLAY_DOWNSCALE=$downscale", '-Dqrcode_IMAGE_RESIZE_MODE=letterbox',
        '-DML_FWK_TMP_MEM_BASE=0x94000000', '-DML_FWK_TMP_MEM_SIZE=0x02000000',
        '-DHDLCD_FRAME_BUFFER_BASE_ADDRESS=0x74000000')
}
Push-Location $repoRoot
try { & (Join-Path $PSScriptRoot 'build-firmware.ps1') @options }
finally { Pop-Location }
