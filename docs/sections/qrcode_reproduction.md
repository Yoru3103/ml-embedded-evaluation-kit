# QR-code TFLM / PTE comparison and reproduction

<!-- SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

## 当前整合后的部署约定

灰度 QOperator 模型保留 DFL8、OBB、阈值筛选与旋转 NMS，应用最多返回五个检测目标。
已确认灰度输入且模型为单通道、INT8、scale=1/255、zero point=-128 时走直接量化快路径；
普通 RGB 输入仍执行 RGB 转灰度，其他量化参数使用通用查表路径。
上游新增的 `qrcode` 用例使用单字节灰度样本，原 `yolov8_detection` 的 Windows 入口继续保留。

## 256 KiB 模型的历史对比

本页比较 2026-10-12 导出的 NHWC PTE 与 `onnx2tf_nhwc_256k` 的 TFLM 模型。
两者来自相同 `best.onnx`（SHA256：
`4e9338b59b0d7ffd9109152f7acd10175fc0654411afa73c2c9dced7eaa629cc`），
采用居中裁剪、RGB /255、相同六张样本、INT8 输入和四个 INT8 输出。
FPGA 固件均为 MPS4 SSE-320、U85 Z1024、Dedicated SRAM、Release、BW PMU、
`ta_config_u85_sys_dram_mid`，内置 `samples_QRcode/1.png`。
256 KiB 指 NPU fast cache，输入分辨率另为 256×256。

> **模型语义待确认，原模型验收暂缓：** 本页 RGB `best.onnx` 的第四输出目前按
> objectness 使用，但其 `cv4` 分支结构提示它可能是原始角度 logits；训练／导出契约
> 和角度映射尚未确认。下面的输出比较和数值检查只验证当前实现规则的一致性，
> 不代表该规则已符合训练模型的检测语义。另行部署的灰度 QOperator 模型使用
> 当时的 DFL8、OBB、Top1 配置，其实板结果不能替代本页 RGB 模型的验收；当前五目标配置见上文。

| 项目 | PTE 256 KiB | TFLM 256 KiB（onnx2tf） |
| --- | --- | --- |
| 前端 | ONNX → onnx2torch → torch.export → PT2E | ONNX → onnx2tf tf_converter → TensorFlow INT8 |
| NPU 编译 | ExecuTorch Arm → TOSA → Vela | INT8 TFLite → Vela |
| 输入逻辑形状 | `[1,3,256,256]` | `[1,256,256,3]` |
| 输入及检测头实际存储 | NHWC，四维 `dim_order=[0,2,3,1]` | NHWC，连续张量 |
| 输出顺序 | 32×32、16×16、8×8、辅助输出（当前按 objectness 使用） | 16×16、辅助输出（当前按 objectness 使用）、32×32、8×8 |
| 输入 scale / zero point | 0.0039043352 / -128 | 0.0039215689 / -128 |
| CPU 算子 / NPU 分区 | 0 / 1 | 0 / 1 |
| 模型文件大小 | 449,776 bytes | 462,944 bytes |
| Vela 总周期估算 | 847,233 | 722,895 |
| Vela SRAM / DRAM 估算（KiB） | 211.53 / 591.44 | 226.00 / 613.41 |
| Vela DRAM 总访问量（bytes） | 3,145,632 | 3,049,232 |
| 固件内存用途 | 4 MiB method pool 在 DDR；NPU scratch 另行分配 | 4 MiB TFLM tensor arena |

当前两个模型的输入存储都是 NHWC。PTE 的 NCHW **逻辑形状**不表示平面存储。
量化器、融合和重写规则仍不同：PTE 为适配 Arm 量化融合将 ReLU 改为 clamp，
而 onnx2tf 直接转换原 ONNX。两者的 scale、权重编码和编译图不同，模型无法互换。
PTE 的 Vela 周期估算高约 17.2%；这里没有实板延迟或完整系统内存的测量。
各输出的量化参数必须分别读取，不能跨模型复用。
PTE 固件从配置方法读取参数；TFLM 从 FlatBuffer 读取参数。

## 统一入口与目录

Windows 固件构建统一使用 `scripts/windows/build.ps1 -Framework tflm|pte`。
`build-executorch.ps1` 继续作为兼容入口，和 TFLM 共用目录生成及参数映射。
两侧均在构建前重新配置、生成模型/图片/标签源代码，再编译，并写出
`build-info.json`（实际 CMake 配置、模型 SHA256、配置与构建命令、完成状态）。

```text
resources_downloaded/yolo_QRcode/
  exports/tflm/                 # 原 ONNX → FP32/INT8 TFLite
  exports/pte/                  # 原 ONNX → FP32/量化 PT2 与 PTE
  compiled/tflm/256k|384k/       # 同一量化 TFLite 的 Vela 缓存变体
  compiled/pte/64k|128k|256k|384k|768k/ # 同一量化 PT2 的缓存变体
  compiled/pte/shared/          # 同一量化 PT2 的 Shared SRAM 变体
build/yolov8_detection/
  tflm/atfe/fpga/dedicated_sram/256k-release-bw/
  pte/atfe/fpga/dedicated_sram/384k-release-bw/
build/board-test-packages/yolov8_detection/
  tflm/256k/
  pte/384k/
```

原有构建目录仍保留；已有 CMake 缓存包含绝对路径，不能通过移动目录迁移。
新命令使用上述目录。`-BuildDirectory` 仍可覆盖默认位置。

## 转换：首次导出并校准

以下均从项目根目录运行。PTE 导出在 WSL Ubuntu-24.04 中使用已有
`resources_downloaded/linux-batch/env`；TFLM 导出在 Windows 使用独立 onnx2tf 环境。
两侧使用相同的六张图、seed、预处理和 Vela system config。
先生成一份 384 KiB 导出，再从固定量化图编译所需缓存变体，避免重复校准。

### TFLM：Windows PowerShell

```powershell
$py = '.\resources_downloaded\windows-onnx2tf\env\Scripts\python.exe'
# 新机器首次准备；现有环境可以跳过。
py -3.12 -m venv .\resources_downloaded\windows-onnx2tf\env
& $py -m pip install -r .\scripts\windows\requirements-qrcode-onnx2tf.txt

& $py .\scripts\export_qrcode_onnx.py `
    --format tflm --tflite-backend onnx2tf `
    --model .\resources_downloaded\yolo_QRcode\best.onnx `
    --calibration-dir .\resources\object_detection\samples_QRcode `
    --calibration-limit 0 --validation-limit 6 --seed 0 `
    --preprocessing mlek-crop --per-channel `
    --target ethos-u85-1024 --system-config Ethos_U85_SYS_DRAM_Mid_1024 `
    --memory-mode Dedicated_Sram --cache-kib 384 `
    --output-dir .\resources_downloaded\yolo_QRcode\exports\tflm

& $py .\scripts\compile_qrcode_fastcache.py `
    --format tflm --cache-kib 256 384 `
    --model-root .\resources_downloaded\yolo_QRcode\exports\tflm `
    --output-root .\resources_downloaded\yolo_QRcode\compiled
```

### PTE：WSL Bash

```bash
cd /mnt/c/Users/MC/Documents/mlek-client/ml-embedded-evaluation-kit
PY=resources_downloaded/linux-batch/env/bin/python
# 现有环境的版本：torch 2.12.0+cpu、torchao 0.17.0+cpu、ExecuTorch 1.3.1+cpu、
# Vela 5.1.0、onnx2torch 1.5.15、ONNX 1.19.1、ONNX Runtime 1.31.0。

PYTHONPATH=scripts "$PY" scripts/export_qrcode_onnx.py \
    --format pte \
    --model resources_downloaded/yolo_QRcode/best.onnx \
    --calibration-dir resources/object_detection/samples_QRcode \
    --calibration-limit 0 --validation-limit 6 --seed 0 \
    --preprocessing mlek-crop --per-channel \
    --target ethos-u85-1024 --system-config Ethos_U85_SYS_DRAM_Mid_1024 \
    --memory-mode Dedicated_Sram --cache-kib 384 \
    --extra-flag=--optimise=Performance --extra-flag=--cop-format=COP1 \
    --extra-flag=--verbose-performance \
    --output-dir resources_downloaded/yolo_QRcode/exports/pte

PYTHONPATH=scripts "$PY" scripts/compile_qrcode_fastcache.py \
    --format pte --cache-kib 256 384 \
    --model-root resources_downloaded/yolo_QRcode/exports/pte \
    --output-root resources_downloaded/yolo_QRcode/compiled
```

`compile_qrcode_fastcache.py` 支持从 `pte_Dedicated_Sram` 导出中读取量化 PT2，
与最初导出时的缓存大小无关。TFLM 路线不会加载 PyTorch/ExecuTorch。
`--cache-kib` 与旧的字节参数 `--arena-cache-size` 互斥。

本次实际复用已有导出，可跳过上面的首次导出，分别使用以下 `--model-root`：

```text
TFLM: resources_downloaded/yolo_QRcode/onnx2tf_nhwc_384k
PTE:  resources_downloaded/yolo_QRcode/pte_nhwc_256k
```

本次新生成的 TFLM 256 KiB 模型 SHA256 与旧 `onnx2tf_nhwc_256k` 模型相同。
PTE 384 KiB 与原 256 KiB 共用同一份量化 PT2，只重新进行目标缓存编译。

## FPGA 构建：同一组参数

Windows PowerShell，项目根目录：

```powershell
$compiler = 'C:\Program Files\Arm\Development Studio 2026.0-1\sw\atfe-pro22.1.0\bin'
$common = @{
    Action = 'Build'
    Toolchain = 'ATFE'
    Target = 'FPGA'
    UseCase = 'yolov8_detection'
    CompilerBin = $compiler
    InputPath = '.\resources\object_detection\samples_QRcode\1.png'
    LabelsPath = '.\resources_downloaded\yolo_QRcode\labels.txt'
    ImageSize = 256
    NumClasses = 15
    ActivationBufferSize = '0x00400000'
    MemoryMode = 'Dedicated_Sram'
    BuildType = 'Release'
    TimingAdapterConfig = 'ta_config_u85_sys_dram_mid'
    PmuGroup = 'BW'
    Jobs = 8
}
$baseArgs = @('-DINTERACTIVE_MODE=OFF', '-DSEMIHOSTING_ENABLED=OFF')

foreach ($cache in @(256, 384)) {
    & .\scripts\windows\build.ps1 @common `
        -Framework tflm -CacheSizeKiB $cache `
        -ModelPath ".\resources_downloaded\yolo_QRcode\compiled\tflm\${cache}k\best_int8_vela.tflite" `
        -CMakeArgs $baseArgs

    & .\scripts\windows\build.ps1 @common `
        -Framework pte -CacheSizeKiB $cache `
        -ModelPath ".\resources_downloaded\yolo_QRcode\compiled\pte\${cache}k\best.pte" `
        -CMakeArgs ($baseArgs + '-Dyolov8_detection_ET_METHOD_MEMORY_IN_DDR=ON')
}
```

单独构建 384 KiB PTE 时，直接调用循环中的 PTE 命令并设置 `$cache = 384`。
初次 Windows 固件环境准备使用相同入口：

```powershell
& .\scripts\windows\build.ps1 -Framework tflm -Action Setup -UseCase yolov8_detection
& .\scripts\windows\build.ps1 -Framework pte -Action Setup -UseCase yolov8_detection
```

模型转换和固件 build 是独立步骤：`-CacheSizeKiB` 设置固件 cache，
必须与所选模型的 Vela 编译缓存相同；构建脚本不会自动重新转换模型。

## 64 / 128 / 768 KiB 与 Shared SRAM PTE

在 WSL 的项目根目录中复用已经校准的 NHWC 量化图：

```bash
PYTHONPATH=scripts resources_downloaded/linux-batch/env/bin/python \
    scripts/compile_qrcode_fastcache.py \
    --format pte --memory-mode all --cache-kib 64 128 768 \
    --model-root resources_downloaded/yolo_QRcode/pte_nhwc_256k \
    --output-root resources_downloaded/yolo_QRcode/compiled
```

前三个模型为 Dedicated SRAM。Shared SRAM 的激活区放在 SRAM，权重放在 DRAM，
不设置 Dedicated fast-cache 容量；报告的 `cache_bytes` 为 null。
只生成 Shared SRAM 时使用 `--memory-mode Shared_Sram`，不传 `--cache-kib`。
相同选项也适用于 `--format tflm`。

Shared PTE 的 FPGA 固件使用 `-MemoryMode Shared_Sram`，不传 `-CacheSizeKiB`。
保留 CPU method pool 在 DDR，同时将 NPU 临时分配区配置在 SRAM，以匹配模型的
SRAM 激活区；四个定义加入 `-CMakeArgs`：

```powershell
'-Dyolov8_detection_ET_METHOD_MEMORY_IN_DDR=ON'
'-DML_FWK_TMP_MEM_BASE=0x31000000'
'-DML_FWK_TMP_MEM_SIZE=0x00200000'
'-DETHOS_U_NPU_CACHE_SIZE=0'
```

默认目录为 `build/yolov8_detection/pte/atfe/fpga/shared_sram/no-cache-release-bw`。
NPU 临时区占 SRAM 的前 2 MiB，帧缓冲由平台配置计算到 `0x31200000`，
两者不重叠；CPU method pool 为 DDR 中的 4 MiB。

PTE 文件格式本身不要求 Linux，通用 ExecuTorch host 支持 Windows；但
[Ethos-U 1.3 官方导出教程](https://docs.pytorch.org/executorch/1.3/backends/arm-ethos-u/tutorials/ethos-u-getting-started.html)
以 Linux/macOS 为参考环境。本工程目前在 WSL 验证 Ethos-U 导出，Windows
用于固件交叉编译。Windows 固件构建环境并不等于完整的 Ethos-U 模型导出环境。

## 构建产物

构建完成后直接使用 `<BuildDirectory>/bin/mlek_yolov8_detection.axf`、
`bin/sectors/yolov8_detection/{boot,bram,ddr}.bin` 和 `bin/sectors/images.txt`。
无需执行打包命令。未执行实板验证；最终 PTE 数值闭环与 Resize 扩展检查继续暂缓。
