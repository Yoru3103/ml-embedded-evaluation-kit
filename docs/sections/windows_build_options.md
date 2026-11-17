# Windows 构建参数手册

<!-- SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

本页覆盖 `scripts/windows/build.ps1` 的入口参数和全部构建 profile。命令在仓库根目录的
Windows PowerShell 中执行。首次准备环境见 [Windows 快速入门](windows.md)。
文件路径有空格时加引号。表中标为「关闭」的开关，直接写参数名即可启用。

[完整示例](#完整示例) · [通用固件](#firmware) · [视觉](#视觉) ·
[音频与串口](#音频与串口) · [生成模型与语言模型](#生成模型与语言模型)

## 入口参数

| 参数 | 默认值或可选值 | 使用方式 |
| --- | --- | --- |
| `-Profile` | 默认 `Firmware`，其余选择见下文 | 选择通用构建或专用预设，例如 `-Profile VisionBenchmark`。 |
| `-ListProfiles` | 关闭 | 列出当前版本提供的构建 profile。 |
| `-ProfileHelp` | 关闭 | 查看所选 profile 的 PowerShell 帮助，例如 `-Profile Vision -ProfileHelp`。 |

## 完整示例

用仓库自带的 `cat.bmp` 和公开 MobileNetV2 模型构建 FVP、MPS4 两份固件。
两端都用 U85 Z1024、Dedicated SRAM、384 KiB cache、TA 开启、PMU A 和 Release。
在已安装 [前置工具](windows.md#首次运行) 的 Windows 上可直接复制执行。已有项目环境和模型时可跳过 Setup。`$compilerBin` 留空时自动查找 ATFE，`$model` 留空时使用 Setup 准备的模型。

```powershell
$useCase = 'img_class'
$framework = 'tflm'
$compilerBin = ''
$image = 'resources/img_class/samples/cat.bmp'
$model = ''
$memoryMode = 'Dedicated_Sram'
$cacheKiB = 384
$taConfig = 'ta_config_u85_sys_dram_mid'
$jobs = 4
$buildRoot = 'build/windows-perf/img-class-384k'

.\scripts\windows\build.ps1 -Profile Firmware -Action Setup -Framework $framework -UseCase $useCase -Jobs $jobs

$buildOptions = @{
    Profile             = 'Firmware'
    Action              = 'Build'
    Framework           = $framework
    Toolchain           = 'ATFE'
    UseCase             = $useCase
    InputPath           = $image
    MemoryMode          = $memoryMode
    CacheSizeKiB        = $cacheKiB
    TimingAdapterConfig = $taConfig
    PmuGroup            = 'A'
    BuildType           = 'Release'
    Jobs                = $jobs
    CMakeArgs           = @(
        '-DETHOS_U_NPU_CONFIG_ID=Z1024'
        '-DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=ON'
        '-DINTERACTIVE_MODE=OFF'
        '-DFVP_VSI_ENABLED=OFF'
        '-DSEMIHOSTING_ENABLED=OFF'
        '-DUSE_SINGLE_INPUT=ON'
    )
}
if ($compilerBin) { $buildOptions.CompilerBin = $compilerBin }
if ($model) { $buildOptions.ModelPath = $model }

.\scripts\windows\build.ps1 @buildOptions -Target FVP -BuildDirectory "$buildRoot/fvp"
.\scripts\windows\build.ps1 @buildOptions -Target FPGA -BuildDirectory "$buildRoot/mps4"
.\scripts\windows\run-fvp.ps1 -Profile Firmware -BuildDirectory "$buildRoot/fvp" -Headless
```

| 要改的项目 | 修改方式 |
| --- | --- |
| 图片 | 修改 `$image`。同一张图会编入两份固件。 |
| 模型、内存模式与 cache | 同时修改 `$model`、`$memoryMode` 和 `$cacheKiB`，传入与固件 NPU 配置和 cache 预算匹配的模型。切换到 Shared SRAM 时将 `$cacheKiB` 设为 `0`。变更配置后可为 `$buildRoot` 另取路径。 |
| 只构建一端 | 保留对应的 `-Target FVP` 或 `-Target FPGA` 那行。最后一条命令使用 FVP 固件。 |
| 查看结果 | 每端的 `build-info.json` 记录最终配置。FVP 的 `fvp.log` 记录 TA 初始化、cache 字节数和 `NPU TOTAL`。 |
| MPS4 上板 | 用 `"$buildRoot/mps4/bin/sectors/img_class"` 中的 `boot.bin`、`bram.bin`、`ddr.bin` 与上一级的 `images.txt`。文件放置和串口步骤见 [部署指南](deployment.md)。 |

如需 ZIP，填好 `$compilerBin` 后执行：

```powershell
.\scripts\windows\package-fpga.ps1 -BuildDirectory "$buildRoot/mps4" -CompilerBin $compilerBin -OutputDirectory "$buildRoot/mps4-package"
```

使用固定 384 KiB 视觉预设时，`-Profile VisionBenchmark -Demo img-class -Target Both`
可同时构建两端并直接生成上板包。

## 通用固件

### Firmware

| 参数 | 默认值或可选值 | 使用方式 |
| --- | --- | --- |
| `-Framework` | `tflm`。另有 `pte`、`executorch` | 选择 TFLM `.tflite` 或 ExecuTorch `.pte`，后两种拼法走同一路线。 |
| `-Toolchain` | TFLM 默认 `AC6`，可选 `ATFE`。PTE 默认 `ATFE`，可选 `GNU` | 选择交叉编译器。 |
| `-Target` | `FPGA`、`FVP`，默认 `FPGA` | 构建 MPS4 SSE-320 实板或 Corstone-320 仿真固件。 |
| `-Action` | `Setup`、`Configure`、`Build`、`All`，默认 `Build` | `Setup` 安装工具并准备适用资源，`Configure` 生成 CMake 构建文件，`Build` 配置并编译，`All` 先 Setup 再 Build。PTE 的 Setup 准备工具，模型由用户提供。 |
| `-UseCase` | `img_class` | 一次选择一个 CMake 用例，例如 `yolov8_detection`、`kws_zh`、`asr`。 |
| `-CompilerBin` | 自动查找 | 传含 `armclang.exe`、`clang.exe` 或 `arm-none-eabi-gcc.exe` 的 bin 目录，优先于编译器环境变量及安装目录扫描。 |
| `-ModelPath` | TFLM 取用例默认模型。PTE 构建时需传模型，Dedicated Pose 有预置路径 | 选择预编译 `.tflite` 或 `.pte`，映射到 `<UseCase>_MODEL_PATH`。 |
| `-InputPath` | 用例默认样本。PTE 的 `yolov8_detection` 默认用 `resources/gesture_detection/samples` | 指定样本文件或目录，映射到 `<UseCase>_FILE_PATH`。可用 `USE_SINGLE_INPUT` 控制样本数量。 |
| `-LabelsPath` | 用例默认标签 | 指定标签文件，映射到 `<UseCase>_LABELS_YAML_FILE`。用例需支持该选项。 |
| `-ImageSize` | 省略时不覆盖。显式值 `1–4096` | 设置支持 `<UseCase>_IMAGE_SIZE` 的用例的输入尺寸，须和模型匹配。 |
| `-NumClasses` | 省略时不覆盖。显式值 `1–1000` | 设置支持 `<UseCase>_NUM_CLASSES` 的用例的类别数，须和模型及标签匹配。 |
| `-ActivationBufferSize` | 用例默认值 | 传给 `<UseCase>_ACTIVATION_BUF_SZ`，单位字节，可写 `0x00400000`。 |
| `-MemoryMode` | TFLM 由 CMake 决定，PTE 默认 `Dedicated_Sram` | TFLM 可选 `Dedicated_Sram`、`Shared_Sram`、`Sram_Only`。PTE 可选前两项。 |
| `-BuildDirectory` | 按用例、框架、工具链、目标、内存、cache、构建类型和 PMU 自动生成 | 指定独立输出目录，例如 `build/my-fvp`。目录含 `build-info.json` 和 `bin/` 下的 AXF。 |
| `-BuildType` | `Debug`、`Release`、`RelWithDebInfo`、`MinSizeRel`，默认 `Debug` | 选择编译配置。性能对照时用相同的构建类型。 |
| `-CacheSizeKiB` | `0–1048576`，默认 `0` 表示沿用 CMake 或用例配置 | `384` 设置固件预留 `393216` 字节。实际可用值受目标 SRAM 布局约束。模型须用对应的 Vela cache 预算编译。 |
| `-TimingAdapterConfig` | 省略时沿用目标或用例配置 | 选择 TA 参数组，如 `ta_config_u85_sys_dram_mid`。用 `-CMakeArgs '-DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=ON'` 显式开启 TA。参见 [TA 配置目录](../../scripts/cmake/timing_adapter/) 和 [TA 说明](timing_adapters.md)。 |
| `-PmuGroup` | `A`、`B`、`B_SRAM0`、`B_SRAM1`、`BW`、`C`、`D`、`LATENCY`，省略时沿用用例或 CMake 默认值 | U85 默认 `A`。`BW` 用于带宽，`LATENCY` 用于延迟。各组选项见 [PMU 说明](windows_executorch.md#pose-的周期日志与-pmu)。 |
| `-Jobs` | `4`，范围 `1–128` | 编译并行数。TFLM Setup 也使用此值指定下载并行数。 |
| `-CMakeArgs` | 空数组 | 传一个或多个 `-DNAME=value`，如 `@('-DINTERACTIVE_MODE=OFF', '-DUSE_SINGLE_INPUT=ON')`。框架、平台、工具链等用命名参数选择。参见 [CMake 构建选项](building.md#build-options)。 |

## 视觉

`Vision`、`VisionBenchmark` 固定使用 ATFE、Release、Dedicated SRAM 和 384 KiB cache。
默认模型和外部资源来源见 [视觉资源说明](../../scripts/vsi/README.md)。

### Vision

| 参数 | 默认值或可选值 | 使用方式 |
| --- | --- | --- |
| `-Demo` | 必填。`gesture`、`coco`、`img-class` | 手势 PTE、COCO TFLM 检测或 MobileNetV2 TFLM 分类。 |
| `-Mode` | `VSI`、`Camera`、`Static`，默认 `VSI` | `VSI` 使用 KEY0，`Camera` 是 img_class 的非 VSI 摄像头版，`Static` 内置图片。 |
| `-Action` | `Setup`、`Configure`、`Build`、`All`，默认 `Build` | 准备、配置或构建。摄像头模式的 Setup/All 同时安装 OpenCV。 |
| `-CompilerBin` | 自动查找 ATFE | 可指定 ATFE bin。 |
| `-BuildDirectory` | `build/<target>-<demo>/<variant>` | 覆盖输出目录。变体包含 `vsi`、`vsi-auto`、`camera`、`static`，快速版加 `-fast`。 |
| `-ModelPath` | 随 Demo 选择 | 替换兼容模型。 |
| `-InputPath` | 随 Demo 选择仓库样本 | 更换内置样本。VSI 运行时文件输入由启动脚本的 `-ImagePath` 选择。 |
| `-AutoCapture` | 关闭 | 仅限 VSI，固件自动请求帧。可用于空格抓拍和文件输入。 |
| `-Fast` | 关闭 | 为 FVP 快速模式关闭固件 TA。启动时也选择 `-Fast`。 |
| `-Jobs` | `4`，范围 `1–128` | 编译并行数。 |
| `-Target` | `FVP`、`FPGA`，默认 `FVP` | FPGA 须选 `-Mode Static`，使用 TA。 |
| `-LabelsPath` | 随 Demo 选择 | 覆盖手势标签文本或 COCO YAML。 |

### VisionBenchmark

| 参数 | 默认值或可选值 | 使用方式 |
| --- | --- | --- |
| `-Demo` | 必填。`gesture`、`coco`、`img-class` | 选择两端共用的模型和固定图片。该预设启用 TA。 |
| `-Target` | `Both`、`FVP`、`MPS4`，默认 `Both` | 生成一端或两端。MPS4 流程同时生成 `mps4-package.zip`。 |
| `-InputPath` | 手势点赞样例、COCO 样例或 `cat.bmp` | 指定两端共用的输入图片。 |
| `-ModelPath` | 随 Demo 选择 384 KiB 模型 | 覆盖模型。 |
| `-LabelsPath` | 随 Demo 选择 | 覆盖与检测模型对应的标签。 |
| `-CompilerBin` | 自动查找 ATFE | 指定编译和打包用的 ATFE bin。 |
| `-Jobs` | `4`，范围 `1–128` | 编译并行数。 |

### HandPose

| 参数 | 默认值或可选值 | 使用方式 |
| --- | --- | --- |
| `-ModelPath` | 专用 256 KiB Pose PTE | 提供兼容的手部关键点模型。 |
| `-BuildDirectory` | `build/fvp-hand-pose/dedicated-256k` 或 `build/fpga-hand-pose/dedicated-256k` | 覆盖输出目录。 |
| `-Target` | `FVP`、`FPGA`，默认 `FVP` | FVP 使用 semihosting，FPGA 使用内置图片。 |
| `-CompilerBin` | 自动查找 ATFE | 可指定 ATFE bin。 |
| `-Action` | `Configure`、`Build`，默认 `Build` | 配置或构建。 |
| `-ImageSize` | `224`，范围 `32–1024`，须为 32 的倍数 | 设置方形模型输入边长。 |
| `-Jobs` | `4`，范围 `1–128` | 编译并行数。 |

### QrCode

| 参数 | 默认值或可选值 | 使用方式 |
| --- | --- | --- |
| `-ModelPath` | 专用 384 KiB 二维码 PTE | 提供与输入尺寸、cache 配置匹配的模型。 |
| `-BuildDirectory` | `build/fvp-qrcode/dedicated-384k` | 覆盖输出目录。比较不同 cache 时使用独立目录。 |
| `-CompilerBin` | 自动查找 ATFE | 可指定 ATFE bin。 |
| `-Action` | `Configure`、`Build`，默认 `Build` | 配置或构建支持 semihosting 输入的 FVP 固件。 |
| `-Width`、`-Height` | 均为 `256`，范围 `32–1024`，须为 32 的倍数 | 设置模型的输入宽和高。 |
| `-CacheSizeKiB` | `384`，范围 `1–4096` | 设置 Dedicated SRAM 的 cache。 |
| `-Jobs` | `4`，范围 `1–128` | 编译并行数。 |

### Key0Runtime

| 参数 | 默认值或可选值 | 使用方式 |
| --- | --- | --- |
| `-FvpExecutable` | 必填 | 已安装的 Windows Corstone-320 FVP 可执行文件。 |
| `-SdlDll` | 从 Arm DS 目录推导 | 指定 Arm DS `sw/models/bin/SDL2.dll`，供过滤器调用。 |
| `-CompilerBin` | 自动查找 ATFE | 提供 `lld-link.exe`、`llvm-readobj.exe` 所在目录。 |
| `-HostCompiler` | 从 PATH 查找支持 x64 的编译器 | 可指向 MSVC `cl.exe` 或完整 LLVM 的 `clang.exe`。 |
| `-OutputDirectory` | `resources_downloaded/windows/corstone-320-key0` | 本地运行时副本目录。详见 [VSI 指南](../../scripts/vsi/README.md)。 |

## 音频与串口

### AudioDemo

| 参数 | 默认值或可选值 | 使用方式 |
| --- | --- | --- |
| `-UseCase` | `kws_zh`、`asr`、`asr_zh`，默认 `kws_zh` | 选择 FVP 音频用例。 |
| `-InputMode` | `VSI` 或 `STATIC`，默认 `VSI` | 选择流式音频或内置 WAV。 |
| `-Action` | `Configure`、`Build`，默认 `Build` | 配置或构建。 |
| `-ModelPath` | 随用例选 PTE | 覆盖模型。预设为 Dedicated SRAM 256 KiB，`asr_zh` 为 768 KiB。 |
| `-InputPath` | KWS 样本目录或 ASR 默认 WAV | 覆盖内置样本。 |
| `-CompilerBin` | 自动查找 ATFE | 可指定 ATFE bin。 |
| `-BuildDirectory` | `build/fvp-audio/<use-case>`，静态版加 `-static` | 覆盖输出目录。 |
| `-VadRms` | `800`，范围 `1–32767` | 设置英文 `asr` 的 VSI VAD RMS 阈值。 |
| `-SilenceMs` | `1000`，范围 `20–5000` | 设置英文 `asr` 判定句尾所需的静音时长，单位 ms。 |
| `-PreRollMs` | `600`，范围 `200–2000` | 保留英文 `asr` 触发前的音频，单位 ms。 |
| `-Jobs` | `4`，范围 `1–128` | 编译并行数。 |

### FpgaFileDemo

| 参数 | 默认值或可选值 | 使用方式 |
| --- | --- | --- |
| `-UseCase` | 必填。`kws_zh`、`asr`、`asr_zh`、`hand-pose`、`qrcode` | 选择串口接收文件的应用。 |
| `-Target` | `FPGA`、`FVP`，默认 `FPGA` | 选择实板或 FVP 的 UART 固件。 |
| `-BuildDirectory` | `build/<target>-file/<use-case>` | 覆盖输出目录。 |
| `-ModelPath` | 随用例选 PTE | 传入与 [演示配置](demos/README.md) 匹配的模型。 |
| `-CompilerBin` | 自动查找 ATFE | 可指定 ATFE bin。 |
| `-Action` | `Configure`、`Build`，默认 `Build` | 配置或构建。 |
| `-Jobs` | `4`，范围 `1–128` | 编译并行数。 |

### NoiseReduction

| 参数 | 默认值或可选值 | 使用方式 |
| --- | --- | --- |
| `-InputPath` | `resources/noise_reduction/samples/p232_113.wav` | 设置输入文件或目录。 |
| `-ModelPath` | 用例默认的 TFLite 模型 | 覆盖 RNNoise 模型。 |
| `-CompilerBin` | 自动查找 | 指定选定工具链的 bin。 |
| `-BuildDirectory` | `build/fvp-noise-reduction` | 覆盖输出目录。 |
| `-Toolchain` | `AC6` 或 `ATFE`，默认 `ATFE` | 选择交叉编译器。 |
| `-Action` | `Setup`、`Configure`、`Build`、`All`，默认 `Build` | 准备资源、配置或构建。 |
| `-Jobs` | `4`，范围 `1–128` | 编译并行数。 |
| `-CMakeArgs` | 空数组 | 追加用例的 CMake `-D` 覆盖。 |

### KwsZh

| 参数 | 默认值或可选值 | 使用方式 |
| --- | --- | --- |
| `-MemoryModes` | `@('Dedicated_Sram', 'Shared_Sram')` | 可只传一种，也可构建两种模式。 |
| `-Target` | `FPGA`、`FVP`，默认 `FPGA` | 选择实板或 FVP。 |
| `-ModelsRoot` | `resources_downloaded/kws_zh` | Dedicated 读取 `pte_Dedicated_Sram_<KiB>k/wekws_dstcn_u85.pte`，Shared 读取 `pte_Shared_Sram/wekws_dstcn_u85.pte`。 |
| `-InputPath` | `resources/kws_zh/samples` | 覆盖 WAV 样本或目录。 |
| `-BuildRoot` | `build/kws_zh` | 设置各变体共用的构建根目录。 |
| `-BuildTag` | 当天的 `yyyyMMdd` | 添加构建目录后缀。允许字母、数字、下划线和连字符。 |
| `-PackageRoot` | `build/board-test-packages/kws_zh` | 设置上板包根目录。 |
| `-CompilerBin` | 自动查找 ATFE | 可指定 ATFE bin。 |
| `-CacheSizeKiB` | `256`，范围 `1–4096` | 设置 Dedicated cache，并决定选哪个 PTE 子目录。 |
| `-Jobs` | `4`，范围 `1–128` | 编译并行数。 |
| `-DumpOutput` | 关闭 | 开启输出转储以供验证。 |
| `-Package` | 关闭 | 通过 `-Target FPGA` 构建后生成上板包。 |

### AsrZh

| 参数 | 默认值或可选值 | 使用方式 |
| --- | --- | --- |
| `-Target` | `FVP`、`FPGA`，默认 `FVP` | 选择仿真或实板。 |
| `-MemoryMode` | `Dedicated_Sram`、`Shared_Sram`，默认 `Dedicated_Sram` | Dedicated 使用 768 KiB，Shared 使用 INT8 state 变体。 |
| `-Action` | `Configure`、`Build`，默认 `Build` | 配置或构建静态音频固件。 |
| `-ModelPath` | 随内存与 INT8 state 选择 PTE | 覆盖 Zipformer 模型。发布包的模型配合 `-Int8State`，见 [中文 ASR](../use_cases/asr_zh.md)。 |
| `-InputPath` | `resources/asr_zh/samples/demo.wav` | 覆盖内置 WAV 文件或目录。 |
| `-CompilerBin` | 自动查找 ATFE | 可指定 ATFE bin。 |
| `-BuildDirectory` | `build/asr_zh/<target>-<variant>` | 覆盖输出目录。 |
| `-TimingOutput` | `AUTO`、`OFF`、`SUMMARY`、`DETAILED`，默认 `DETAILED` | 设置统计粒度，`OFF` 同时关闭 CPU profiling。 |
| `-Jobs` | `4`，范围 `1–128` | 编译并行数。 |
| `-FixedFeatures` | 关闭 | 使用定点特征提取。 |
| `-Int8State` | 关闭，Shared 自动开启 | 使用 INT8 state 及对应的模型。 |
| `-AllInputs` | 关闭 | 关闭 `USE_SINGLE_INPUT`，使用样本目录中的全部输入。 |
| `-NoValidation` | 关闭 | 跳过内置参考结果验证。 |
| `-NoTensorDump` | 关闭 | 关闭 FVP 张量转储。 |

### AsrStageProfile / ConformerStageProfile

前者构建 TFLM Wav2Letter，后者构建 ExecuTorch Conformer。

| 参数 | 默认值或可选值 | 使用方式 |
| --- | --- | --- |
| `-ModelPath` | 两者均必填 | Wav2Letter 传 `.tflite`，Conformer 传 `.pte`。 |
| `-MemoryMode` | `Dedicated_Sram`、`Shared_Sram`，默认 `Dedicated_Sram` | 选择与模型匹配的内存模式。 |
| `-TimingMode` | 仅 `AsrStageProfile`。`Simple` 或 `Detailed`，默认 `Simple` | 选择简要或详细阶段统计。 |
| `-Target` | `FPGA`、`FVP`，默认 `FPGA` | 选择测试平台。 |
| `-Action` | `Configure`、`Build`，默认 `Build` | 配置或构建。 |
| `-CompilerBin` | 自动查找 ATFE | 可指定 ATFE bin。 |
| `-InputPath` | `resources/asr/samples/another_door.wav` | 更换输入样本。 |
| `-BuildDirectory` | `build/asr-stage/<framework>-<variant>` | 覆盖固件输出目录。 |
| `-PackageDirectory` | `build/board-test-packages/asr-stage/<framework>-<variant>` | 覆盖打包目录。 |
| `-ActivationBufferSize` | 仅 `AsrStageProfile`。默认 `0x00400000` | 设置激活缓冲区字节数。 |
| `-CacheSizeKiB` | `256`，范围 `1–1048576` | 设置 Dedicated SRAM cache。 |
| `-Jobs` | `4`，范围 `1–128` | 编译并行数。 |
| `-CMakeArgs` | 空数组 | 追加用例的 CMake `-D` 选项。 |
| `-Package` | 关闭 | 与 `-Target FPGA -Action Build` 配合生成上板包。 |

### ConformerUart

| 参数 | 默认值或可选值 | 使用方式 |
| --- | --- | --- |
| `-ModelPath` | 必填 | 传入 Dedicated SRAM 256 KiB 的 Conformer PTE。 |
| `-CompilerBin` | 自动查找 ATFE | 可指定 ATFE bin。 |
| `-Action` | `Configure`、`Build`，默认 `Build` | 配置或构建。 |
| `-Target` | `FPGA`、`FVP`，默认 `FPGA` | 选择 UART 固件目标。 |
| `-BaudRate` | `115200`、`460800`、`921600`，默认 `115200` | 设置与主机端一致的串口波特率。 |
| `-BuildDirectory` | `build/asr-uart/dedicated-256k`，FVP 版加 `-fvp` | 覆盖固件目录。 |
| `-PackageDirectory` | `build/board-test-packages/asr-uart/dedicated-256k` | 覆盖打包目录。 |
| `-Jobs` | `4`，范围 `1–128` | 编译并行数。 |
| `-Package` | 关闭 | `-Target FPGA -Action Build` 时打包固件和主机发送工具。 |

## 生成模型与语言模型

### PicoFace

| 参数 | 默认值或可选值 | 使用方式 |
| --- | --- | --- |
| `-Action` | `Export`、`Build`、`Run`、`Package`、`All`，默认 `Build` | 导出、构建、运行、打包或组合流程。`Package` 仅用于 FPGA，`Run` 和 `All` 仅用于 FVP。 |
| `-Toolchain` | `ATFE`、`GNU`，默认 `ATFE` | 选择构建工具链。 |
| `-Target` | `FVP`、`FPGA`，默认 `FVP` | 选择固件目标。 |
| `-MemoryMode` | `Dedicated_Sram`、`Shared_Sram`，默认 `Dedicated_Sram` | Dedicated 使用 384 KiB cache，Shared 使用另一模型变体。 |
| `-Variant` | `m3_long_cfg`、`m3_decD_deep_full`，默认前者 | 选择生成模型。 |
| `-CompilerBin` | 自动查找 | 覆盖目标交叉编译器目录。 |
| `-ExportPython` | 参考工程的 Python 环境 | 指定可用于模型导出的 Python。 |
| `-Checkpoints` | 随模型变体选择 | 指定 checkpoint 目录。 |
| `-BuildDirectory` | 由工具链、目标、内存和变体决定 | 覆盖固件输出路径。 |
| `-FvpExecutable` | 参考工程的 FVP 路径 | 覆盖 `Run`、`All` 使用的 FVP。单独启动时可用 `run-fvp.ps1 -Profile PicoFace`。 |
| `-UartImage` | 关闭 | 在 FPGA 上通过 UART 输出生成的图像。 |
| `-Preview` | 关闭 | FVP 逐步预览，适用于 `Build`、`Run`、`All`，固定 8 步。 |
| `-Headless` | 关闭 | 需配合 `-Preview`，以无界面方式接收结果。 |
| `-PreviewPython` | `python` | 指定主机预览用的 Python。 |
| `-TimeoutSeconds` | `180`，范围 `1–3600` | FVP 逐步预览的超时时间。 |
| `-Jobs` | `4` | 编译并行数。 |
| `-Seed` | `3`，无符号 32 位整数 | 设置生成图片的随机种子。 |
| `-Steps` | `1`、`2`、`4`、`8`，默认 `4` | 设置推理步数。 |
| `-ClassId` | `1`，范围 `0–4` | 选择生成类别。 |
| `-Guidance` | `4.0`，范围 `0–100` | 设置生成引导强度。 |

模型和导出工具的准备步骤见 [PicoFace 用例](../use_cases/pico_face.md)。

### TinyStoriesFvp / TinyStoriesFpga / TinyStoriesTest

| 参数 | 默认值或可选值 | 使用方式 |
| --- | --- | --- |
| `-Target` | `TinyStoriesFpga`、`TinyStoriesTest` 可设 `FPGA` 或 `FVP`，默认 `FPGA` | `TinyStoriesFvp` 固定为 FVP。 |
| `-Mode` | 仅 `TinyStoriesFpga` 可设 `DEMO` 或 `TEST`，默认 `DEMO` | 其余两个预设分别固定为 `DEMO` 和 `TEST`。 |
| `-CompilerBin` | 预设的 Arm DS ATFE bin 路径 | 按本机的安装位置覆盖。 |
| `-BuildDirectory` | `build/tinystories42m/<mode>/<target>` | 覆盖固件目录。 |
| `-ModelPath` | 预设的 ctx128 PTE | 传入与上下文和 INT16 KV 兼容的 PTE。 |
| `-Context` | `128`，范围 `1–1024` | 选择上下文 token 数，须和模型匹配。 |
| `-Jobs` | `8`，范围 `1–64` | 编译并行数。 |
| `-TimingAdapter` | 关闭 | 启用 Ethos-U TA。 |
| `-LogitDump` | 关闭 | 输出 logits 以供一致性验证。 |

### MiniMind2Test

| 参数 | 默认值或可选值 | 使用方式 |
| --- | --- | --- |
| `-Target` | `FPGA`、`FVP`，默认 `FVP` | 选择固件目标。 |
| `-CompilerBin` | 预设的 Arm DS ATFE bin 路径 | 按本机的安装位置覆盖。 |
| `-BuildDirectory` | `build/minimind2_small/test/<target>` | 覆盖固件目录。 |
| `-ModelPath` | 省略时由用例决定 | 选择预编译 PTE。128 token 路线必填。 |
| `-CacheOffsets` | 省略时由用例决定 | 提供逗号分隔的 KV 状态偏移字符串。128 token 路线必填，偏移须对应当前 PTE。 |
| `-ContextLength` | `64`、`128`，默认 `64` | 设置上下文。128 token 模式须同时传 `-ModelPath`、`-CacheOffsets`、`-Int16Kv`。 |
| `-TimingAdapter` | 关闭 | 启用 Ethos-U TA。 |
| `-ProfileStages` | 关闭 | 输出应用分阶段统计。 |
| `-Int16Kv` | 关闭 | 使用 INT16 KV 处理，须与模型匹配。 |
| `-Demo` | 关闭 | 构建交互演示。 |
| `-FastDemo` | 关闭 | 需配合 `-Target FVP -Demo`，使用 Release。TA 由 `-TimingAdapter` 单独控制。 |
| `-Jobs` | `8`，范围 `1–64` | 编译并行数。 |

模型变体和偏移示例见 [MiniMind2 用例](../../source/app/use_case/minimind2_small/README.md)。
