# Windows ExecuTorch 构建与 Arm Development Studio 调试

<!-- SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Arm ML Embedded Evaluation Kit 提供在 Cortex-M CPU 和 Ethos-U NPU 上运行机器学习用例的应用框架。本页说明在原生 Windows 上将已导出的 ExecuTorch `.pte` 模型编入 MPS4 SSE-320 FPGA 或 SSE-320 FVP 固件，再用 Arm Development Studio（Arm DS）加载 AXF。TFLM 的 `.tflite` 构建见 [Windows TFLM 与 Arm DS 指南](windows_armds.md)。

## 准备环境和模型

在 Windows 本地磁盘上准备项目源码及完整依赖、Python 3.12（含 `py` 启动器）、GNU Arm Embedded 工具链或 Arm Toolchain for Embedded（ATFE）。GNU 工具链应包含 `arm-none-eabi-gcc.exe`、`arm-none-eabi-g++.exe`、`arm-none-eabi-objcopy.exe`；ATFE 应包含 `clang.exe`、`llvm-ar.exe`、`llvm-objcopy.exe`。此脚本不支持 AC6。调试还需 Arm DS，以及 SSE-320 FVP 或 MPS4 板卡与调试连接。

使用仓库锁定的 ExecuTorch 子模块版本。`-ModelPath` 指向预先导出的 `.pte`。Pose
的 Dedicated SRAM 构建也可使用下文的默认模型路径。构建前应先导出 checkpoint，
使 PTE 的输入输出、Ethos-U 配置和内存模式与固件一致。

## 构建参数

`scripts/windows/build.ps1 -Framework pte` 安装构建工具、配置 CMake、生成模型代码并
交叉编译固件。框架共用构建目录命名规则、缓存和 PMU 等选项以及 `build-info.json`。
并排构建示例见 [二维码复现指南](qrcode_reproduction.md)。

构建可能给 ExecuTorch 子模块应用补丁，以复用 Python 包中的 Windows `flatc.exe`。
此时 `dependencies/executorch` 会显示本地修改。脚本退出时恢复临时环境变量。

完整的参数、默认值和双平台示例见 [Windows 构建参数手册](windows_build_options.md#firmware)。

脚本固定 `TARGET_PLATFORM=mps4`、`TARGET_SUBSYSTEM=sse-320`、`ML_FRAMEWORK=executorch` 和 `USE_SINGLE_INPUT=ON`。构建类型默认 Debug，可通过 `-BuildType Release` 选择 Release。`-Target FVP` 设置 `FPGA_PLATFORM_SSE_320=OFF`，默认 U85 Z1024；`-Target FPGA` 设置为 `ON`，使用 U85 Z1024。模型若按其他 NPU 配置导出，需通过 `-CMakeArgs` 指定对应的 `ETHOS_U_NPU_CONFIG_ID`。修改目标或内存模式不会重新导出 PTE。FVP 模型的 MAC 数也必须设为 1024；如果使用 Z256 模型，须显式覆盖 `ETHOS_U_NPU_CONFIG_ID=Z256` 并设置对应的 FVP MAC 数。尺寸、类别数等覆盖参数仅在用例声明相应 CMake 选项时使用。

脚本每次配置会清除 `SRAM_*`、`EXT_*` 的旧 TA 缓存，然后加载所选配置；自定义 TA 寄存器参数请通过 `-CMakeArgs` 传入，例如 `-DEXT_MAXR=48`，不要只依赖旧构建目录的缓存。直接运行 CMake 切换 TA 配置时，使用新构建目录或添加 `-U "SRAM_*" -U "EXT_*"`，再传入需要的 `-D` 覆盖。

`yolov8_pose` 默认使用 `ETHOS_U_NPU_CACHE_SIZE=262144`（256 KiB）和 `TA_CONFIG_FILE=ta_config_u85_sys_dram_1024`，也可通过 `-CMakeArgs` 显式覆盖。Dedicated SRAM 的默认模型位于 `resources_downloaded/yolov8_pose/yolo_hand_pose_u85-1024_ded_256KiB.pte`；FVP 和 FPGA 默认选用这份模型，固件分别构建。Shared SRAM 必须显式提供匹配的 `-ModelPath` 和内存配置。

## 安装、配置与构建

在项目根目录打开 PowerShell，首次执行 Setup（需要网络连接）：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build.ps1 -Framework pte `
    -Action Setup
```

Setup 将共用的 CMake/Ninja 安装在 `resources_downloaded/windows-tools/env`，将 ExecuTorch/PyTorch 安装在 `resources_downloaded/windows-executorch/env`，并检查 Python 包和 `flatc.exe`。TFLM 使用单独的 Python 环境。

以下是 GNU、FVP、`img_class` 的 Debug 构建示例；替换为本机实际编译器目录和 PTE：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build.ps1 -Framework pte `
    -Action Build -Toolchain GNU -CompilerBin "path-to-arm-gnu/bin" `
    -Target FVP -UseCase img_class -ModelPath "path-to-your-pte"
```

使用 ATFE 构建 MPS4 FPGA：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build.ps1 -Framework pte `
    -Action Build -Toolchain ATFE -Target FPGA -UseCase img_class `
    -ModelPath "D:/models/img-class-u85-z1024.pte" `
    -BuildDirectory "build/et-atfe-fpga-img-class" -Jobs 8
```

`yolov8_detection` 等自定义用例可传入图片、标签、尺寸、类别数和激活区大小。例如 PTE 按 FVP 的 U85 Z1024、Dedicated SRAM 导出时：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build.ps1 -Framework pte `
    -Action Build -Toolchain ATFE -Target FVP -UseCase yolov8_detection `
    -ModelPath "D:/models/gesture-u85-z1024.pte" `
    -InputPath "resources/gesture_detection/samples" `
    -LabelsPath "resources/gesture_detection/labels.txt" `
    -ImageSize 320 -NumClasses 10 -ActivationBufferSize 0x00300000 `
    -MemoryMode Dedicated_Sram -CMakeArgs "-DETHOS_U_NPU_CONFIG_ID=Z1024" `
    -BuildDirectory "build/et-atfe-fvp-gesture-z1024"
```

模型路径只是填写示例，应使用实际存在且与用例兼容的文件。AXF 位于 `<BuildDirectory>/bin/mlek_<UseCase>.axf`。模型变更后重新执行 Build，脚本会先运行 CMake 配置并重新嵌入 PTE。

## Arm DS 配置

项目导入方式与 TFLM 相同，按 [TFLM 文档的“导入项目”](windows_armds.md#导入项目) 操作即可。

在 `Run → External Tools → External Tools Configurations...` 中新建 `Program`：

| 字段 | FVP 示例 |
| --- | --- |
| Name | `MLEK ExecuTorch GNU FVP Debug` |
| Location | `C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe` |
| Working Directory | 项目根目录，例如 `D:/work/mlek-client`；已导入 Arm DS 工程时也可用 `${workspace_loc:/mlek-client}`。 |
| Arguments | 下方单行参数；替换本机编译器和 PTE 路径。 |

```text
-NoProfile -ExecutionPolicy Bypass -File "scripts/windows/build.ps1" -Framework pte -Action Build -Toolchain GNU -CompilerBin "path-to-arm-gnu/bin" -Target FVP -UseCase img_class -ModelPath "path-to-your-pte" -BuildDirectory "build/et-gnu-fvp-img-class" -Jobs 4
```

点击 `Apply`、`Run`，在 Console 中确认 AXF 构建完成。首次使用可另建 Setup 配置，Arguments 为 `-NoProfile -ExecutionPolicy Bypass -File "scripts/windows/build.ps1" -Framework pte -Action Setup`。为 ATFE、FPGA 或其他用例复制配置，并同步修改名称、模型和构建目录。

调试连接和 trace 的设置也与 TFLM 相同，直接参阅 [加载 AXF 调试](windows_armds.md#加载-axf-调试) 和 [MPS4 实板调试](windows_armds.md#mps4-实板调试)。选择 ExecuTorch 本次构建生成的 AXF，并确保构建目标、PTE 的 NPU 配置及调试连接一致。

FVP 的启动参数应与固件配置一致：U85 Z1024 使用 `-C mps4_board.subsystem.ethosu.num_macs=1024`。在 Arm DS 的模型连接配置中设置同名参数；CMake 的 MPS4 FVP 集成测试会根据 `ETHOS_U_NPU_CONFIG_ID` 自动传入 MAC 数。

## YOLOv8 Pose 构建与调试

先执行 `build.ps1 -Framework pte -Action Setup` 准备环境，再手动把新后端导出的模型放入
`resources_downloaded/yolov8_pose/`。该目录已被 Git 忽略，Setup 不下载或导出 Pose 模型。
图片和标签保留在 `resources/yolov8_pose/`；两个旧的 `dedicated_128k` PTE 已移除。

默认模型为 `yolo_hand_pose_u85-1024_ded_256KiB.pte`，对应 U85 Z1024、Dedicated SRAM、
256 KiB 缓存和 `Ethos_U85_SYS_DRAM_1024` system config。Dedicated SRAM 构建不传
`-ModelPath` 时使用此模型，显式传参优先。

### 单次调用：一个模型构建 FVP 固件

在项目根目录的 PowerShell 中执行：

```powershell
$fvp = 'C:/Program Files/Arm/FVP_Corstone_SSE-320/models/Win64_VC2019/FVP_Corstone_SSE-320.exe'
$compilerBin = 'C:/Program Files/Arm/Development Studio 2026.0-1/sw/atfe-pro22.1.0/bin'
$model = '.\resources_downloaded\yolov8_pose\yolo_hand_pose_u85-1024_ded_256KiB.pte'
.\scripts\windows\build.ps1 -Framework pte -Action Build -Toolchain ATFE -Target FVP `
    -UseCase yolov8_pose -CompilerBin $compilerBin -ModelPath $model `
    -MemoryMode Dedicated_Sram `
    -BuildDirectory build/windows-ATFE-et-FVP-yolov8_pose-Dedicated_Sram-256k `
    -CMakeArgs @('-DCPU_PROFILE_ENABLED=ON', '-DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=ON',
        '-DBUILD_FVP_TESTS=ON', "-DFVP_PATH=$fvp")
```

一次 Build 依次配置 CMake、生成并嵌入模型、编译固件。上面的默认模型可以省略
`-ModelPath`。运行 FVP 是单独一步：

```powershell
& .\resources_downloaded\windows-tools\env\Scripts\ctest.exe `
    --test-dir .\build\windows-ATFE-et-FVP-yolov8_pose-Dedicated_Sram-256k `
    -R mlek_yolov8_pose_fvp_test -V
```

FVP 测试检查 `YOLOv8 pose use case terminated successfully`，超时为 300 秒。
运行时打印 TA 寄存器回读；实际配置以构建目录的
`generated/ta/timing_adapter_settings.h` 为准。

### 单次调用：一个模型构建 FPGA 固件

使用上面定义的 `$compilerBin` 和 `$model`：

```powershell
.\scripts\windows\build.ps1 -Framework pte -Action Build -Toolchain ATFE -Target FPGA `
    -UseCase yolov8_pose -CompilerBin $compilerBin -ModelPath $model `
    -MemoryMode Dedicated_Sram `
    -BuildDirectory build/windows-ATFE-et-FPGA-yolov8_pose-Dedicated_Sram-256k `
    -CMakeArgs @('-DCPU_PROFILE_ENABLED=ON', '-DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=ON')
```

AXF 位于各自的 `<BuildDirectory>/bin/mlek_yolov8_pose.axf`；FPGA 的板上镜像在
`bin/sectors/`。构建完成后按 Arm DS / MPS4 流程加载对应固件，构建命令不会烧板。
FVP 与 FPGA 可以使用这份相同的 Z1024 Dedicated PTE，但固件地址不同，不能互换 AXF。
在 Arm DS 的 FVP 连接中将模型 MAC 数设为 1024。

### 批量调用：多个缓存配置分别构建 FVP 和 FPGA

循环调用现有入口即可。以下选择本地已有的 Z1024 Dedicated 模型，缓存大小随模型一起
传入；每个模型与目标使用独立构建目录。

```powershell
$ErrorActionPreference = 'Stop'
$compilerBin = 'C:/Program Files/Arm/Development Studio 2026.0-1/sw/atfe-pro22.1.0/bin'
$sizesKiB = @(64, 128, 256, 384, 768)
$targets = @('FVP', 'FPGA')
foreach ($size in $sizesKiB) {
    $model = ".\resources_downloaded\yolov8_pose\yolo_hand_pose_u85-1024_ded_${size}KiB.pte"
    if (-not (Test-Path -LiteralPath $model)) { throw "Missing model: $model" }
    foreach ($target in $targets) {
        .\scripts\windows\build.ps1 -Framework pte -Action Build -Toolchain ATFE -Target $target `
            -UseCase yolov8_pose -CompilerBin $compilerBin -ModelPath $model `
            -MemoryMode Dedicated_Sram `
            -BuildDirectory "build/pose-et-$target-ded-${size}k" `
            -CMakeArgs @("-DETHOS_U_NPU_CACHE_SIZE=$($size * 1024)",
                '-DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=ON')
    }
}
```

`$sizesKiB` 可改为 `@(256)`，`$targets` 可只保留 `FVP` 或 `FPGA`。流程是先准备环境和
模型，逐个配置、编译，最后分别测试或加载。上述批量命令只构建；自动 FVP 测试需要在
FVP 调用中追加 `-DBUILD_FVP_TESTS=ON`、`-DFVP_PATH=...`，再对对应目录运行 CTest。
Shared SRAM 与 Z512 模型需要单独匹配内存和 NPU 配置，不包含在这个 Dedicated Z1024 循环中。

迁移后首次构建使用新目录，避免缓存旧模型路径。切换模型不会重新导出 PTE。
默认 PTE 输入逻辑形状为 FP32 `[1,3,224,224]`，物理 dim_order 为 `[0,2,3,1]`；
EtTensor 按实际布局报告 NHWC，预处理按交错 RGB 写入。四个 FP32 输出形状分别为
`[1,4,1029]`、`[1,1,1029]`、`[1,42,1029]`、`[1,21,1029]`，应用默认使用 SPLIT 输出、
像素坐标和 21 个手部关键点。更换输出约定时显式覆盖相应 Pose CMake 选项。

TA 生成配置在 `generated/ta/timing_adapter_settings.h`，FVP 的 SRAM/EXT 地址为
`0x50006000/0x50006800`，FPGA 为 `0x51102000/0x51102400`。
原 `source/app/CMakeLists.txt` 每次配置已清理生成目录，样本暂存沿用该机制；
项目目录和生成目录不应作为输入样本目录。

## 在 Z1024 板上比较 U85 的 512 / 1024 PTE

模型生成默认要求 PTE 与固件的 NPU 配置一致。对已确认能在目标板运行的 U85 PTE，
可通过 `-CMakeArgs` 追加 `-DALLOW_PTE_MAC_MISMATCH=ON`，允许模型和固件使用不同的
MAC 数。该选项默认关闭，仅对 PTE 的 U85 MAC 数差异生效；*目前为测试项，未成功上板*
在 Z1024 板上测量 512 PTE 时，保持 `ETHOS_U_NPU_CONFIG_ID=Z1024`；
Release / O3 设置与该开关独立。

## Pose 的周期日志与 PMU

Windows 构建脚本和预设默认启用 `CPU_PROFILE_ENABLED=ON`。两种运行时对每张图片分别打印 `Inference`、`Pre-processing`、`Post-processing` 的统计；推理测量包围 `model.RunInference()`，不包括图片显示、预处理、后处理和结果打印。

`CPU TOTAL` 是该测量区间的 CPU 时钟周期，包括驱动调用和等待 NPU 的时间，并不等于 CPU 实际执行算子的周期。`NPU TOTAL` 是 NPU 提交区间的周期累计，`NPU ACTIVE` 是活动事件计数，`NPU IDLE = NPU TOTAL - NPU ACTIVE`。IDLE 不包含 NPU 提交区间之外的主机空闲时间。前后处理阶段没有 NPU 提交时，NPU 周期为 0 是正常结果。CPU 和 NPU 的计数区间不同，不能相减来计算 CPU 算子耗时。

默认每张图片预热 0 次、测量 1 次。可通过 `-CMakeArgs` 设置：

```powershell
-CMakeArgs @('-Dyolov8_pose_PMU_WARMUP_RUNS=1',
    '-Dyolov8_pose_PMU_MEASURED_RUNS=3', '-DETHOS_U_NPU_PMU_GROUP=C')
```

预热不进入统计。日志沿用 alif 工程的单值格式，例如 `CPU TOTAL: 3688120 cycles`、`NPU IDLE: 525 cycles` 和 `DURATION: 115 milliseconds`，不显示样本数及 Total / Avg. / Min / Max。默认测量 1 次时显示本次测量值；显式配置多次测量时显示平均值，前后处理仍各测量一次。输出在每张图片结束时重置，不会混入上一张图片的结果。上述统计默认输出，不依赖详细日志宏。

U85 的 `ETHOS_U_NPU_PMU_GROUP` 支持 `A`（默认，计算活动/停顿）、`B`、`B_SRAM0`、`B_SRAM1`（SRAM）、`BW`（SRAM/外存混合带宽）、`C`（外存）、`D`（事务/接口启用）和 `LATENCY`（延迟分布）。各组保留 ACTIVE、IDLE、TOTAL，硬件一次最多采样 8 个事件；比较不同组时需重新构建并使用相同模型、输入、TA 和预热配置。LATENCY 另可设置 `ETHOS_U_NPU_PMU_AXI_SEL=0/1`（SRAM/EXT）及 `ETHOS_U_NPU_PMU_CHANNEL=1/2/4/5/8/9`，默认 0/1。

`BW` 在同一次推理中采集 `NPU_ACTIVE`、SRAM/EXT 的读写数据 beats、
`SRAM_RD_TRAN_REQ_STALLED`、`EXT_RD_TRAN_REQ_STALLED` 和 `AO_STALLED_BY_OB`。
两条路径的事件分别对各自所有端口求和，不要再次乘端口数。
读请求 stall 表示地址请求握手被阻塞，不包含读响应延迟或写阻塞；多端口求和的
stall 计数不能直接当作总耗时的百分比。该组由共享 HAL 实现，TFLM 也可使用。
使用 `-CMakeArgs @('-DETHOS_U_NPU_PMU_GROUP=BW')` 选择它。
对于 128-bit AXI 数据接口，平均读/写带宽为 `beats * 16 * NPU频率 / NPU_TOTAL`，
单位为字节/秒；应使用实际 NPU 频率。该值描述推理期间的总线流量率，不能作为
内存峰值能力。启用 TA 时，结果包含 TA 的时延和带宽限制。

FVP 的计数用于模型验证；与 FPGA 的真实测量应分别保存。当前验证采用 Debug 构建，性能不能代表 Release。
