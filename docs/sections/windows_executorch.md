# Windows ExecuTorch 构建与 Arm Development Studio 调试

<!-- SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Arm ML Embedded Evaluation Kit 提供在 Cortex-M CPU 和 Ethos-U NPU 上运行机器学习用例的应用框架。本页说明在原生 Windows 上将已导出的 ExecuTorch `.pte` 模型编入 MPS4 SSE-320 FPGA 或 SSE-320 FVP 固件，再用 Arm Development Studio（Arm DS）加载 AXF。TFLM 的 `.tflite` 构建见 [Windows TFLM 与 Arm DS 指南](windows_armds.md)。

## 准备环境和模型

在 Windows 本地磁盘上准备项目源码及完整依赖、Python 3.12（含 `py` 启动器）、GNU Arm Embedded 工具链或 Arm Toolchain for Embedded（ATFE）。GNU 工具链应包含 `arm-none-eabi-gcc.exe`、`arm-none-eabi-g++.exe`、`arm-none-eabi-objcopy.exe`；ATFE 应包含 `clang.exe`、`llvm-ar.exe`、`llvm-objcopy.exe`。此脚本不支持 AC6。调试还需 Arm DS，以及 SSE-320 FVP 或 MPS4 板卡与调试连接。

脚本要求 `dependencies/executorch` 为 1.3.1。`-ModelPath` 必须指向预先导出的 `.pte`；脚本不会将 `.pt` / `.pth` checkpoint 转换为 `.pte`。模型的用例、输入输出、Ethos-U 配置和内存模式须与固件一致。

## `build-executorch.ps1` 的功能与参数

`scripts/windows/build-executorch.ps1` 安装和检查 Windows 工具，配置 CMake，生成模型代码并交叉编译固件。构建前会给 ExecuTorch 1.3.1 的 `third-party/CMakeLists.txt` 应用项目提供的补丁，以复用 Python 包中的 Windows `flatc.exe`；因此 `dependencies/executorch` 会出现本地修改。脚本结束时恢复临时修改的进程环境变量。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-Action` | `Build` | `Setup` 安装并检查工具；`Configure` 只配置和生成；`Build` 先配置再编译；`All` 执行全部步骤。首次使用先执行 Setup。 |
| `-Toolchain` | `GNU` | `GNU` 或 `ATFE`；不支持 AC6。 |
| `-Target` | `FVP` | `FVP` 为 SSE-320 虚拟平台；`FPGA` 为 MPS4 SSE-320 实板。 |
| `-UseCase` | `img_class` | 单个用例名，映射到 `USE_CASE_BUILD`，例如 `yolov8_detection`。 |
| `-CompilerBin` | 自动查找 | 编译器 `bin` 目录；优先于 `MLEK_GNU_BIN` / `MLEK_ATFE_BIN`、`PATH` 和常见安装目录。 |
| `-ModelPath` | 无，构建时必填 | 已导出的 `.pte` 文件，映射到 `<UseCase>_MODEL_PATH`；Setup 不需要。 |
| `-InputPath` | 用例默认输入 | 样本文件或目录，映射到 `<UseCase>_FILE_PATH`。`yolov8_detection` 未指定时使用 `resources/gesture_detection/samples`。 |
| `-LabelsPath` | 用例默认标签 | 标签文件，映射到 `<UseCase>_LABELS_YAML_FILE`。 |
| `-ImageSize` | 不覆盖 | 图像尺寸，允许 1–4096；映射到 `<UseCase>_IMAGE_SIZE`。 |
| `-NumClasses` | 不覆盖 | 类别数，允许 1–1000；映射到 `<UseCase>_NUM_CLASSES`。 |
| `-ActivationBufferSize` | 不覆盖 | 激活区大小，传给 `<UseCase>_ACTIVATION_BUF_SZ`。 |
| `-BuildDirectory` | `build/windows-<Toolchain>-et-<Target>-<UseCase>-<MemoryMode>` | 构建目录；相对路径从项目根目录解析。不同编译器、目标、用例和内存模式使用独立目录。 |
| `-MemoryMode` | `Dedicated_Sram` | `Dedicated_Sram` 或 `Shared_Sram`，映射到 `ETHOS_U_NPU_MEMORY_MODE`；须与 PTE 一致。 |
| `-Jobs` | `4` | 编译并行数，允许 1–128。 |
| `-CMakeArgs` | 空 | 额外的 `-DNAME=value` 定义，例如 `-DETHOS_U_NPU_CONFIG_ID=Z1024`；不能覆盖脚本控制的编译器、框架、平台、资源路径、用例和内存模式等选项。 |

脚本固定 `TARGET_PLATFORM=mps4`、`TARGET_SUBSYSTEM=sse-320`、`ML_FRAMEWORK=executorch`、`USE_SINGLE_INPUT=ON` 和 `CMAKE_BUILD_TYPE=Debug`。**Debug 默认开启**，脚本没有单独的 Debug 参数。`-Target FVP` 设置 `FPGA_PLATFORM_SSE_320=OFF`，默认 U85 Z1024；`-Target FPGA` 设置为 `ON`，使用 U85 Z1024。模型若按其他 NPU 配置导出，需通过 `-CMakeArgs` 指定对应的 `ETHOS_U_NPU_CONFIG_ID`。修改目标或内存模式不会重新导出 PTE。FVP 模型的 MAC 数也必须设为 1024；如果使用 Z256 模型，须显式覆盖 `ETHOS_U_NPU_CONFIG_ID=Z256` 并设置对应的 FVP MAC 数。尺寸、类别数等覆盖参数仅在用例声明相应 CMake 选项时使用。

脚本每次配置会清除 `SRAM_*`、`EXT_*` 的旧 TA 缓存，然后加载所选配置；自定义 TA 寄存器参数请通过 `-CMakeArgs` 传入，例如 `-DEXT_MAXR=48`，不要只依赖旧构建目录的缓存。直接运行 CMake 切换 TA 配置时，使用新构建目录或添加 `-U "SRAM_*" -U "EXT_*"`，再传入需要的 `-D` 覆盖。

`yolov8_pose` 默认使用 `ETHOS_U_NPU_CACHE_SIZE=131072` 和 `TA_CONFIG_FILE=ta_config_u85_sys_dram_1024`，也可通过 `-CMakeArgs` 显式覆盖。FVP 与 FPGA 使用不同 PTE；构建时仍须通过 `-ModelPath` 指定对应文件。

## 安装、配置与构建

在项目根目录打开 PowerShell，首次执行 Setup（需要网络连接）：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build-executorch.ps1 `
    -Action Setup
```

Setup 将共用的 CMake/Ninja 安装在 `resources_downloaded/windows-tools/env`，将 ExecuTorch/PyTorch 安装在 `resources_downloaded/windows-executorch/env`，并检查 Python 包和 `flatc.exe`。TFLM 使用单独的 Python 环境。

以下是 GNU、FVP、`img_class` 的 Debug 构建示例；替换为本机实际编译器目录和 PTE：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build-executorch.ps1 `
    -Action Build -Toolchain GNU -CompilerBin "path-to-arm-gnu/bin" `
    -Target FVP -UseCase img_class -ModelPath "path-to-your-pte"
```

使用 ATFE 构建 MPS4 FPGA：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build-executorch.ps1 `
    -Action Build -Toolchain ATFE -Target FPGA -UseCase img_class `
    -ModelPath "D:/models/img-class-u85-z1024.pte" `
    -BuildDirectory "build/et-atfe-fpga-img-class" -Jobs 8
```

`yolov8_detection` 等自定义用例可传入图片、标签、尺寸、类别数和激活区大小。例如 PTE 按 FVP 的 U85 Z1024、Dedicated SRAM 导出时：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build-executorch.ps1 `
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
-NoProfile -ExecutionPolicy Bypass -File "scripts/windows/build-executorch.ps1" -Action Build -Toolchain GNU -CompilerBin "path-to-arm-gnu/bin" -Target FVP -UseCase img_class -ModelPath "path-to-your-pte" -BuildDirectory "build/et-gnu-fvp-img-class" -Jobs 4
```

点击 `Apply`、`Run`，在 Console 中确认 AXF 构建完成。首次使用可另建 Setup 配置，Arguments 为 `-NoProfile -ExecutionPolicy Bypass -File "scripts/windows/build-executorch.ps1" -Action Setup`。为 ATFE、FPGA 或其他用例复制配置，并同步修改名称、模型和构建目录。

调试连接和 trace 的设置也与 TFLM 相同，直接参阅 [加载 AXF 调试](windows_armds.md#加载-axf-调试) 和 [MPS4 实板调试](windows_armds.md#mps4-实板调试)。选择 ExecuTorch 本次构建生成的 AXF，并确保构建目标、PTE 的 NPU 配置及调试连接一致。

FVP 的启动参数应与固件配置一致：U85 Z1024 使用 `-C mps4_board.subsystem.ethosu.num_macs=1024`。在 Arm DS 的模型连接配置中设置同名参数；CMake 的 MPS4 FVP 集成测试会根据 `ETHOS_U_NPU_CONFIG_ID` 自动传入 MAC 数。

## YOLOv8 Pose 最简调试

Pose 用例支持 ExecuTorch 和 TFLM，沿用原 Windows 脚本和链接脚本，并接入 CPU/NPU 周期统计及可选 U85 PMU 事件组。在工程根目录的 PowerShell 中执行（必要时先打开 `powershell -NoProfile -ExecutionPolicy Bypass` 会话）：

```powershell
$fvp = 'C:/Program Files/Arm/FVP_Corstone_SSE-320/models/Win64_VC2019/FVP_Corstone_SSE-320.exe'
$compilerBin = 'C:/Program Files/Arm/Development Studio 2026.0-1/sw/atfe-pro22.1.0/bin'
.\scripts\windows\build-executorch.ps1 -Action Build -Toolchain ATFE -Target FVP `
    -UseCase yolov8_pose -CompilerBin $compilerBin `
    -ModelPath .\resources\yolov8_pose\yolo_hand_pose_dedicated_128k_ethos_u85_1024.pte `
    -CMakeArgs @('-DCPU_PROFILE_ENABLED=ON', '-DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=ON',
        '-DBUILD_FVP_TESTS=ON', "-DFVP_PATH=$fvp")
& .\resources_downloaded\windows-tools\env\Scripts\ctest.exe `
    --test-dir .\build\windows-ATFE-et-FVP-yolov8_pose-Dedicated_Sram `
    -R mlek_yolov8_pose_fvp_test -V
```

FVP 测试检查 `YOLOv8 pose use case terminated successfully`，超时为 300 秒。运行时打印 TA 寄存器回读，并核对 MAXR/MAXW/MAXRW、读写延迟、脉冲和带宽上限；不将动态计数器或 FVP 固定的 MODE 值作为配置一致性断言。当前配置的 EXT_MAXR 为 63，不再使用被 6 位寄存器掩码成 0 的 64。

FPGA 只进行编译和链接验证：

```powershell
.\scripts\windows\build-executorch.ps1 -Action Build -Toolchain ATFE -Target FPGA `
    -UseCase yolov8_pose -CompilerBin $compilerBin `
    -ModelPath .\resources\yolov8_pose\yolo_hand_pose_dedicated_128k_MPS4_ethos_u85_1024.pte `
    -CMakeArgs @('-DCPU_PROFILE_ENABLED=ON', '-DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=ON')
```

各自的 AXF 位于构建目录下 `bin/mlek_yolov8_pose.axf`。Arm DS 载入 FVP 版本并将模型 MAC 数设为 1024；FPGA 版本不应在 FVP 上运行。TA 生成配置位于 `generated/ta/timing_adapter_settings.h`，FVP 的 SRAM/EXT 地址为 `0x50006000/0x50006800`，FPGA 为 `0x51102000/0x51102400`。

Pose 的默认 PTE 输入逻辑形状为 `[1,3,224,224]`，物理 dim_order 为 `[0,2,3,1]`；EtTensor 按实际布局报告 NHWC，预处理按交错 RGB 写入。输出为四个 FP32 张量，使用像素坐标和 21 个手部关键点。

原 `source/app/CMakeLists.txt` 每次配置已经清理生成目录，样本暂存沿用该机制。项目目录和生成目录不应作为输入样本目录。

## Pose 的周期日志与 PMU

Windows 构建脚本和预设默认启用 `CPU_PROFILE_ENABLED=ON`。两种运行时对每张图片分别打印 `Inference`、`Pre-processing`、`Post-processing` 的统计；推理测量包围 `model.RunInference()`，不包括图片显示、预处理、后处理和结果打印。

`CPU TOTAL` 是该测量区间的 CPU 时钟周期，包括驱动调用和等待 NPU 的时间，并不等于 CPU 实际执行算子的周期。`NPU TOTAL` 是 NPU 提交区间的周期累计，`NPU ACTIVE` 是活动事件计数，`NPU IDLE = NPU TOTAL - NPU ACTIVE`。IDLE 不包含 NPU 提交区间之外的主机空闲时间。前后处理阶段没有 NPU 提交时，NPU 周期为 0 是正常结果。CPU 和 NPU 的计数区间不同，不能相减来计算 CPU 算子耗时。

默认每张图片预热 0 次、测量 1 次。可通过 `-CMakeArgs` 设置：

```powershell
-CMakeArgs @('-Dyolov8_pose_PMU_WARMUP_RUNS=1',
    '-Dyolov8_pose_PMU_MEASURED_RUNS=3', '-DETHOS_U_NPU_PMU_GROUP=C')
```

预热不进入统计。日志沿用 alif 工程的单值格式，例如 `CPU TOTAL: 3688120 cycles`、`NPU IDLE: 525 cycles` 和 `DURATION: 115 milliseconds`，不显示样本数及 Total / Avg. / Min / Max。默认测量 1 次时显示本次测量值；显式配置多次测量时显示平均值，前后处理仍各测量一次。输出在每张图片结束时重置，不会混入上一张图片的结果。上述统计默认输出，不依赖详细日志宏。

U85 的 `ETHOS_U_NPU_PMU_GROUP` 支持 `A`（默认，计算活动/停顿）、`B`、`B_SRAM0`、`B_SRAM1`（SRAM）、`C`（外存）、`D`（事务/接口启用）和 `LATENCY`（延迟分布）。各组保留 ACTIVE、IDLE、TOTAL，硬件一次最多采样 8 个事件；比较不同组时需重新构建并使用相同模型、输入、TA 和预热配置。LATENCY 另可设置 `ETHOS_U_NPU_PMU_AXI_SEL=0/1`（SRAM/EXT）及 `ETHOS_U_NPU_PMU_CHANNEL=1/2/4/5/8/9`，默认 0/1。

FVP 的计数用于模型验证；与 FPGA 的真实测量应分别保存。当前验证采用 Debug 构建，性能不能代表 Release。
