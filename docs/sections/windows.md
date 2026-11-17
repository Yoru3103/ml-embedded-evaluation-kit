# 原生 Windows 构建与运行

<!-- SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

在 Windows PowerShell 中使用 `scripts/windows/build.ps1` 准备环境及构建固件，
使用 `scripts/windows/run-fvp.ps1` 启动 Corstone-320 FVP。切换模型或输入方式时，
选择对应的 `-Profile`。以下命令从仓库根目录运行。

## 首次运行

1. 安装 Git、64 位 [Python 3.12](https://www.python.org/downloads/)（含 Windows `py` 启动器）、
   [Arm Toolchain for Embedded（ATfE）](https://github.com/arm/arm-toolchain/releases)
   和 Windows 版 [Corstone-320 FVP](deployment.md#installing-an-fvp)。
2. 将本项目克隆到 Windows 本地磁盘并初始化子模块：

   ```powershell
   git clone --recurse-submodules https://github.com/Yoru3103/ml-embedded-evaluation-kit.git
   Set-Location ml-embedded-evaluation-kit
   py -3.12 --version
   ```

   已有克隆可在仓库根目录执行 `git submodule update --init --recursive`。

3. 构建并运行静态图像分类示例：

   ```powershell
   .\scripts\windows\build.ps1 -Profile Vision -Demo img-class -Mode Static -Action All
   .\scripts\windows\run-fvp.ps1 -Profile Vision -Demo img-class -Mode Static `
       -Headless -SmokeTimeoutSeconds 300
   ```

   `-Action All` 安装 CMake、Ninja 和项目 Python 依赖，下载并编译 MobileNetV2 模型，
   再用仓库中的 `resources/img_class/samples/cat.bmp` 构建固件。首次配置需要联网。
   固件位于 `build/fvp-img-class/static/bin/mlek_img_class.axf`。运行命令在检测到分类输出后
   自动退出，日志保存在同一构建目录。该流程无需摄像头、麦克风或 FVP 的 Python 扩展。

构建找不到编译器时，用 `-CompilerBin "D:/Arm/ATfE/bin"` 指向含 `clang.exe` 的目录。
启动脚本在常见安装目录查找 FVP。也可传入 `-FvpExecutable`，或把 `$env:FVP_PATH`
设为 `FVP_Corstone_SSE-320.exe` 的完整路径。

如果当前 PowerShell 的执行策略阻止脚本，使用以下形式在单独的进程中执行：
`powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build.ps1 <参数>`。
启动脚本可用同样的形式。

## 选择构建和运行方式

```powershell
.\scripts\windows\build.ps1 -ListProfiles
.\scripts\windows\run-fvp.ps1 -ListProfiles
.\scripts\windows\build.ps1 -Profile Vision -ProfileHelp
.\scripts\windows\run-fvp.ps1 -Profile Vision -ProfileHelp
```

`Firmware` 是通用构建 profile，可通过 `-Framework tflm` 或 `-Framework pte`、`-Target FVP`
或 `-Target FPGA` 选择框架和目标。运行此方式生成的 Corstone-320 FVP 固件时，
将实际构建目录传给 `-Profile Firmware -BuildDirectory`：

```powershell
.\scripts\windows\build.ps1 -Profile Firmware -Framework tflm -Target FVP `
    -Toolchain ATFE -UseCase img_class -Action All -BuildDirectory build/windows-img-class
.\scripts\windows\run-fvp.ps1 -Profile Firmware -BuildDirectory build/windows-img-class
```

注册在 `resources/use_case_resources.json` 中的 TFLM 用例可在 Setup 时下载默认模型。
ExecuTorch 的 Setup 准备构建工具和 Python 环境。构建 PTE 用例时，还需提供与应用输入输出、
NPU 型号、内存模式和缓存预算匹配的已编译模型。各演示指南列出所需模型和传入方式。

- [统一构建参数表与完整示例](windows_build_options.md)
- [Windows 演示索引与 FPGA 文件上传](demos/README.md)
- [TFLM 构建参数、Arm DS 与 MPS4 调试](windows_armds.md)
- [ExecuTorch 构建参数、模型与 Arm DS 调试](windows_executorch.md)
- [摄像头与 VSI 按键演示](../../scripts/vsi/README.md)
