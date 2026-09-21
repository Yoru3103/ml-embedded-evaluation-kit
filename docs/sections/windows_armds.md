# Windows / Arm Development Studio 开发与同步

<!-- SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

## 分支和支持范围

基于官方库建立 `codex/windows-armds`，主线为 `main`。
`origin` 是自己的 GitHub 仓库，`upstream` 是 Arm 官方仓库。
日常业务功能进入 main，Windows 分支定期合并 main 并重新验证。

本分支提供原生 Windows 的 TFLM 构建入口：
VS Code 编辑 → CMake/Ninja → Arm 编译器 → AXF → Arm DS 加载调试。
AC6 6.24 与 ATfE Pro 22.1 是不同编译器，不得共用 CMakeCache 或混用链接器。
编译、链接成功不代表已经通过 MPS4 实板运行验收。

当前官方 `scripts/cmake/executorch.cmake` 明确拒绝 ARMClang；
`scripts/py/mlek_tools/setup/install.py` 明确不支持 Windows ExecuTorch 安装。
本分支不宣称支持 Windows ExecuTorch/PTE 全流程，相关迁移需要单独验证。

## 安装与克隆

正式客户 checkout 使用 Windows 本地路径，如 `D:/work/mlek-client`，避免 UNC 或 WSL 共享路径。
本机 `D:/work/mlek-windows` 是本次测试使用的源码验证副本，不含 Git 元数据；
它用于编译验证，不能直接执行 git pull。发布分支后按下面命令建立正式 checkout。
尽量使用短路径。验证中较长的 Downloads 路径触发了 Ninja 的路径长度限制，
表现为 `build.ninja still dirty after 100 tries`。可将源码放到上述短路径，
或通过脚本的 `-BuildDirectory D:/build/mlek-ac6` 指定独立的短构建目录。
启用需要的 Arm 编译器许可证，安装 Git for Windows 和 Python 3.12（包含 py 启动器）。

```powershell
git clone --recurse-submodules --branch codex/windows-armds https://github.com/Yoru3103/ml-embedded-evaluation-kit.git D:/work/mlek-client
cd D:/work/mlek-client
```

上述命令需要先将已提交的 Windows 分支推送到 origin。
初次准备一个 use case（需要联网下载依赖和模型）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build.ps1 -Action Setup -UseCase img_class
```

脚本将 Windows Python、CMake 3.30.5 和 Ninja 安装在
`resources_downloaded/windows/env`，模型下载到 `resources_downloaded/windows`，
不会使用 WSL 的 env。新增其他 use case 时，再以相应名称运行 Setup。
Python/pip 临时文件和缓存也放在该 checkout 的 `resources_downloaded/windows` 下，
脚本退出后恢复当前进程原来的环境变量。默认构建目录位于 D 盘 checkout 的 `build` 下。
ExecutionPolicy Bypass 仅作用于该次进程，不修改机器策略。

若本机 Python 的 ensurepip 报错指向不可写的 `C:/nss_ssl_sfagent.log`，检查
当前会话的 `SSLKEYLOGFILE`。本次验证在子进程中清除了该日志变量后成功安装；
没有关闭 TLS 证书校验，也没有修改系统环境变量。

## 构建与模型替换

AC6 默认路径：`D:/Program Files/ArmCompilerforEmbedded6.24/bin`。
ATfE 默认路径：`D:/Program Files/Arm/Development Studio 2026.0-1/sw/atfe-pro22.1.0/bin`。
其他机器用 `-CompilerBin` 指定安装目录，或设置
`MLEK_AC6_BIN` / `MLEK_ATFE_BIN` 环境变量。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build.ps1 -Toolchain AC6 -UseCase img_class
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build.ps1 -Toolchain ATFE -UseCase img_class
```

两个 preset 分别是 `windows-ac6-mps4`、`windows-atfe-mps4`。
默认 Debug 便于 Arm DS 调试，MPS4 SSE-320 实板配置包含
`FPGA_PLATFORM_SSE_320=ON`，由官方配置选择 Cortex-M85、U85 Z1024、
Dedicated_Sram。保留 use case 的输入、标签、预处理、activation 默认设置。
Debug 的性能不应用来代表 Release 性能。

换入已由 Vela 按相同 NPU/内存配置生成的模型：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build.ps1 -Toolchain AC6 -UseCase img_class -ModelPath D:/models/new_model_vela.tflite
```

此版本正确的 CMake 参数是 `img_class_MODEL_PATH`，不是
`img_class_MODEL_TFLITE_PATH`。模型代码在 configure 阶段生成，所以脚本每次
先 configure 再 build；仅运行 build 不足以保证同路径替换模型后重新嵌入。

ModelPath 会保存在该 build 目录的 CMakeCache。省略参数不会自动清除以前的覆盖；
恢复默认时使用新 build 目录，或 configure 时传 `-Uimg_class_MODEL_PATH`。
新模型仍需满足输入输出、标签和算子约束，并验证内存用量。

也可直接调用 CMake（在仓库根目录执行）：

```powershell
$env:MLEK_AC6_BIN = 'D:/Program Files/ArmCompilerforEmbedded6.24/bin'
$cmake = './resources_downloaded/windows/env/Scripts/cmake.exe'
& $cmake --preset windows-ac6-mps4 -DUSE_CASE_BUILD=img_class
& $cmake --build --preset windows-ac6-mps4 --parallel 4
```

ATfE 使用 `MLEK_ATFE_BIN` 和 `windows-atfe-mps4`。
如需 FVP，使用脚本的 `-Target FVP`，它会选择独立的 `windows-*-fvp` preset，
自动指定 `FPGA_PLATFORM_SSE_320=OFF`，并使用 FVP 默认的 U85 Z256 配置。
不得直接把实板固件作为 FVP 固件。示例：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build.ps1 `
    -Toolchain AC6 -Target FVP -UseCase img_class `
    -BuildDirectory D:/work/mlek-client/build/windows-ac6-fvp
```

ATfE 将 `-Toolchain AC6` 改为 `-Toolchain ATFE`，输出目录改为
`windows-atfe-fvp`。直接使用 CMake 时，分别使用
`windows-ac6-fvp` 或 `windows-atfe-fvp` preset。

生成 AXF 后，在 PowerShell 中启动 SSE-320 FVP：

```powershell
$fvp = 'D:/Program Files/Arm/FVP_Corstone_SSE-320/models/Win64_VC2019/FVP_Corstone_SSE-320.exe'
$axf = 'D:/work/mlek-client/build/windows-ac6-fvp/bin/mlek_img_class.axf'
& $fvp -a $axf `
  -C mps4_board.telnetterminal0.start_telnet=0 `
  -C mps4_board.uart0.out_file=- `
  -C mps4_board.uart0.shutdown_on_eot=1 `
  -C mps4_board.visualisation.disable-visualisation=1 `
  -C vis_hdlcd.disable_visualisation=1
```

FVP 的 `-a` 参数表示加载 AXF；如果要使用 Arm DS 调试器，则把同一个 AXF
配置到 Arm DS 的 FVP debug launch 中，由 Arm DS 启动 FVP 和连接调试器。

## Arm DS 中构建和调试

将本地源码目录加入 Arm DS 的现有代码/外部构建工程，关闭自动生成 Makefile，
将 Build 命令设置为 `powershell.exe`，参数例如：

```text
-NoProfile -ExecutionPolicy Bypass -File scripts/windows/build.ps1 -Toolchain AC6 -UseCase img_class
```

FVP use case 的 Arm DS Build 参数为：

```text
-NoProfile -ExecutionPolicy Bypass -File scripts/windows/build.ps1 -Toolchain AC6 -Target FVP -UseCase img_class
```

工作目录为仓库根目录。不要在 Arm DS 中再维护一份不同的源文件和链接选项。
按实际版本的 IDE 外部构建设置填写；本分支不提供机器相关的 DS workspace。

实板生成文件位于 `build/windows-ac6-mps4/bin` 或
`build/windows-atfe-mps4/bin`；FVP 生成文件位于对应的
`build/windows-ac6-fvp/bin` 或 `build/windows-atfe-fvp/bin`。
在 Arm DS 的 FVP Debug Configuration 中选择 FVP AXF 作为程序和符号文件，
让 Arm DS 使用 SSE-320 FVP 模型；实板 AXF 不应放入 FVP 配置。
未实际连接板卡时不能声称已验证下载或运行。

## 后续同步

首次发布前查看差异，运行检查，再提交和推送（DCO sign-off）：

```bash
cd /home/xx/ml-embedded-evaluation-kit
git status
git diff
git add CMakePresets.json scripts/cmake/source_gen_utils.cmake scripts/cmake/windows-presets.json scripts/windows/build.ps1 scripts/py/mlek_tools/setup/util.py scripts/py/mlek_tools/setup/python_venv.py scripts/py/mlek_tools/optimize/tflite.py tests/windows docs/sections/windows_armds.md
git commit -s -m "Add native Windows Arm DS build workflow"
git push -u origin codex/windows-armds
```

后续在 main 完成功能并提交推送后，从干净的工作区执行：

```bash
git fetch origin
git switch codex/windows-armds
git merge origin/main
git submodule sync --recursive
git submodule update --init --recursive
# 在 Windows checkout 验证 configure/build 和模型替换，必要时解决冲突并提交
git push origin codex/windows-armds
```

客户更新（先提交或另存客户自己的修改）：

```powershell
git switch codex/windows-armds
git pull --ff-only origin codex/windows-armds
git submodule sync --recursive
git submodule update --init --recursive
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build.ps1 -Toolchain AC6 -UseCase img_class
```

若依赖版本有变，重新运行 Setup。merge 成功只说明文本合并成功；
需重新验证 Windows 构建、替换模型及实板行为后才能发布。
Windows 通用修复稳定后可以合回 main，以减少长期分支维护成本。

## 回归检查

### 本次验证结果（2026-09-21）

基于 main 的 f2f6247，在 D:/work/mlek-windows 的验证副本中测试：

| 项目 | 结果 |
| --- | --- |
| Python 3.12.10 / CMake 3.30.5 / Ninja 1.11.1.3 | 原生 Windows 环境准备通过 |
| Vela 5.1.0 / U85 Z1024 / Dedicated_Sram | 模型生成及已有报告覆盖通过 |
| AC6 6.24 / TFLM img_class / MPS4 SSE-320 FPGA / Debug | 完整编译、链接、二进制生成通过 |
| ATfE Pro 22.1.0 / 相同配置 | 完整编译、链接、二进制生成通过 |
| ATfE / 带空格的 ModelPath | 模型代码重新生成及增量链接通过 |
| Python 参数回归、CMake 标签生成 | Windows / Linux 均通过 |
| PyLint 3.3.8 / 修改的 Python 文件及新测试 | 通过，10.00/10 |
| 错误工具链复用已有 build 目录 | 在运行 CMake 前拒绝，通过 |

模型路径测试使用默认模型的副本，不代表任何客户新模型的精度验收。
未测试实板下载、推理运行、其他 use case 或 Windows ExecuTorch。
产物为两个 preset 构建目录内的 `bin/mlek_img_class.axf`；
本机完整日志为 `D:/work/mlek-windows/build-ac6.log`、`build-atfe.log`，
模型路径覆盖测试日志为 `model-override-atfe.log`。

### 重跑检查

```powershell
$python = './resources_downloaded/windows/env/Scripts/python.exe'
$cmake = './resources_downloaded/windows/env/Scripts/cmake.exe'
& $python -m unittest discover -s tests/windows -p test_command_arguments.py
& $cmake "-DPYTHON=$((Resolve-Path $python).ProviderPath)" -P tests/windows/source_generator_smoke.cmake
```

Linux 上也应执行参数传递测试以及 `cmake --list-presets`，保证原有 preset 仍可用。
如果没有安装 mlek-tools，可从仓库根目录使用
`PYTHONPATH=scripts/py python3 -m unittest discover -s tests/windows`。
