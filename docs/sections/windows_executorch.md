# Windows 编译与 Arm DS 操作教程（TFLM / ExecuTorch）

<!-- SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

本仓库保持 CMake 构建。Windows 上在 Arm DS 图形界面调用 PowerShell 脚本，
脚本调用 CMake/Ninja 和交叉编译器，Arm DS 再加载生成的 AXF 调试。
所有示例源码、构建、虚拟环境、pip 缓存和临时文件都放在 D 盘。
Windows/Arm DS 本身的已安装程序可以位于 C 盘，不在 C 盘创建构建目录。

## 1. 支持范围与当前状态

| 内容 | 使用方式 |
| --- | --- |
| TFLM / .tflite | `scripts/windows/build.ps1`，AC6 或 ATFE |
| ExecuTorch / .pte | `scripts/windows/build-executorch.ps1`，GNU 或 ATFE |
| AC6 + ExecuTorch | 上游 CMake 明确不支持，不能只改编译器名称绕过 |
| PyTorch .pt / .pth | checkpoint，需要模型专用导出代码先导出 .pte；本构建脚本不自动转换 |
| YOLOv8 | 原 Alif API 调用已按要求删除，保留禁用入口，等待本项目重新实现 |

ExecuTorch 新脚本是原生 Windows 流程，不在后台启动 WSL。
它使用官方 ExecuTorch 1.3.1 Windows wheel，要求依赖源码也是 1.3.1。
它独立安装 Python 环境，不调用现有仅面向 Linux/macOS 的 ExecuTorch
资源安装/导出流程。这里区分“安装 Python 包”“交叉编译 AXF”和“运行推理”；
安装成功并不能证明后两步成功。

环境验证：PowerShell 语法检查、固定版本环境的 pip check、
ExecuTorch 序列化模块导入及 flatc 24.3.25 检查通过。
测试过的非固定版本 PyTorch 2.14.0 遇到 `WinError 4551`；
本脚本固定的 PyTorch 2.12.0 导入成功，不需要更改系统安全策略。
交叉编译和运行验收结果见本文末尾，不应从环境安装结果推断运行成功。

## 2. VS Code / WSL 提交与 Windows 分支同步

开发仓库：`/home/xx/ml-embedded-evaluation-kit`。
在 VS Code 的 Remote - WSL 窗口打开它，不要使用另一个 Alif 仓库。
本次检查时当前分支是 `main`，Windows 基础脚本已经存在于 main。

在 VS Code 终端检查修改，并对实际需要提交的文件使用 Source Control 的“+”暂存：

```bash
cd /home/xx/ml-embedded-evaluation-kit
git branch --show-current
git status --short
git diff --check
git diff
```

确认暂存内容后（包括新脚本、新教程和清理后的 YOLOv8 占位目录）：

```bash
git diff --cached
git commit -s -m "Add native Windows ExecuTorch build entry"
git push origin main
```

以上假设本次修改在 main；如果实际开发分支不同，替换成实际分支名。
不要提交 build、resources_downloaded、虚拟环境或客户私有模型。
VS Code 的“推送”对应 `git push`；不确定远端状态时先 Fetch 和检查差异。

保留独立 Windows 发布分支时，在干净工作区执行：

```bash
git fetch origin
git switch codex/windows-armds
git merge main
git submodule update --init --recursive
# 解决冲突、检查脚本和 Windows 编译后，再推送
git push origin codex/windows-armds
```

如果该分支不存在，用 `git switch -c codex/windows-armds main` 创建，
然后 `git push -u origin codex/windows-armds`。
合并冲突需要人工检查；不要用强制推送覆盖。每次 merge 后仍需验证 Windows 构建，
Git 合并成功本身不保证新增 Linux 代码兼容 Windows。

Windows 首次克隆：

```powershell
git clone --recurse-submodules --branch codex/windows-armds https://github.com/Yoru3103/ml-embedded-evaluation-kit.git D:/work/mlek-client
Set-Location D:/work/mlek-client
code .
```

已有 checkout 后每次更新：

```powershell
Set-Location D:/work/mlek-client
git status --short
git fetch origin
git switch codex/windows-armds
git pull --ff-only
git submodule update --init --recursive
```

新脚本会给 ExecuTorch 1.3.1 的 third-party/CMakeLists.txt 应用一个局部补丁，
复用 wheel 自带的 Windows flatc，避免错误构建 Unix 主机工具。
补丁保存在主仓库 scripts/windows/patches/executorch-host-flatc.patch，
不需要另建 ExecuTorch 远程仓库。Windows 构建后子模块出现工作区修改是预期现象，
不要为它提交新的 gitlink。在升级子模块前，先在 Windows checkout 根目录撤销：

```powershell
git -C dependencies/executorch apply --reverse --check ../../scripts/windows/patches/executorch-host-flatc.patch
git -C dependencies/executorch apply --reverse ../../scripts/windows/patches/executorch-host-flatc.patch
```

只在补丁已经应用时执行撤销；未应用时跳过。随后更新仓库和子模块，
下次运行脚本会检查版本并重新应用。若有其他用户修改或补丁上下文不匹配，
脚本报错停止，不覆盖这些修改。

有本地修改时先提交或保存，再拉取。不要把 D 盘上的构建验证副本误认为 Git checkout。

## 3. Windows 前置条件

- Git for Windows，Python 3.12 x64（`py -3.12 --version`）。
- GNU Arm bare-metal 工具链：`arm-none-eabi-gcc`，Cortex-M85 要求 GCC 13.2.1 或更新；
  不是 MinGW 的 x86 GCC，也不是 aarch64-linux-gnu。
- 或使用已安装的 ATFE：
  `D:/Program Files/Arm/Development Studio 2026.0-1/sw/atfe-pro22.1.0/bin`。
- Arm DS 与 SSE-320 FVP，用于 AXF 加载和调试。
- 正常的系统 DLL 加载策略。若 PyTorch DLL 被安全策略拦截，先让管理员核查并批准
  所用官方软件包；脚本不改变 Smart App Control、WDAC 或其他系统安全设置。

GNU 安装路径由用户通过 `-CompilerBin` 明确指定，也可以设置当前终端的
`$env:MLEK_GNU_BIN`。新脚本不会自动下载或替换全局编译器。
可从 [Arm 官方安装指南](https://learn.arm.com/install-guides/gcc/arm-gnu)
进入下载页，选择 Windows 主机、arm-none-eabi 目标的 ZIP 便携包并解压到 D 盘。
源码放在 D 盘本地短路径，不能直接使用 WSL 的 UNC 路径执行此脚本。

## 4. 准备 ExecuTorch 环境

```powershell
Set-Location D:/work/mlek-client
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build-executorch.ps1 -Action Setup
```

生成目录：

```text
resources_downloaded/windows-executorch/env
resources_downloaded/windows-executorch/tmp
resources_downloaded/windows-executorch/pip-cache
```

Setup 固定安装 ExecuTorch 1.3.1、PyTorch 2.12.0、torchao 0.17.0、
CMake 3.30.5、Ninja 1.11.1.3 和本仓库 mlek-tools。
安装后执行 pip check、ExecuTorch 序列化模块导入和 flatc 检查；
任一步失败即停止，不进入编译。

如果出现 SSLKEYLOGFILE 指向不可写的 `C:/nss_ssl_sfagent.log`，
在仅用于本次安装的 PowerShell 窗口中执行 `$env:SSLKEYLOGFILE=$null` 后重试。
这只停用该进程的 TLS 密钥日志，不关闭证书验证。

## 5. 准备 PTE 模型

将已导出的模型放到例如 `D:/models/mv2-u85-z256-dedicated.pte`。
文件名仅为示例，不是脚本自动下载的资源；必须替换成实际存在的文件。

编译 img_class 时必须使用与 img_class 相符的分类模型：
输入尺寸、布局、数据类型、归一化、输出类别及标签均要一致。
不能拿 YOLO 检测模型替代分类模型来验证检测功能。

默认配置：

| Target | 硬件配置 | 默认内存模式 |
| --- | --- | --- |
| FVP | SSE-320 / U85 Z256 | Dedicated_Sram |
| FPGA | MPS4 SSE-320 / U85 Z1024 | Dedicated_Sram |

模型必须预先按对应 NPU 配置导出。修改 `-Target` 或 `-MemoryMode` 不会重新
编译 PTE 内嵌的 NPU 指令。Shared_Sram 模型必须同时指定 `-MemoryMode Shared_Sram`。
模型导出还依赖模型定义、量化/校准数据和 Arm backend/Vela；
本脚本只负责 PTE 到固件的构建，不声称已完成通用 PT→PTE Windows 导出。

## 6. 编译 ExecuTorch

GNU 路径示例（替换为真实安装目录）：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build-executorch.ps1 -Action Build -Toolchain GNU -CompilerBin "D:/tools/arm-gnu/bin" -Target FVP -UseCase img_class -ModelPath "D:/models/mv2-u85-z256-dedicated.pte" -BuildDirectory D:/build/mlek-et-gnu-fvp -Jobs 8
```

使用现有 ATFE：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build-executorch.ps1 -Action Build -Toolchain ATFE -Target FVP -UseCase img_class -ModelPath "D:/models/mv2-u85-z256-dedicated.pte" -BuildDirectory D:/build/mlek-et-atfe-fvp -Jobs 8
```

- Setup：只安装和检查 Python 工具，不要求 PTE 或交叉编译器。
- Configure：检查环境和 PTE，运行 CMake 配置及代码生成。
- Build：运行 Configure 后增量编译，模型替换会重新生成 C 数组。
- All：Setup + Configure + Build，日常编译无需反复使用。

成功后输出形如：

```text
D:/build/mlek-et-atfe-fvp/bin/mlek_img_class.axf
```

可选 `-InputPath` 指定样本文件/目录，`-LabelsPath` 指定标签文本。
这些参数映射到 usecase 的 FILE_PATH / LABELS_TXT_FILE；其他 usecase
必须定义对应选项。高级用户可从 PowerShell 通过数组 `-CMakeArgs` 传递其他
CMake 选项，例如 `@('-Dimg_class_IMAGE_SIZE=224')`。
不要用高级参数覆盖脚本的编译器、框架、平台、Python 或构建目录配置。

GNU、ATFE、FVP、FPGA 和不同 usecase 分别使用独立构建目录。
新脚本会拒绝 C 盘构建和已检测到的编译器/框架/目标缓存不匹配。

## 7. 在 Arm DS 图形界面点击构建

打开 `Run → External Tools → External Tools Configurations...`，
新建 Program，填写：

- Name：`MLEK ExecuTorch img_class FVP`
- Location：`C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe`
- Working Directory：`D:\work\mlek-client`
- Arguments（单行）：

```text
-NoProfile -ExecutionPolicy Bypass -File "D:/work/mlek-client/scripts/windows/build-executorch.ps1" -Action Build -Toolchain ATFE -Target FVP -UseCase img_class -ModelPath "D:/models/mv2-u85-z256-dedicated.pte" -BuildDirectory D:/build/mlek-et-atfe-fvp -Jobs 8
```

点击 Apply、Run，在 Console 查看日志。以后从 External Tools 下拉菜单选择该配置。
第一次先单独运行 Setup，也可以创建一个相同入口、参数仅含 `-Action Setup` 的配置。
这绑定的是 External Tools，不是工程的默认锤子按钮。
切换到 GNU 时替换 Toolchain，补充 CompilerBin，并换独立 BuildDirectory。

## 8. Arm DS 加载 FVP 调试

从 Arm DS 的 New Debug Connection 向导创建 Model Connection；
不同版本也可能从 Debug Configurations 进入。
选择 SSE-320 FVP；如果列表中不存在，使用 Add a new model 添加已安装模型：

```text
D:/Program Files/Arm/FVP_Corstone_SSE-320/models/Win64_VC2019/FVP_Corstone_SSE-320.exe
```

选择 Cortex-M85 bare-metal 调试，在 Files/Application 页面指定刚生成的 AXF。
加载程序与符号，设置 main 断点，启动 Debug，再用 Continue/Step 操作。
不要用 FPGA 的 AXF 加载 FVP，也不要把分区 .bin 当作包含调试符号的 AXF。

External Tool 构建和 Debug Connection 是两个配置；点击 Debug 不代表它已经
重新执行 External Tool。每次修改源码/模型后先确认 Build 成功再 Debug。
GNU/ATFE 编译出的 Arm ELF 可以作为调试输入，但具体加载、内存布局和运行结果
仍需要用当前模型平台验收。

## 9. TFLM 流程保持原入口

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build.ps1 -Action Setup -UseCase img_class
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build.ps1 -Action Build -Toolchain AC6 -Target FVP -UseCase img_class -BuildDirectory D:/build/mlek-tflm-ac6-fvp
```

替换模型添加 `-ModelPath D:/models/your-model.tflite`，要求模型已按目标配置准备。
Arm DS External Tool 同样调用该脚本即可。TFLM 不使用 ExecuTorch 虚拟环境。
更详细的 AC6/ATFE 操作见 [Windows Arm DS 基础教程](windows_armds.md)。

## 10. 常见错误

- `WinError 4551` / `torch_python.dll`：Windows 应用控制拦截，
  不是 CMake 参数问题；交给管理员检查 CodeIntegrity/Operational 事件。
- 找不到 flatc.exe：Setup 未完成，或 Python 环境安装不完整。
- 只安装 GNU 仍无法配置：ExecuTorch 的主机 Python/flatc 也参与代码生成，
  必须先通过 Setup 检查。
- 源码版本与 wheel 不一致：同步 submodule，不能任意升级 pip 包后直接编译旧源码。
- PTE 的 NPU 或内存配置不匹配：按目标重新导出模型，换名称不能解决。
- YOLOv8 被拒绝：当前有意禁用，等待新 API 设计和模型接入。
- Setup 成功但没有 AXF：Setup 本身不会编译，需再运行 Build。

## 11. 本机实测结果（2026-09-22）

已通过 Windows 原生 ATFE Pro 22.1.0 / ExecuTorch 1.3.1 / SSE-320 FVP
的 img_class 编译、链接和一次 NPU 推理。使用 U85 Z256 / Shared_Sram 模型；
这项结果不代表 Dedicated_Sram、FPGA 实板或重新设计前的 YOLOv8 已验收。

- AXF：`D:/build/mlek-et-atfe-fvp/bin/mlek_img_class.axf`
- 环境日志：`D:/work/mlek-windows/setup-et.log`
- 构建日志：`D:/work/mlek-windows/build-et-atfe.log`
- 最终链接日志：`D:/work/mlek-windows/link-et-atfe.log`
- FVP 日志：`D:/work/mlek-windows/run-et-atfe.log`
- 推理次数：1；Top-1：281，tabby, tabby cat；NPU ACTIVE：3062718 cycles。
- 应用日志：`Main loop terminated successfully.`
- 重复运行 Build 成功，已应用的子模块补丁通过反向检查，不会重复修改。

GNU 参数入口和路径处理已提供，但本次没有完成 GNU 编译验收。
官方便携工具链下载未完成，已停止下载，部分 ZIP 保留在
`D:/tools/mlek-downloads/arm-gnu-14.3.zip`，不能直接解压使用。
当前先采用已验证的 ATFE 命令。本文提供的 GNU 命令属于待验证配置。

SSE-320 LLVM 链接脚本额外修复了 libc++ override 段边界，以及 ATFE 22
picolibc 的 combined initializer 边界；否则可能分别出现链接失败和后端未注册。
FVP 通过 UART EOT 停止时进程退出码不一定为零，本次验收依据应用成功日志、
分类输出和 NPU 计数，不仅依赖进程退出码。
