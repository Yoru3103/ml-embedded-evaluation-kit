# Windows 编译与 Arm DS 操作教程（TFLM / ExecuTorch）

<!-- SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

本仓库保持 CMake 构建。Windows 上在 Arm DS 图形界面调用 PowerShell 脚本，
脚本调用 CMake/Ninja 和交叉编译器，Arm DS 再加载生成的 AXF 调试。
仓库和构建目录可以位于任意可写的本地磁盘；脚本从自身位置推导仓库根目录，
不依赖固定盘符。本文中的 `D:/work/mlek-client` 只是示例路径。

## 1. 支持范围与当前状态

| 内容 | 使用方式 |
| --- | --- |
| TFLM / .tflite | `scripts/windows/build.ps1`，AC6 或 ATFE |
| ExecuTorch / .pte | `scripts/windows/build-executorch.ps1`，GNU 或 ATFE |
| AC6 + ExecuTorch | 上游 CMake 明确不支持，不能只改编译器名称绕过 |
| PyTorch .pt / .pth | checkpoint，需要模型专用导出代码先导出 .pte；本构建脚本不自动转换 |
| YOLOv8 | `build.ps1` 和 `build-executorch.ps1` 均支持，模型/图片/标签通过参数传入 |

ExecuTorch 新脚本是原生 Windows 流程，不在后台启动 WSL。
它使用官方 ExecuTorch 1.3.1 Windows wheel，要求依赖源码也是 1.3.1。
它独立安装 Python 环境，不调用现有仅面向 Linux/macOS 的 ExecuTorch
资源安装/导出流程。这里区分“安装 Python 包”“交叉编译 AXF”和“运行推理”；
安装成功并不能证明后两步成功。

环境验证：PowerShell 语法检查、Python 环境的 pip check、
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

确认暂存内容后（包括新脚本、新教程和 YOLOv8 适配代码）：

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

有本地修改时先提交或保存，再拉取。不要把其他目录中的构建验证副本误认为 Git checkout。

## 3. Windows 前置条件

- Git for Windows，Python 3.12 x64（`py -3.12 --version`）。
- GNU Arm bare-metal 工具链：`arm-none-eabi-gcc`，Cortex-M85 要求 GCC 13.2.1 或更新；
  不是 MinGW 的 x86 GCC，也不是 aarch64-linux-gnu。
- 或使用 Arm Development Studio 安装的 ATFE。
- Arm DS 与 SSE-320 FVP，用于 AXF 加载和调试。
- 正常的系统 DLL 加载策略。若 PyTorch DLL 被安全策略拦截，先让管理员核查并批准
  所用官方软件包；脚本不改变 Smart App Control、WDAC 或其他系统安全设置。

工具链目录按以下顺序解析：`-CompilerBin`、`MLEK_AC6_BIN` / `MLEK_ATFE_BIN` /
`MLEK_GNU_BIN`、当前 `PATH`、各本地磁盘上的常见 Arm 安装目录。因此脚本不绑定
Arm DS 的具体版本号或安装盘符。新脚本不会自动下载或替换全局编译器。
可从 [Arm 官方安装指南](https://learn.arm.com/install-guides/gcc/arm-gnu)
进入下载页，选择 Windows 主机、arm-none-eabi 目标的 ZIP 便携包并解压到本地工具目录。
源码应放在 Windows 本地路径，不能直接使用 WSL 的 UNC 路径执行此脚本。

## 4. 准备 ExecuTorch 环境

```powershell
Set-Location D:/work/mlek-client
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/windows/build-executorch.ps1 -Action Setup
```

生成目录：

```text
resources_downloaded/windows-tools/env
resources_downloaded/windows-executorch/env
resources_downloaded/windows-executorch/tmp
resources_downloaded/windows-executorch/pip-cache
```

`windows-tools/env` 是 TFLM 和 ExecuTorch 共用的唯一 CMake/Ninja 环境。
任一 Windows 构建脚本执行 Setup 时都会运行 `pip install --upgrade cmake ninja`，
不固定精确版本；框架环境中旧的 CMake/Ninja 会被卸载。两个框架的 Python 环境仍然分开：

- TFLM：`resources_downloaded/windows/env`
- ExecuTorch/PyTorch：`resources_downloaded/windows-executorch/env`

ExecuTorch Setup 仍固定安装相互匹配的 ExecuTorch 1.3.1、PyTorch 2.12.0、
torchao 0.17.0 和本仓库 mlek-tools。
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

### 5.1 为什么必须区分 FVP 和 MPS4 FPGA

两者都使用 `TARGET_PLATFORM=mps4`、`TARGET_SUBSYSTEM=sse-320` 和 Cortex-M85，
但 FVP 是 Corstone-320 的软件仿真模型，MPS4 是实际 FPGA 原型板。脚本中的
`-Target FVP` 会设置 `FPGA_PLATFORM_SSE_320=OFF`，`-Target FPGA` 则设置为 `ON`。
这个选择不只是 timing adapter 开关，而是决定固件使用哪一套硬件地址和默认配置：

| 项目 | FVP | MPS4 FPGA |
| --- | --- | --- |
| UART0 基地址 | `0x58203000` | `0x59303000` |
| ML 临时内存基地址 | `0x94000000`（按动态区计算） | `0x72000000` |
| HDLCD framebuffer 基地址 | `0x96000000` | `0x74000000` |
| U85 默认 MAC 配置 | Z256，可通过 CMake 覆盖 | 固定 Z1024 |
| 平台编译定义 | 无 `FPGA` | `FPGA=1` |

timing adapter 的 SRAM/EXT 寄存器基地址也不同。除此之外，FVP 还可以使用
VSI 摄像头、semihosting 和 FVP 自动化测试；实板通常使用静态图片/板级外设路径。
因此 FVP 与 FPGA 的 AXF 不可互换，二者必须使用独立构建目录。选择 FPGA 时，
模型也必须按 U85 Z1024 和相同 memory mode 导出；选择 FVP 时可以像本文的
YOLOv8 示例一样显式覆盖成 Z1024，但这只是让虚拟 NPU 配置匹配模型，不会把
FVP 固件变成可直接烧录到 MPS4 的固件。

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

可选 `-InputPath` 指定样本文件/目录，`-LabelsPath` 指定标签 YAML 或文本，
并用 `-ImageSize`、`-NumClasses`、`-ActivationBufferSize` 覆盖 YOLOv8 的输入尺寸、类别数和 Tensor arena 大小。
这些参数映射到 usecase 的 FILE_PATH / LABELS_YAML_FILE；其他 usecase
必须定义对应选项。高级用户可从 PowerShell 通过数组 `-CMakeArgs` 传递其他
CMake 选项，例如 `@('-Dimg_class_IMAGE_SIZE=224')`。
不要用高级参数覆盖脚本的编译器、框架、平台、Python 或构建目录配置。

GNU、ATFE、FVP、FPGA 和不同 usecase 分别使用独立构建目录。
脚本接受任意可写的本地构建目录，并拒绝已检测到的编译器/框架/目标缓存不匹配。
相对 `-BuildDirectory` 始终按仓库根目录解析。CMakeCache 会记录编译器等绝对路径，
所以仓库或工具链迁移到另一台机器后应使用新的构建目录，不要复制旧 `build/` 缓存。

### 6.1 YOLOv8 手势检测模型

当前仓库内的 ExecuTorch YOLOv8 模型为：

```text
resources_downloaded/windows-executorch/gesture_detection/best_dedicated_ethos-u85-1024.pte
```

它对应 `resources/gesture_detection/labels.txt` 中的 10 个类别，输入尺寸按当前模型配置使用
`320x320`，不是 `640x640`。`-ImageSize 320` 也会让样本生成器把输入图片转换为模型需要的尺寸。
文件名中的 `ethos-u85-1024` 表示模型按 U85 Z1024 导出；而 FVP 默认使用 Z256，
因此下面的 FVP 命令显式覆盖为 Z1024：

模型旁边的 `best_dedicated_ethos-u85-1024.json` 记录了该模型契约：输入为
`float32 RGB NCHW /255`，输出为 `[1, 4+C, N]` 的归一化 `XYWH + probabilities`，不需要
额外 sigmoid 或 score scaling。构建时应使用与此模型配套的手势图片和标签文件。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts/windows/build-executorch.ps1" -Action Build -Toolchain ATFE -Target FVP -UseCase yolov8_detection -ModelPath "resources_downloaded/windows-executorch/gesture_detection/best_dedicated_ethos-u85-1024.pte" -InputPath "resources/gesture_detection/samples" -LabelsPath "resources/gesture_detection/labels.txt" -ImageSize 320 -NumClasses 10 -ActivationBufferSize 0x00300000 -MemoryMode Dedicated_Sram -CMakeArgs "-DETHOS_U_NPU_CONFIG_ID=Z1024" -BuildDirectory "build/mlek-et-gesture_detection-z1024-atfe-fvp" -Jobs 8
```

首次使用时先执行一次 ExecuTorch Setup：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "scripts/windows/build-executorch.ps1" -Action Setup
```

上述命令使用相对于仓库根目录的路径，因此移动整个 checkout 后无需修改参数。
它显式指定了仓库内的默认图片目录 `resources/gesture_detection/samples`。
如果图片在其他目录，替换
`-InputPath "D:/path/to/gesture-images"`；该目录中的图片会在生成阶段统一转换为 320×320。
显式传入 `-InputPath` 也会覆盖旧 BuildDirectory 中缓存的错误路径。如果使用的是其他 NPU 配置导出的 PTE，必须同时更换 `-ModelPath`、`-DETHOS_U_NPU_CONFIG_ID`
和独立的 `-BuildDirectory`，不能只修改文件名。

## 7. 在 Arm DS 图形界面点击构建

打开 `Run → External Tools → External Tools Configurations...`，
新建 Program，填写：

- Name：`MLEK ExecuTorch img_class FVP`
- Location：`C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe`
- Working Directory：仓库根目录；Arm DS 工程名为 `mlek-client` 时可使用
  `${workspace_loc:/mlek-client}`
- Arguments（单行）：

```text
-NoProfile -ExecutionPolicy Bypass -File "scripts/windows/build-executorch.ps1" -Action Build -Toolchain ATFE -Target FVP -UseCase img_class -ModelPath "D:/models/mv2-u85-z256-dedicated.pte" -BuildDirectory "build/mlek-et-atfe-fvp" -Jobs 8
```

点击 Apply、Run，在 Console 查看日志。以后从 External Tools 下拉菜单选择该配置。
第一次先单独运行 Setup，也可以创建一个相同入口、参数仅含 `-Action Setup` 的配置。
这绑定的是 External Tools，不是工程的默认锤子按钮。
切换到 GNU 时替换 Toolchain，补充 CompilerBin，并换独立 BuildDirectory。

### 7.1 用相同模板创建其他 use case

Arm DS 中不需要为每个 use case 创建不同类型的工程。保留相同的 Program 类型、
PowerShell Location 和 Working Directory，只复制一份 External Tools 配置并替换
Arguments 中的 use case、模型和构建目录即可。

ExecuTorch/PTE 通用模板：

```text
-NoProfile -ExecutionPolicy Bypass -File "scripts/windows/build-executorch.ps1" -Action Build -Toolchain ATFE -Target <FVP或FPGA> -UseCase <usecase目录名> -ModelPath "<模型.pte>" -MemoryMode <Dedicated_Sram或Shared_Sram> -CMakeArgs "-DETHOS_U_NPU_CONFIG_ID=<Z256或Z1024等>" -BuildDirectory "build/mlek-et-<usecase>-<目标>-<模型配置>" -Jobs 8
```

TFLM/TFLite 通用模板：

```text
-NoProfile -ExecutionPolicy Bypass -File "scripts/windows/build.ps1" -Action Build -Toolchain ATFE -Target <FVP或FPGA> -UseCase <usecase目录名> -ModelPath "<模型.tflite>" -BuildDirectory "build/mlek-tflm-<usecase>-<目标>-<模型配置>" -Jobs 8
```

尖括号内容是占位符，不能原样保留。例如为 ExecuTorch `img_class` 创建 FVP
Z256 Dedicated SRAM 配置时，可以填写：

```text
-NoProfile -ExecutionPolicy Bypass -File "scripts/windows/build-executorch.ps1" -Action Build -Toolchain ATFE -Target FVP -UseCase img_class -ModelPath "D:/models/mobilenet-u85-z256-dedicated.pte" -MemoryMode Dedicated_Sram -CMakeArgs "-DETHOS_U_NPU_CONFIG_ID=Z256" -BuildDirectory "build/mlek-et-img_class-fvp-z256-dedicated" -Jobs 8
```

创建其他 use case 时遵守以下规则：

- `-UseCase` 使用 `source/app/use_case/<名称>` 中的目录名，并确认该 use case 声明支持所选框架。
- ExecuTorch 使用 `.pte`，TFLM 使用 `.tflite`；脚本不会在构建时把一种格式转换成另一种格式。
- 模型的 Ethos-U 配置和内存模式必须与 `ETHOS_U_NPU_CONFIG_ID`、`-MemoryMode` 一致。
- 每个框架、use case、FVP/FPGA 和模型配置使用不同的 `-BuildDirectory`，不要复制旧 CMakeCache。
- 只有 use case 定义了对应 CMake 选项时才添加 `-InputPath`、`-LabelsPath`、
  `-ImageSize`、`-NumClasses` 或 `-ActivationBufferSize`。
- 输出通常位于 `<BuildDirectory>/bin/mlek_<usecase>.axf`；在对应的 Debug Connection
  中加载 AXF，不要加载分区 `.bin`。

Setup External Tool 可以长期复用。ExecuTorch 配置使用：

```text
-NoProfile -ExecutionPolicy Bypass -File "scripts/windows/build-executorch.ps1" -Action Setup
```

TFLM 配置使用：

```text
-NoProfile -ExecutionPolicy Bypass -File "scripts/windows/build.ps1" -Action Setup -UseCase <usecase目录名>
```

### 7.2 Arm DS 中构建 YOLOv8

建议在 Arm DS 中创建两个 External Tool 配置：一个只执行 Setup，另一个执行下面的
YOLOv8 Build。Arguments 使用单行文本：

- Name：`MLEK ExecuTorch YOLOv8 gesture_detection FVP Z1024`
- Location：`C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe`
- Working Directory：仓库根目录；Arm DS 工程名为 `mlek-client` 时可使用
  `${workspace_loc:/mlek-client}`
- Arguments：

```text
-NoProfile -ExecutionPolicy Bypass -File "scripts/windows/build-executorch.ps1" -Action Build -Toolchain ATFE -Target FVP -UseCase yolov8_detection -ModelPath "resources_downloaded/windows-executorch/gesture_detection/best_dedicated_ethos-u85-1024.pte" -InputPath "resources/gesture_detection/samples" -LabelsPath "resources/gesture_detection/labels.txt" -ImageSize 320 -NumClasses 10 -ActivationBufferSize 0x00300000 -MemoryMode Dedicated_Sram -CMakeArgs "-DETHOS_U_NPU_CONFIG_ID=Z1024" -BuildDirectory "build/mlek-et-gesture_detection-z1024-atfe-fvp" -Jobs 8
```

Arm DS 的 Arguments 中必须写普通下划线，例如 `yolov8_detection`；不要把 Markdown
显示时可能出现的转义形式 `yolov8\_detection` 原样复制进去。

Setup 配置的 Arguments 为：

```text
-NoProfile -ExecutionPolicy Bypass -File "scripts/windows/build-executorch.ps1" -Action Setup
```

点击 Build 配置的 Run 后，AXF 默认位于：

```text
<仓库根目录>/build/mlek-et-gesture_detection-z1024-atfe-fvp/bin/mlek_yolov8_detection.axf
```

然后在 Arm DS 的 Debug Connection 中选择 SSE-320 FVP、Cortex-M85，并加载该 AXF。
External Tool 的 Build 和 Debug Connection 是两个独立配置；修改模型或源码后，必须先重新
运行 Build，再启动 Debug。若实际使用 MPS4 FPGA，把 `-Target FVP` 改为 `-Target FPGA`，
并使用另一个 FPGA 专用构建目录。

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
- `sample_files.h` 找不到：应用目标需要包含生成的 `generated/<usecase>/samples` 目录；
  当前 CMake 已自动加入该目录。
- `llvm-objcopy` 找不到：LLVM 工具链 CMake 会在配置时解析工具链目录中的绝对路径；
  确认 ATFE 的 `bin` 目录中存在 `llvm-objcopy.exe` 后重新 Configure/Build。
- YOLOv8：单输出以及三/四输出 DFL 输出均已接入；若模型输出布局不同，需要按实际输出张量调整后处理。
- Setup 成功但没有 AXF：Setup 本身不会编译，需再运行 Build。

## 11. 本机实测结果（2026-09-22）

已通过 Windows 原生 ATFE Pro 22.1.0 / ExecuTorch 1.3.1 / SSE-320 FVP
的 img_class 编译、链接和一次 NPU 推理。使用 U85 Z256 / Shared_Sram 模型；
这项结果不代表 Dedicated_Sram、FPGA 实板或 YOLOv8 的具体模型已经完成端到端验收。

- AXF：`D:/build/mlek-et-atfe-fvp/bin/mlek_img_class.axf`
- 环境日志：`D:/work/mlek-windows/setup-et.log`
- 构建日志：`D:/work/mlek-windows/build-et-atfe.log`
- 最终链接日志：`D:/work/mlek-windows/link-et-atfe.log`
- FVP 日志：`D:/work/mlek-windows/run-et-atfe.log`
- 推理次数：1；Top-1：281，tabby, tabby cat；NPU ACTIVE：3062718 cycles。
- 应用日志：`Main loop terminated successfully.`
- 重复运行 Build 成功，已应用的子模块补丁通过反向检查，不会重复修改。

Windows 原生 GNU 参数入口和路径处理已提供，但本次没有完成 Windows GNU 编译验收。
Windows 官方便携工具链下载未完成，已停止下载，部分 ZIP 保留在
`D:/tools/mlek-downloads/arm-gnu-14.3.zip`，不能直接解压使用。
Windows 当前先采用已验证的 ATFE 命令；本文提供的 Windows GNU 命令属于待验证配置。

同日还在 Ubuntu 24.04 / WSL 中直接使用仓库原有 CMake 入口、CMake 4.4.3、
Unix Makefiles 和 GNU Arm 13.3.1 完成了 TFLM YOLOv8 FVP Z1024 Dedicated SRAM
的配置、代码生成、编译和链接，输出为
`/tmp/mlek-current-linux-yolov8-fvp-20260922/bin/mlek_yolov8_detection.axf`。
验证过程中修复了 GNU 对 `std::min` 两侧整数类型严格检查所暴露的问题。
Windows 的共享 CMake/Python 环境逻辑只存在于 `scripts/windows/*.ps1` 和
`scripts/cmake/windows-presets.json`，Linux 原有 CMake 调用方式未被替换。

SSE-320 LLVM 链接脚本额外修复了 libc++ override 段边界，以及 ATFE 22
picolibc 的 combined initializer 边界；否则可能分别出现链接失败和后端未注册。
FVP 通过 UART EOT 停止时进程退出码不一定为零，本次验收依据应用成功日志、
分类输出和 NPU 计数，不仅依赖进程退出码。
