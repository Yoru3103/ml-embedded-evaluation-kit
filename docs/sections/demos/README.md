# Corstone-320 FVP 与 FPGA 四模型演示总报告

更新时间：2026 年 11 月 2 日。

本演示覆盖中文关键词唤醒、英文语音识别、手部关键点和二维码／条码检测。
PC 完成采集、格式转换和数据传输，Corstone-320 FVP 或 MPS4 FPGA 执行模型推理、
后处理及 LCD 可视化，终端同步输出结果。

FVP 支持实时采集和文件输入：音频使用麦克风或 WAV，图像使用摄像头按键拍摄或图片。
FPGA 使用文件输入：加载常驻接收固件后，通过 UART 上传 WAV 或图片；
更换输入文件不需要重新构建或下载固件。

| 模型 | FVP 输入 | FPGA 输入 | LCD 展示 |
| --- | --- | --- | --- |
| 中文 KWS | 麦克风 / WAV | UART 上传 WAV | 波形、唤醒词、触发秒数、问候 |
| 英文 ASR | 麦克风 / WAV | UART 上传 WAV | 收音／处理状态、英文文本 |
| 手部关键点 | 摄像头空格拍摄 / 图片 | UART 上传图片 | RGB 图像、手部框、21 点及 20 条连线 |
| 二维码／条码 | 摄像头空格拍摄 / 图片 | UART 上传图片 | 灰度图像、OBB 旋转框、类别、置信度及角度 |

截至 2026 年 10 月 29 日，用户已完成四种模型的演示实测，并确认新增 FPGA 文件调用链路可用。
本文沿用此前报告的四部分结构，统一说明当前版本的执行方法、展示效果、实现技术和部署产物。

以下命令均在 Windows PowerShell 中，从仓库根目录执行：

```powershell
Set-Location C:\Users\MC\Documents\mlek-client\ml-embedded-evaluation-kit
```

## 1. 四种模型的演示流程与说明

### 1.1 中文关键词唤醒 KWS

**展示目标：** 输入“你好问问”时，在波形上标记触发位置，显示关键词、源音频秒数和
`NI HAO! ^_^`。现场演示统一使用 `Nihao_Wenwen`，LCD 提示为 `Say Nihao Wenwen`。
模型内部仍保留两个输出类别；本次不调整模型权重和类别定义。

首次构建，或修改应用代码后重新构建：

```powershell
.\scripts\windows\build-fvp-audio-demo.ps1 -UseCase kws_zh
```

先用固定 WAV 复现：

```powershell
.\scripts\windows\run-fvp-audio-demo.ps1 -UseCase kws_zh `
    -Wav resources/kws_zh/samples/02_nihao_wenwen.wav
```

切换到麦克风：

```powershell
.\scripts\windows\run-fvp-audio-demo.ps1 -UseCase kws_zh -Microphone
```

看到 `KWS_VSI_READY` 后，对着麦克风说“你好问问”；等待检测结果，再进行下一次输入。
现场提示只显示 `Say Nihao Wenwen`。
终端的 `WAKE` 行包含关键词、分数、触发时间和队列时长。时间按源音频采样位置换算，
表示检测器触发时刻。LCD 的问候与最近结果保持到下一次检测成功。

WAV 要求单声道、16 kHz、PCM16。麦克风由主机直接按该格式采集。
KWS 使用约 1.995 秒上下文，每前进 1 秒做一次推理，主机队列默认 5 秒。

**FPGA：上传 WAV。** 首次构建文件接收固件：

```powershell
.\scripts\windows\build-fpga-file-demo.ps1 -UseCase kws_zh
```

按 1.5 节将 KWS 固件加载到 FPGA，再发送文件：

```powershell
.\scripts\windows\send-fpga-demo-file.ps1 -UseCase kws_zh `
    -InputPath resources/kws_zh/samples/02_nihao_wenwen.wav -Port COM5
```

板端显示整份文件的波形，触发时显示关键词、秒数和问候；终端打印 `WAKE` 明细，
以 `FILE_RESULT <id> <count>` 标记本次文件处理完成。
再次上传可换一份音频，结果保持到下一份有效文件开始处理。

### 1.2 英文语音识别 ASR

**展示目标：** 说出一句英文，静音结束后运行推理，在终端和 LCD 显示识别文本。
LCD 保持上一句结果，新的状态提示显示在上方。

构建：

```powershell
.\scripts\windows\build-fvp-audio-demo.ps1 -UseCase asr
```

固定 WAV：

```powershell
.\scripts\windows\run-fvp-audio-demo.ps1 -UseCase asr `
    -Wav resources/asr/samples/another_door.wav
```

麦克风：

```powershell
.\scripts\windows\run-fvp-audio-demo.ps1 -UseCase asr -Microphone
```

状态依次为 `Listening`、`Capturing speech`、`Processing sentence`，随后显示文本。
终端使用 `ASR_VSI_UTTERANCE` 标记句段，使用 `ASR_VSI_RESULT` 输出结果。
演示时可以说 “Open the door”，或读一段短英文，再停顿约半秒。

**收音门限与断句。** 默认固件使用 RMS 800，连续约 120 ms 的语音启动收音，
保留约 600 ms 前导音频；约 1000 ms 静音触发断句，至少 250 ms 有效语音才提交识别。

麦克风模式支持启动时调整门限，无需重编固件：

```powershell
.\scripts\windows\run-fvp-audio-demo.ps1 -UseCase asr -Microphone `
    -LevelMeter
```

默认直接传输原始 PCM，不启用主机音频门控。`-LevelMeter` 仅显示电平，不修改采样。
显式指定 `-VadDbfs` 才会开启将低电平块置零的可选门控；该选项可能截掉轻声和辅音，
当前优先使用不带该参数的原音频路线。数值越接近 0，门控越严格。
例如环境较吵时可改为 -22；轻声说话被漏掉时可改为 -30。
`-LevelMeter` 每秒打印一次麦克风 RMS 数字电平，可先看安静时和讲话时的读数，
将门限放在两者之间。dBFS 相对 PCM 满量程计算，表示数字输入电平。

可选的主机低电平门控保留原始采样数量和时间位置，低于门限的块替换为零；
持续 120 ms 达到门限后，句内延续门限降低 6 dB，连续安静 500 ms 后恢复。
固件继续使用已有的起句确认、最短语音和句末静音判断。WAV 文件模式不应用主机门控。
默认固件 RMS 门限为 800，约 -32.2 dBFS；若需接收更低电平的语音，
构建时也需通过 `-VadRms` 降低该值。

调整固件断句参数时可重新构建，例如：

```powershell
.\scripts\windows\build-fvp-audio-demo.ps1 -UseCase asr -VadRms 1200 -PreRollMs 600 -SilenceMs 1000
```

ASR 主机队列默认 30 秒，用于保存上一句推理期间采集的音频；30 秒是容量上限，
不是固定等待时间。原有命令若写了 `-QueueMs 1000`，可删除该参数，或改为 `30000`。

**FPGA：上传 WAV。** 构建和调用命令：

```powershell
.\scripts\windows\build-fpga-file-demo.ps1 -UseCase asr
.\scripts\windows\send-fpga-demo-file.ps1 -UseCase asr `
    -InputPath resources/asr/samples/another_door.wav -Port COM5
```

先加载 ASR 固件，再执行上传。文件模式直接处理整份 WAV，不使用麦克风起句门限；
超过单次模型窗口的音频按现有滑窗逻辑处理。终端显示识别文本以及处理计时，
LCD 显示英文结果并保持。发送下一份 WAV 即可进行下一次识别。
ASR 传输沿用 `ASR1` 协议，板端用 `ASR_RESULT` 返回结果，统一上传脚本负责解析。

### 1.3 手部关键点

**展示目标：** 在摄像头预览中摆出手势，按空格拍摄；LCD 显示该帧 RGB 图像、
手部框、21 个关键点和 20 条骨架连线。终端打印各点坐标及置信度。

构建 FVP 固件：

```powershell
.\scripts\windows\build-hand-pose-fvp-demo.ps1
```

启动摄像头：

```powershell
.\scripts\windows\run-hand-pose-fvp-demo.ps1
```

点击摄像头窗口使其获得焦点，把完整手部放在方形取景范围内，按 **空格**。
推理期间预览持续刷新，新的空格输入暂不提交。提示可再次拍摄后，改变手势再按空格。
LCD 保持最近拍摄帧的识别结果。

默认取景为“中心正方形裁剪 → 224×224 RGB”，预览窗口设为模型输入尺寸的 2 倍，即 448×448。预览提示位于图像外的黑条，模型收到的是干净图像。
需要完整相机视野时使用等比例补边：

```powershell
.\scripts\windows\run-hand-pose-fvp-demo.ps1 -Framing letterbox
```

固定图片输入：

```powershell
.\scripts\windows\run-hand-pose-fvp-demo.ps1 `
    -ImagePath resources/yolov8_pose/samples/test03.jpg
```

运行中，在第二个终端发送另一张图：

```powershell
.\scripts\windows\send-hand-pose-image.ps1 -ImagePath "D:\demo\hand.jpg"
```

摄像头索引或后端可调整为 `-CameraIndex 1 -CameraBackend msmf`。
关键点置信度低于 0.25 时不绘制；一条骨架的两个端点都达到阈值才绘制连线。
当前模型输出关键点位置，演示不额外给手势命名。

**FPGA：上传图片。** 构建和调用命令：

```powershell
.\scripts\windows\build-fpga-file-demo.ps1 -UseCase hand-pose
.\scripts\windows\send-fpga-demo-file.ps1 -UseCase hand-pose `
    -InputPath resources/yolov8_pose/samples/test03.jpg -Port COM5
```

PC 将图片整理为 224×224 RGB 再上传，FPGA 使用相同 PTE 和关键点后处理，
在 LCD 上叠加手部框、21 个点和骨架连线。结果保持到下一份有效图片。
需要保留图片完整视野时，上传命令可加 `-Framing letterbox`。
终端打印关键点明细，并以 `FILE_RESULT <id> <count>` 标记完成。
当前默认模型采用重新校准的 INT8 主体与 A16W8 检测头，资源配置见 4.3 节。

### 1.4 二维码与条码检测

**展示目标：** 输入一张包含码的图像，在 LCD 上显示灰度输入、旋转检测框、
类别和置信度。模型做定位与分类，码内文本解码可作为独立后续功能。

构建：

```powershell
.\scripts\windows\build-qrcode-fvp-demo.ps1
```

固定图片：

```powershell
.\scripts\windows\run-qrcode-fvp-demo.ps1 `
    -ImagePath resources/qrcode/smaples/01.bmp
```

运行中换图：

```powershell
.\scripts\windows\send-qrcode-image.ps1 -ImagePath "D:\demo\qr.jpg"
```

按键拍摄模式：

```powershell
.\scripts\windows\run-qrcode-fvp-demo.ps1 -Camera
```

点击预览窗口，按 **空格** 拍摄；按 **Q / Esc** 或关闭预览窗口退出。
图像经 EXIF 方向校正、灰度转换和等比例补边，形成 256×256 输入。
终端的 `QRCODE_RESULT id=... count=...` 表示一帧处理结束，检测明细包含类别、
分数、中心位置、宽高和旋转角。类别包括 QR、BARCODE、DM 等。

**FPGA：上传图片。** 构建和调用命令：

```powershell
.\scripts\windows\build-fpga-file-demo.ps1 -UseCase qrcode
.\scripts\windows\send-fpga-demo-file.ps1 -UseCase qrcode `
    -InputPath resources/qrcode/smaples/01.bmp -Port COM5
```

PC 生成 256×256 灰度图后上传。FPGA 完成 DFL／角度解码和旋转 NMS，
LCD 绘制四条旋转边，并显示类别、置信度及 `angle: ... deg`。
终端打印检测明细，以 `FILE_RESULT <id> <count>` 标记完成。
旋转图像样例已在 FVP 验证；当前版本的框几何和绘图约定见 3.7 节。

### 1.5 FPGA 通用部署与文件调用

每种模型使用各自的常驻文件接收固件。切换模型时更换固件，模型内切换 WAV／图片时
只需再次执行上传命令。

**步骤一：准备三份 BIN 和加载配置。** 当前文件接收固件的产物目录如下：

| 模型 | AXF | BIN 子目录 |
| --- | --- | --- |
| KWS | `build/fpga-file/kws_zh/bin/mlek_kws_zh.axf` | `build/fpga-file/kws_zh/bin/sectors/kws_zh/` |
| ASR | `build/fpga-file/asr/bin/mlek_asr.axf` | `build/fpga-file/asr/bin/sectors/asr/` |
| 手部 | `build/fpga-file/hand-pose/bin/mlek_yolov8_pose.axf` | `build/fpga-file/hand-pose/bin/sectors/yolov8_pose/` |
| 二维码 | `build/fpga-file/qrcode/bin/mlek_qrcode.axf` | `build/fpga-file/qrcode/bin/sectors/qrcode/` |

每个 BIN 子目录均包含 `boot.bin`、`bram.bin`、`ddr.bin`；
对应的 `images.txt` 位于各自 `bin/sectors/` 下。使用同一次构建产生的整套文件。
ASR 另有完整包 `build/fpga-file/asr/package.zip`，包含 `SOFTWARE/`、
`images.txt`、AXF、配置记录和校验文件。

**步骤二：部署到板卡。**

1. 将目标模型的三份 BIN 复制到板卡盘的 `SOFTWARE/` 目录。
2. 将该模型的 `images.txt` 复制到当前使用的 FPGA image 目录。
   Revision B 的目录示例为 `MB/HBI0376B/FI101`，按板上实际 image 配置选择。
3. 安全弹出板卡盘，启动或复位板卡，等待模型初始化完成。
4. 使用 Arm DS 调试时，也可直接加载对应 AXF。

`images.txt` 配置的加载地址：

| 文件 | 加载地址 |
| --- | --- |
| `boot.bin` | `0x11000000` |
| `bram.bin` | `0x12000000` |
| `ddr.bin` | `0x70000000` |

**步骤三：上传输入文件。** 使用 1.1～1.4 节对应的 `send-fpga-demo-file.ps1` 命令。
示例中的 `COM5` 替换为实际 UART0 端口；上传前关闭占用该端口的串口终端。
默认波特率 115200，上传脚本会显示推理结果，不需要同时打开另一串口终端。

| 参数 | 用途 |
| --- | --- |
| `-UseCase kws_zh / asr / hand-pose / qrcode` | 选择与已加载固件对应的模型 |
| `-InputPath <路径>` | 指定 WAV 或图片，可使用带空格的路径 |
| `-Port COM5` | FPGA 的串口端口 |
| `-TimeoutSeconds 180` | 等待上传响应和推理结果的超时，默认 180 秒 |
| `-Framing letterbox` | 手部图片保留完整视野，默认方形裁剪 |
| `-ValidateOnly` | 只做文件检查与预处理，不连接板卡 |

WAV 要求 16 kHz、单声道、PCM16，最长 30 秒。手部图片整理为 224×224 RGB，
二维码图片整理为 256×256 灰度并等比例补边。图片预处理会打印 `input-preview.png`
的位置，便于查看实际传入模型的像素。

输入文件转换为采样或原始像素后传输，板端无需解析 WAV／JPG 容器。
每 512 字节由板端确认一次，整份数据通过 CRC32 校验后才进入推理。
若在二进制文件发送中途断开连接，复位板卡后再发送。

新构建默认使用 `build/fpga-file/<UseCase>/`。
本次 ASR 复用了 `build/asr-uart/dedicated-256k/` 构建目录，
已将 AXF、BIN 和 `images.txt` 整理到表中的统一目录。

## 2. 演示效果与截图位置

本次四种模型的演示均已由用户完成实测，新增 FPGA 文件调用可正常执行。
下表记录本次功能验收范围：

| 模型 | FVP 演示 | FPGA 文件演示 | 可观察结果 |
| --- | --- | --- | --- |
| KWS | 麦克风／WAV 输入 | 上传关键词 WAV | 关键词响应、波形标记、秒数及问候 |
| ASR | 麦克风断句／WAV 输入 | 上传英文 WAV | 英文识别文本及结果保持 |
| 手部 | 摄像头按键拍摄／图片 | 上传手部图片 | 手部框、21 点及骨架连线 |
| 二维码 | 摄像头按键拍摄／图片 | 上传码图 | 旋转框、类别、置信度及角度 |

FVP 截图建议同时包含 LCD 和终端，摄像头模型再保留预览窗口；
FPGA 截图可使用板卡 LCD 照片配合上传终端。以下位置留待插入实际截图。

### 2.1 中文 KWS：波形、触发时间和问候

操作：播放固定关键词 WAV，或说出唤醒词，等待 LCD 显示关键词和 `NI HAO! ^_^`。

> **截图占位 A：在此插入 KWS 的 FVP 展示截图。**
>
> 建议保留：波形、触发标记、关键词、秒数、问候和终端 WAKE 行。

图注：关键词出现后，检测器在源音频时间轴上给出触发时刻，并刷新 LCD 结果。

> **截图占位 A-FPGA：在此插入 FPGA KWS 的 LCD 与上传结果。**
>
> 建议保留：关键词、时间、波形标记与 FILE_RESULT。

### 2.2 英文 ASR：句段识别与文本保持

操作：读出一段短英文，停顿断句，等待文字显示；再说第二句，展示结果更新。

> **截图占位 B：在此插入 ASR 识别结果截图。**
>
> 建议保留：LCD 英文文本、Listening 状态、终端 ASR_VSI_RESULT。

图注：收音和断句后进行推理，LCD 保持最近一次有效识别文本。

> **截图占位 B-FPGA：在此插入 FPGA ASR 的 LCD 文本与上传终端。**
>
> 建议保留：识别文本、输入文件名和处理计时。

### 2.3 手部关键点：张手与弯指

操作：先拍摄张开的手掌，再拍摄弯指或其他手势，展示关键点和连接关系。

> **截图占位 C1：在此插入张手场景截图。**
>
> 建议保留：方形预览、LCD 图像与骨架、终端关键点输出。

> **截图占位 C2：在此插入弯指或第二种手势截图。**
>
> 建议使用同一光照和相机位置，便于观察不同手势的定位效果。

图注：空格捕获单帧，在模型输入图像上绘制关键点；LCD 保持拍摄结果，预览继续刷新。

> **截图占位 C-FPGA：在此插入 FPGA 手部关键点的 LCD 照片。**
>
> 建议保留：输入图像、关键点和骨架连线。

### 2.4 二维码/条码：旋转框与类别

操作：加载样例图片或拍摄码图，等待旋转框、类别及置信度显示。

> **截图占位 D：在此插入二维码/条码检测截图。**
>
> 建议保留：灰度输入、旋转框、类别、置信度、角度读数和终端检测明细。

图注：模型输出候选区域，经解码与旋转 NMS 后，在 LCD 上绘制最终结果。

> **截图占位 D-FPGA：在此插入 FPGA 二维码的 LCD 照片与检测明细。**
>
> 建议选择倾斜码图，保留旋转框和角度读数。

## 3. 脚本构建思路与涉及技术

### 3.1 应用层适配与脚本分工

实现分成“构建入口、主机运行与输入准备、传输适配、端侧应用处理”四层。
模型推理、后处理和 LCD 绘制在 FVP／FPGA 执行；采集、格式整理和上传由 PC 完成。

| 层次 | 主要文件 | 职责 |
| --- | --- | --- |
| 音频构建与启动 | `scripts/windows/build-fvp-audio-demo.ps1`、`run-fvp-audio-demo.ps1` | 选择 KWS/ASR，配置模型、内存、VSI 和 LCD，管理主机与 FVP 进程 |
| 音频主机 | `scripts/py/fvp_audio_demo.py`、`fvp_audio_demo_sources.py` | 麦克风/WAV 采集、定长分块、有限队列和本机传输 |
| 音频 VSI | `scripts/py/fvp_audio_demo_vsi/`、`source/app/audio_demo/FvpAudioInput.cc` | 主机请求、VSI 寄存器元数据、DMA 搬运和采样时序 |
| 音频业务 | `KwsVsiHandler.cc`、`ConformerVsiHandler.cc` | 连续窗口、特征、VAD、推理调度、结果与状态显示 |
| 图像构建 | `build-hand-pose-fvp-demo.ps1`、`build-qrcode-fvp-demo.ps1` | 固定模型接口、输入尺寸、显示布局及内存配置 |
| 图像运行监督 | `run-hand-pose-fvp-demo.ps1`、`run-qrcode-fvp-demo.ps1` | 启动 FVP、摄像头进程、日志监控及退出清理 |
| 图像准备 | `prepare_hand_pose_demo.py`、`prepare_qrcode_demo.py` | 方向校正、裁剪/缩放、颜色格式和请求封包 |
| 按键拍摄 | `scripts/py/qrcode_camera_demo.py` | 共用摄像头预览、空格拍摄、单请求在途和结果确认 |
| 图像接收 | `source/app/image_demo/include/ImageRequest.hpp` | 请求 ID、尺寸、长度和 CRC32 检查 |
| ASR 电平门控 | `scripts/py/audio_level_gate.py` | PCM16 RMS、dBFS、语音延续门限和静音释放 |
| FPGA 构建与上传 | `build-fpga-file-demo.ps1`、`send-fpga-demo-file.ps1` | 四模型统一构建入口、文件预处理、串口握手与分块传输 |
| UART 通用文件接收 | `source/app/file_demo/include/UartFileInput.hpp` | KWS／手部／二维码二进制接收、长度和 CRC32 校验 |
| KWS 文件业务 | `KwsFileHandler.cc` | 文件窗口推理、波形绘制、关键词与时间输出 |
| ASR 文件业务 | `ConformerUartHandler.cc`、`send-asr-audio.ps1` | 复用 ASR1、完整 WAV 处理和 LCD 文本显示 |
| FPGA 打包 | `scripts/windows/package-fpga.ps1` | 三份 BIN 与 AXF 交叉校验、加载配置、模型记录与 ZIP |

构建脚本最终调用统一的 Windows CMake/ATFE 构建入口，将 PTE 作为模型资源链接进固件。
VSI 音频、semihosting 图像及 UART 文件输入都在应用层完成适配。
HAL 继续提供原有显示、缓存和外设接口，`source/lib/` 保持平台无关。

### 3.2 音频链路：VSI 拉取与连续上下文

```text
麦克风 / WAV
  → 主机 Python：16 kHz、单声道、PCM16
  → 每块 320 个采样（20 ms）→ 有限队列
  → localhost 连接 → FVP Python VSI0 回调
  → VSI 寄存器发布位置/长度/标志 → DMA 复制 PCM
  → 应用特征与模型推理 → LCD + UART 结果
```

音频数据通过 VSI 输入，UART 用于结果输出。主机连接采用 Python
`multiprocessing.connection`，默认端口 6010。
FVP 一次拉取一个块；采样位置使用绝对计数，时间戳不会随队列积压重新从零开始。

WAV 可采用反压，保留完整样本。麦克风持续采集，队列到上限时淘汰旧块，并标记不连续。
应用遇到不连续音频会重置相应上下文：KWS 重新积累窗口，ASR 丢弃被截断的句段。
恢复路径合并连续丢块事件，避免逐块 UART 日志和重复 LCD 刷新拖住读取。
`queue_ms` 表示待处理音频时长，便于观察积压。

KWS 把滚动窗口转成 Fbank 特征，再做唤醒检测和冷却控制；波形先在应用 RGB 缓冲区
中一次性绘制，再提交 LCD。ASR 使用能量 VAD 收集完整句段，完成特征、Conformer
推理和文本后处理；较长句段采用滑窗拼接。

### 3.3 图像链路：semihosting 与原子文件请求

```text
图片 / 摄像头空格拍摄
  → PC：方向校正 → 裁剪或等比例补边 → RGB / 灰度
  → 请求头 + 原始像素 → 原子替换输入文件
  → FVP semihosting fopen/fread
  → 尺寸、长度、CRC32 校验 → 输入量化与布局整理
  → NPU 推理 → 后处理 → LCD 图像与结果
```

使用临时文件写完后原子替换，FVP 读到的是一份完整请求。头部采用小端编码，
包含请求标识、ID、宽、高、像素字节数与 CRC32。二维码使用 `QRD1` 和灰度像素，
手部使用 `HPD1` 和 RGB 像素。

模型保持常驻；只有新 ID 才触发推理和清屏。按键拍摄一次只允许一个请求在途，
收到匹配的 `HAND_POSE_RESULT` 或 `QRCODE_RESULT` 后才允许再次提交。
等待期间 LCD 不清空，所以检测结果能保留到下一次拍摄。

### 3.4 模型接口和显示映射

| 模型 | 输入与主要输出 | 后处理 |
| --- | --- | --- |
| KWS | 16 kHz 连续音频 → Fbank → 关键词帧分数 | 连续帧判断、时间戳、冷却 |
| ASR | 700×80 特征；文本输出头为 174×129 | 文本解码、长句滑窗拼接 |
| 手部 | 224×224 RGB；boxes、confidence、42 路 xy、21 路关键点分数 | 候选筛选、NMS、21 点与手部拓扑 |
| 二维码 | 256×256 灰度；三路 47 通道特征头和一路 angle | DFL 解码、旋转框、旋转 NMS |

手部 LCD 显示实际模型输入，默认 224×224 原尺寸，因此关键点直接使用模型像素坐标。
二维码显示缩小后的模型输入，绘图时使用同一缩放系数变换框的位置和尺寸。
两种图像演示的预览操作提示都不进入模型输入。

### 3.5 FPGA 链路：常驻模型与 UART 文件传输

```text
PC 的 WAV / JPG / BMP 等文件
  → 主机检查格式、裁剪／缩放、转换 PCM16 / RGB / 灰度
  → 查询目标类型与容量 → 请求头 → 每 512 字节等待 ACK
  → FPGA 应用接收 → 长度与 CRC32 校验
  → 特征／输入量化与布局整理 → NPU 推理 → 后处理
  → LCD 保持结果 + UART 返回结果
```

| 模型 | 协议标识 | 载荷 | 完成响应 |
| --- | --- | --- | --- |
| KWS | `KWD1` | 16 kHz PCM16 | `FILE_RESULT <id> <count>`，另有 WAKE 明细 |
| ASR | `ASR1` | 16 kHz PCM16 | `ASR_RESULT`，主机解析后显示文本和计时 |
| 手部 | `HPD1` | 224×224×3 RGB | `FILE_RESULT <id> <count>`，另有关键点明细 |
| 二维码 | `QRD1` | 256×256 灰度 | `FILE_RESULT <id> <count>`，另有 OBB 明细 |

KWS、手部和二维码共用 24 字节小端请求头，包含类型、请求编号、
尺寸／采样数、载荷字节数和 CRC32。ASR 保留原有 16 字节协议头。
接收端先确认头部，再逐块确认，避免主机连续写入超过 UART FIFO 容量。

UART 原始字节读取位于应用适配层，保留 CR、LF、NUL、0xff 等字节值，
避免文本串口函数的换行转换破坏采样或图像。
校验成功后才替换 LCD 画面并进入推理；CRC 错误不触发推理。
模型在启动时加载一次，循环接收不同文件，实现“换文件、不换固件”。

### 3.6 进程与资源管理

运行脚本统一管理自己启动的进程。终端 **Ctrl+C / Q** 触发退出，摄像头窗口的
**Q / Esc / 关闭窗口** 也会结束演示。退出时关闭摄像头、主机连接和 FVP。
同一图像构建目录使用独占锁，避免两个实例读取同一个请求文件。

### 3.7 二维码 OBB 解码与旋转绘图

当前模型输出三路 DFL／类别原始头以及一路角度 logits，角度按
`theta = (sigmoid(raw_angle) - 0.25) * pi` 转成弧度。
DFL 距离与锚点通过同一角度恢复中心坐标，输出 `cx, cy, width, height, theta`，
再旋转四个局部角点、缩放到 LCD 并连接四条边；NMS 使用旋转框 IoU。
LCD 同时显示 `angle: ... deg`，便于核对实际预测角度。

坐标定义参照 [Ultralytics OBB 文档](https://docs.ultralytics.com/tasks/obb/)和
[xywhr 四角转换](https://docs.ultralytics.com/reference/utils/ops/#ultralytics.utils.ops.xywhr2xyxyxyxy)。

## 4. 环境、产物与演示准备

### 4.1 本机环境

| 项目 | 当前配置 |
| --- | --- |
| 主机 | Windows + PowerShell；量化在 Ubuntu-24.04 WSL |
| FVP | Corstone SSE-320，Cortex-M85 + Ethos-U85 1024 MAC |
| FPGA | MPS4 SSE-320，使用对应的板端 UART、DDR 和 timing adapter 配置 |
| 编译器 | ATFE 22.1.0；CMake + Ninja |
| 主机 Python | `resources_downloaded/windows-executorch/env/Scripts/python.exe` |
| CMake/Ninja | `resources_downloaded/windows-tools/env/Scripts` |
| 音频依赖 | sounddevice；FVP 内使用 Python 3.9 VSI 回调与 socket 扩展 |
| 图像依赖 | Pillow、NumPy、带 GUI 的 OpenCV |
| 显示 | 320×240 LCD |

默认 FVP 路径：

```text
C:/Program Files/Arm/FVP_Corstone_SSE-320/models/Win64_VC2019/FVP_Corstone_SSE-320.exe
```

### 4.2 FVP 固件与 FPGA 产物

| 模型 | 默认构建目录 | 固件 |
| --- | --- | --- |
| KWS | `build/fvp-audio/kws_zh` | `bin/mlek_kws_zh.axf` |
| ASR | `build/fvp-audio/asr` | `bin/mlek_asr.axf` |
| 手部 | `build/fvp-hand-pose/dedicated-256k` | `bin/mlek_yolov8_pose.axf` |
| 二维码 | `build/fvp-qrcode/dedicated-384k` | `bin/mlek_qrcode.axf` |

FPGA 文件接收固件、三份 BIN 和加载配置统一见 1.5 节。
ASR 当前完整下载包为 `build/fpga-file/asr/package.zip`，三份 BIN 已与对应 AXF 校验一致。
AXF 用于调试器加载；`boot.bin`、`bram.bin`、`ddr.bin` 配合 `images.txt` 用于板卡启动。

前三种使用 Dedicated SRAM 256 KiB NPU cache，二维码使用 384 KiB。
图像演示打开 semihosting，音频演示通过 VSI 输入。FVP 演示采用 functional fast 模式，
展示采集、推理和显示链路；FPGA 固件保留板端时序及硬件配置。

### 4.3 演示前的操作清单

使用的模型资源：

| 模型 | PTE 资源 |
| --- | --- |
| KWS | `resources_downloaded/kws_zh/pte_Dedicated_Sram_256k/wekws_dstcn_u85.pte` |
| ASR | `resources_downloaded/asr/et151-u85-1024-dedicated-256k/conformer_fp32_cln_wer_6_47_arm_delegate_ethos-u85-1024_Dedicated_Sram.pte` |
| 手部 | `resources_downloaded/yolov8_pose/hand_pose_u85-1024_ded_256KiB_mixed.pte` |
| 二维码 | `resources_downloaded/yolo_QRcode/usr_quant_nhwc_angle3d/pte/best_bifpn_gray_nhwc_angle3d_u85_1024_384KiB.pte` |

手部模型由 WSL 原始 `best.pt` 和训练集的 500 张校准图重新量化，主体采用 INT8、检测头采用
A16W8，外部仍为 RGB 输入和四路像素坐标输出。重新量化入口为
`scripts/py/requantize_hand_pose.py`，使用现有 `pt2pte_toolchain/.venv` 运行。
在仓库根目录重新量化和导出 PTE：

```powershell
wsl -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Users/MC/Documents/mlek-client/ml-embedded-evaluation-kit && PYTHONDONTWRITEBYTECODE=1 /home/mokuroo/documents/python/pt2pte_toolchain/.venv/bin/python scripts/py/requantize_hand_pose.py --output build/hand-pose-requantized-20261029/model'
```

导出文件为输出目录中的 `hand_pose_u85-1024_ded_256KiB_mixed.pte`。
构建脚本默认读取 `resources_downloaded/yolov8_pose/` 中的同名文件；
再次导出后可复制到该目录，或构建时通过 `-ModelPath` 指定新文件。

同一 PTE 的 FPGA 文件接收固件可用以下命令构建：

```powershell
.\scripts\windows\build-fpga-file-demo.ps1 -UseCase hand-pose
```

FPGA 固件输出到 `build/fpga-file/hand-pose/bin/mlek_yolov8_pose.axf`。

1. 将需要使用的模型固件构建好，确认 FVP、Python 和模型路径存在。
2. 先运行一次固定 WAV 或图片，再切换麦克风和摄像头。
3. 麦克风设备可用以下命令列出；选定后加 `-Device <编号>`。
4. 摄像头窗口获得焦点后再按空格；手部放在方形视野内，保持清晰和完整。
5. 一次启动一个音频演示；退出后再切换模型，避免占用同一端口或设备。
6. 拍摄截图时，区分持续刷新的 LIVE 预览与 LCD 上保持的捕获帧。

```powershell
.\resources_downloaded\windows-executorch\env\Scripts\python.exe `
    scripts/py/fvp_audio_demo.py --list-devices
```

最近的图像模型输入保存在构建目录的 `hand-pose-input-preview.png` 或
`qrcode-input-preview.png`。定位显示问题时，先检查这张图与终端坐标。

### 4.4 常用调整

| 现象或需求 | 操作 |
| --- | --- |
| 切换麦克风 | `-Device <编号>` |
| 收入过多环境声音 | ASR 启动时提高 `-VadDbfs`，用 `-LevelMeter` 看电平 |
| 调整句末停顿 | ASR 构建时调整 `-SilenceMs` |
| 调整音频积压上限 | 运行时 `-QueueMs <毫秒>` |
| 切换摄像头 | `-CameraIndex <编号>` |
| 摄像头后端切换 | `-CameraBackend dshow` 或 `msmf` |
| 手部保留完整视野 | `-Framing letterbox` |
| 换兼容模型 | 构建时 `-ModelPath <路径>`，再重新运行 |
| 单独保存构建 | 构建及运行使用相同的 `-BuildDirectory` |
| FPGA 更换输入文件 | 再次执行对应 `send-fpga-demo-file.ps1` 命令 |
| FPGA 上传前检查文件 | 上传命令加 `-ValidateOnly` |
| FPGA 上传中途断开 | 复位板卡后重新上传完整文件 |
| 无 LCD 的脚本检查 | 图片/WAV 模式使用 `-Headless -SmokeTimeoutSeconds 180` |

### 4.5 推荐现场展示顺序

先用 KWS 展示“连续输入与关键词响应”，再用 ASR 展示“句段收音与文本识别”；
随后用手部模型展示“空格拍摄与关键点骨架”，最后用二维码模型展示“换图与旋转框”。
每个模型准备一个固定样例作入口，再进行一次现场采集，便于说明输入、推理和可视化的关系。

### 4.6 验证状态与相关资料

| 项目 | 当前状态 |
| --- | --- |
| 四模型 FVP 演示 | 用户已完成实测 |
| 四模型 FPGA 文件调用 | 用户已完成实测 |
| FPGA 文件接收固件 | 四份均已构建，AXF／BIN 已生成 |
| ASR 完整下载包 | 已补齐三份 BIN、images.txt，且 BIN 与 AXF 校验一致 |
| 文件协议软件检查 | 握手、分块确认、字节保真、CRC 校验与结果返回通过 |
| OBB 绘图检查 | 旋转图片 FVP 验证和原生旋转绘图检查通过 |
| ASR 门控检查 | 数字电平、短脉冲、延续门限与静音释放检查通过 |
| 静态检查 | pre-commit、clang-format、cppcheck、PyLint 通过 |

相关资料：

- [FVP 音频链路说明](audio.md)
- [手部关键点演示说明](hand_pose.md)
- [二维码演示说明](qrcode.md)

本报告记录当前演示功能和复现方法。截图按第 2 节的占位补入；
模型版本、构建参数和二进制来源可通过对应目录的 `build-info.json` 追溯。
