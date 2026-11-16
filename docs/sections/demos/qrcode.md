<!--
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
<open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
-->
# 二维码 / 条码 FVP 图像演示

本样例使用已有灰度 QR 模型，在 SSE-320 FVP 内执行 Ethos-U85 推理。
PC 负责读取 JPG/PNG/BMP、按 EXIF 旋转、转灰度、保持比例缩放及补边；
FVP 通过 semihosting 读取图像，在 LCD 显示输入图像、旋转检测框、类别和置信度。
结果保持到下一张有效图像开始处理。终端同步打印检测结果。

**当前是码的定位和分类，不包含码内文本或 URL 的解码。**
模型可识别 QR、BARCODE、DM 等类别；具体类别取决于模型及标签。
显示的是模型输入灰度图，默认 256 × 256，LCD 上进一步缩小显示。

## 构建

从仓库根目录执行 PowerShell：

```powershell
.\scripts\windows\build-qrcode-fvp-demo.ps1
```

默认使用已有的灰度四输出模型：
`resources_downloaded/yolo_QRcode/usr_quant_nhwc_angle3d/pte/best_bifpn_gray_nhwc_angle3d_u85_1024_384KiB.pte`。
配置为 Dedicated SRAM、384 KiB cache、U85 1024 MAC，默认构建目录为
`build/fvp-qrcode/dedicated-384k`。需要本机 ATFE、CMake/Ninja、SSE-320 FVP，
以及 `resources_downloaded/windows-executorch/env` Python 环境中的 Pillow。
这些路径均可通过脚本参数覆盖。

不要直接替换为 RGB 模型：本处理链要求输入 `[1,H,W,1]`，
三个 47 通道特征头及一个 angle 输出；尺寸和模型编译配置必须匹配。

## 启动和换图

```powershell
.\scripts\windows\run-qrcode-fvp-demo.ps1 -ImagePath resources/qrcode/smaples/01.bmp
```

默认弹出 FVP LCD。首张图由脚本准备后自动识别；在原终端按 **Ctrl+C** 或 **Q**
退出并清理该脚本启动的 FVP。在另一个终端发送下一张图：

```powershell
.\scripts\windows\send-qrcode-image.ps1 -ImagePath "D:\demo\next.jpg"
```

FVP 和模型保持运行，无须重新编译或重启。文件传输采用“最新一张”语义；
推理期间连续发送多张图时，中间尚未读取的图可能被覆盖，不保证逐张排队。
一次完整推理结束后才处理下一张图。

每个构建目录允许一个运行实例。若指定了其他 `-BuildDirectory`，启动和发送脚本
必须使用相同目录。原图不会被覆盖。最近的模型输入预览保存在构建目录的
`qrcode-input-preview.png`；终端日志为 `qrcode-fvp.log` / `qrcode-fvp.err.log`。

## 摄像头按键拍摄

```powershell
.\scripts\windows\run-qrcode-fvp-demo.ps1 -Camera
```

启动 FVP LCD 和 PC 摄像头预览。点击摄像头预览窗口使其获得焦点：

- **空格**：拍摄当前帧，送入 FVP 识别。
- 推理期间预览继续更新；此时空格不提交新请求，避免积压。
- 预览提示 `Detections: ... | SPACE: capture again` 后可以再次拍摄。
- **Q / Esc** 或关闭预览窗口：结束演示并清理 FVP。
- 原 PowerShell 终端的 **Ctrl+C / Q** 也可退出。

启动时不会自动拍摄，也不会重放上次运行的请求。LCD 的图像和检测结果只由
拍摄触发更新；等待下一次拍摄时保持原结果。预览画面上的操作提示不进入模型输入。
最近一次提交的模型输入仍保存在 `qrcode-input-preview.png`，不保存连续视频。

默认摄像头编号为 0、后端为 DirectShow，尝试采集 640 × 480；实际分辨率由摄像头
支持情况决定。多摄像头或打开失败时，可指定编号及后端：

```powershell
.\scripts\windows\run-qrcode-fvp-demo.ps1 -Camera -CameraIndex 1 -CameraBackend msmf
```

摄像头模式需要带 GUI 的 OpenCV Python 包（不是 headless 版本），安装到脚本使用的
同一 Python 环境。本机已安装；重建环境时执行：

```powershell
.\resources_downloaded\windows-executorch\env\Scripts\python.exe -m pip install -r scripts/windows/requirements-qrcode-camera.txt
```

摄像头日志在 `qrcode-camera.log` / `qrcode-camera.err.log`。
摄像头读取失败时检查设备编号、Windows 摄像头权限，以及是否被其他程序占用。
文件输入模式保持可用；摄像头模式复用原固件和 semihosting 请求格式，无须重新编译。

## 链路与轻量检查

图像 → PC 预处理 → 原子替换 `demo-input/qrcode-input.bin` →
FVP semihosting 读取并校验尺寸/长度/CRC32 → 输入量化 → NPU 推理 →
四输出解码/NMS → LCD 图像和框、终端结果。

每次发送生成独立的随机请求 ID（uint32），同一 ID 不重复推理。
只有新请求才清屏并绘制新图；轮询等待期间不清屏。
实现仅涉及应用层和外围脚本，没有修改 HAL。
这是按图片更新的演示，轮询与 semihosting 的运行时间不代表真实硬件实时性能。

无需打开 LCD 的单次冒烟检查：

```powershell
.\scripts\windows\run-qrcode-fvp-demo.ps1 `
    -ImagePath resources/qrcode/smaples/01.bmp -Headless -SmokeTimeoutSeconds 180
```

看到 `QRCODE_RESULT id=... count=...` 后自动退出；count=0 表示本帧未检测到码。
通常演示不要设置此超时参数，这样结果可持续显示并接收后续图片。

2026-10-29 本机轻量验收：固件构建、4 项 Python 测试、C++ 请求解析测试、
PowerShell 语法检查和选定文件的 pre-commit 检查通过。
SSE-320 FVP 无界面运行 `01.bmp` 得到 1 个 BARCODE，置信度约 0.716，
并输出 `QRCODE_RESULT`。该检查验证文件读取及推理链路，未人工检查 LCD 窗口外观。

摄像头扩展轻量验收：7 项文件输入/摄像头交互测试、Python 静态检查和
PowerShell 语法检查通过；本机摄像头 0 通过 DirectShow 成功读取 640 × 480 帧，
原文件输入的 FVP 回归检查通过。空格拍摄交互使用模拟帧验证，未做完整 GUI 人工验收。
