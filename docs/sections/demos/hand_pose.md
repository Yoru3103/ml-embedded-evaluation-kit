<!--
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
<open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
-->
# 手部关键点 FVP 按键拍摄演示

PC 摄像头预览 → 中心正方形裁剪 → 224×224 RGB → 空格拍摄 → semihosting →
SSE-320 FVP / U85 模型推理 → LCD 输入图像、手部框、21 个关键点及 20 条骨架连线。
只绘制超过置信度阈值的关键点；一条连线的两个端点都达到阈值才显示。
LCD 保持本次结果，直到下一次拍摄开始处理。终端同时打印手部框及关键点坐标。

## 启动

从仓库根目录运行 PowerShell，首次构建：

```powershell
.\scripts\windows\build-hand-pose-fvp-demo.ps1
```

默认使用已有模型：
`resources_downloaded/yolov8_pose/hand_pose_u85-1024_ded_256KiB_mixed.pte`，
输入 224 × 224 RGB，split 四输出、像素坐标，U85 1024 MAC、Dedicated SRAM 256 KiB。
默认 LCD 按 1:1 显示 224 × 224 模型输入，便于观察手部关键点及连线。
构建目录为 `build/fvp-hand-pose/dedicated-256k`。

同一 PTE 的 FPGA 静态图片固件使用：

```powershell
.\scripts\windows\build-hand-pose-fvp-demo.ps1 -Target FPGA
```

产物为 `build/fpga-hand-pose/dedicated-256k/bin/mlek_yolov8_pose.axf`。

启动摄像头模式（默认；也可显式加 `-Camera`）：

```powershell
.\scripts\windows\run-hand-pose-fvp-demo.ps1
```

点击摄像头预览窗口使其获得焦点：

- **空格**：拍摄当前帧并识别。
- 推理期间继续预览，忽略新的拍摄，不积压帧。
- 提示 `Detections: ... | SPACE: capture again` 后可再次拍摄。
- **Q / Esc / 关闭预览窗口**：退出，清理摄像头和该次启动的 FVP。
- 原终端的 **Ctrl+C / Q** 也可退出。

启动时不自动拍摄，不重放旧请求。预览操作提示不进入模型输入。
默认从摄像头原生画面截取中心正方形，再缩放至模型输入；预览显示这张 224×224
448×448 的可调显示窗口（输入尺寸的 2 倍），操作提示放在图像外的黑条中。
空格提交的像素与当时预览一致。请把完整手部放在方形取景范围内。
预览标注 LIVE 并持续刷新，LCD 保持最近一次拍摄结果；比较定位时请使用
`hand-pose-input-preview.png`，避免拿不同时间的手势直接比较。
同一构建目录只运行一个实例。相机编号或后端可调整：

```powershell
.\scripts\windows\run-hand-pose-fvp-demo.ps1 -CameraIndex 1 -CameraBackend msmf
```

保留完整摄像头视野、恢复此前等比例补边方式：

```powershell
.\scripts\windows\run-hand-pose-fvp-demo.ps1 -Framing letterbox
```

此时预览仍显示模型实际接收的 224×224 图像及补边，而不是原始宽幅画面。

需要与二维码拍摄相同的 OpenCV / Pillow 环境，本机已经安装；如需补装 OpenCV：

```powershell
.\resources_downloaded\windows-executorch\env\Scripts\python.exe -m pip install -r scripts/windows/requirements-qrcode-camera.txt
```

## 文件输入与排查

不用摄像头，也可运行一张图片；正常模式会保持结果：

```powershell
.\scripts\windows\run-hand-pose-fvp-demo.ps1 -ImagePath resources/yolov8_pose/samples/test03.jpg
```

另一个终端可以向同一实例发送下一张图：

```powershell
.\scripts\windows\send-hand-pose-image.ps1 -ImagePath "D:\demo\hand.jpg"
```

无界面轻量检查（输出 `HAND_POSE_RESULT` 后退出）：

```powershell
.\scripts\windows\run-hand-pose-fvp-demo.ps1 `
    -ImagePath resources/yolov8_pose/samples/test03.jpg -Headless -SmokeTimeoutSeconds 180
```

构建目录中 `hand-pose-input-preview.png` 是实际提交给模型的 RGB 图像，
`hand-pose-fvp.log` / `hand-pose-fvp.err.log` 是 FVP 日志，
`hand-pose-camera.log` / `hand-pose-camera.err.log` 是摄像头日志。
自定义构建目录时，所有命令使用相同的 `-BuildDirectory`。

输入采用独立的 `HPD1` 请求格式，带 ID、尺寸、RGB 字节数和 CRC32，避免与二维码
灰度输入混用。预处理保持比例，默认裁剪、可选灰色补边；LCD 绘制模型输入对应的图像，
所以点与线不需要再换算回原始摄像头分辨率。
复用已有手部后处理及摄像头监督脚本；实现位于应用层与外围脚本，没有修改 HAL。
这里使用 FVP functional fast 模式，计时不能当成真实硬件性能。

轻量验收：8 项 Python 测试、灰度/RGB 请求解析 C++ 测试及选定文件静态检查通过。
FVP 无界面运行 `test03.jpg` 检测到 1 只手并打印全部 21 个关键点。
摄像头按键交互以模拟帧验证，完整 LCD / 预览窗口效果尚未人工验收。
