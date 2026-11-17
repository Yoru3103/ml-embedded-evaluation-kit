# Windows FVP 与 FPGA 演示

<!-- SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

首次配置 Windows 环境以及静态图像分类验证见 [Windows 快速入门](../windows.md)。
从仓库根目录执行各指南中的 PowerShell 命令。构建固件使用 `build.ps1`，所有 Windows
FVP 演示使用 `run-fvp.ps1`。有摄像头或麦克风的演示可先用仓库里的图片或 WAV 检查推理链路。

| 演示 | Profile | 输入与操作指南 |
| --- | --- | --- |
| 图像分类、33 类手势、COCO 检测 | `Vision` | [静态图片、VSI 摄像头和 KEY0](../../../scripts/vsi/README.md) |
| 视觉性能测量 | `VisionBenchmark` | [单张图片、FVP 时序和 MPS4 打包](../../../scripts/vsi/README.md#fixed-image-timing-benchmark) |
| 中文 KWS、英文 ASR | `AudioDemo` | [WAV、麦克风和分句参数](audio.md) |
| 中文 ASR | `AudioDemo` | [WAV、麦克风和 FPGA 文件上传](asr_zh.md) |
| 手部关键点 | `HandPose` | [图片、摄像头和视野设置](hand_pose.md) |
| 二维码与条码检测 | `QrCode` | [换图和摄像头拍摄](qrcode.md) |
| RNNoise 降噪 | `NoiseReduction` | [WAV 与输出文件](../../use_cases/noise_reduction.md) |
| 英文 Conformer 串口演示 | `ConformerUart` | [FPGA UART 与 FVP TCP 上传](../conformer_uart_demo.md) |
| Pico Face | `PicoFace` | [模型导出、构建和运行](../../use_cases/pico_face.md) |
| TinyStories | `TinyStoriesFvp` / `TinyStories` | [固件和 Raw TCP 交互](../../../source/app/use_case/tinystories/DEMO.md) |
| MiniMind2 Small | `MiniMind2Test` | [模型资料、固件和串口交互](../../../source/app/use_case/minimind2_small/README.md) |

## MPS4 FPGA 文件上传

中文 KWS、英文 ASR、中文 ASR、手部关键点和二维码支持常驻文件接收固件。
切换模型时加载对应固件。更换同一模型的输入时，再次运行发送脚本即可。
以下示例选择中文 ASR，其他模型的输入与模型准备要求见上表。

```powershell
.\scripts\windows\build.ps1 -Framework pte -Action Setup
.\scripts\windows\build.ps1 -Profile FpgaFileDemo -UseCase asr_zh
```

`-UseCase` 可选 `kws_zh`、`asr`、`asr_zh`、`hand-pose`、`qrcode`。
构建输出在 `build/fpga-file/<UseCase>/`。将同一次构建的三份 BIN
（`boot.bin`、`bram.bin`、`ddr.bin`）复制到板卡盘的 `SOFTWARE/`，
将 `bin/sectors/images.txt` 复制到当前 FPGA image 目录。
Revision B 的目录示例为 `MB/HBI0376B/FI101`，以板上的 image 为准。
安全弹出板卡盘并复位，等待固件初始化。有关板卡和加载流程见 [MPS4 部署](../deployment.md#mps3--mps4-fpga-board)。

板卡准备好后，关闭占用 UART0 的串口终端，将 `COM5` 换成实际端口，再发送样例：

```powershell
.\scripts\windows\send-fpga-demo-file.ps1 -UseCase asr_zh `
    -InputPath resources/asr_zh/samples/demo.wav -Port COM5
```

发送脚本使用 115200 波特率，并输出固件返回的结果。加 `-ValidateOnly` 可在连接
板卡前检查和预处理输入。WAV 需要单声道、16 kHz、PCM16，最长 30 秒。
手部输入会转为 224×224 RGB，二维码输入会转为 256×256 灰度图。
若文件发送中途中断，复位板卡后重新发送。
