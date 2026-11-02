<!--
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
<open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
-->
# 中文 ASR 演示

使用当前 INT8 状态 Zipformer、U85-1024、Dedicated SRAM 768 KiB。
沿用已有模型及数值表现；替换兼容 PTE 后重新构建即可。

## FVP：WAV 或按键录音

```powershell
.\scripts\windows\build-fvp-audio-demo.ps1 -UseCase asr_zh
.\scripts\windows\run-fvp-audio-demo.ps1 -UseCase asr_zh `
    -Wav resources/asr_zh/samples/demo.wav
```

按键录音：

```powershell
.\scripts\windows\run-fvp-audio-demo.ps1 -UseCase asr_zh -Microphone
```

在弹出的窗口点击 **Start recording**，说完点击 **Stop / Recognize**。
整句录音通过已有 VSI0＋DMA 链路送入 FVP；不使用能量门控、降噪或额外滤波。
需要指定麦克风时加 `-Device <编号>`。每次录音最长 30 秒。

WAV 必须为 16 kHz、单声道、PCM16，最长 30 秒。识别结果在终端的
`ASR_ZH_TEXT ... text=...` 行显示中文；LCD 显示处理状态和结果提示。
结果保持到按 Q／Ctrl+C 退出，再次运行命令可识别另一句。
这是完整句子输入、内部增量推理的演示，尚非边录音边输出文字。

## FPGA：文件上传

```powershell
.\scripts\windows\build-fpga-file-demo.ps1 -UseCase asr_zh
.\scripts\windows\send-fpga-demo-file.ps1 -UseCase asr_zh `
    -InputPath resources/asr_zh/samples/demo.wav -Port COM5
```

先加载固件，`COM5` 替换为实际 UART0 端口，默认 115200。
同一固件可连续上传不同 WAV，每句会重置模型和 CTC 状态。

上传结束后，终端以毫秒汇总四项计时：

- `UART.Upload`：请求头开始发送至最后一个数据块确认，包含上传及 ACK 等待。
- `End-to-end`：板端从状态／特征准备到中文解码完成的实际处理耗时，不含传输和显示。
- `NPU.Total`：所有音频块的 NPU 驱动调用累计时间，包含提交和等待。
- `Global.Total`：主机脚本开始到收到结果的墙钟时间，包含文件准备、握手、上传及结果返回。

板端仅输出汇总，不逐块打印计时。更新计时功能后需重新加载 FPGA 固件。

- FVP AXF：`build/fvp-audio/asr_zh/bin/mlek_asr_zh.axf`
- FPGA AXF：`build/fpga-file/asr_zh/bin/mlek_asr_zh.axf`
- FPGA 三份 BIN：`build/fpga-file/asr_zh/bin/sectors/asr_zh/`
- FPGA 加载配置：`build/fpga-file/asr_zh/bin/sectors/images.txt`

模型替换时，构建命令加 `-ModelPath <新PTE路径>`。
INT8 状态模型同目录需带有匹配的 `StateQuantization.hpp`，外部张量接口保持一致。
