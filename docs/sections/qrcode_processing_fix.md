<!--
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
-->

# QR-code 前后处理修复审查说明

## 分支与审查结论

修复分支：`codex/qrcode-angle-postprocessing`，基于已上传的 LCD 画框提交
`bc31e70b89023e1aeb50cc914ec0b9c0167cc1bb`。

对方分支 `origin/feature/dev_lzh` 的审查版本为
`c16f487279f0282c65b053bb0a50cf56ca2803e8`。发现以下问题，因此采用本分支修复：

1. 旧 RGB 部署链默认没有启用该分支新增的 OBB 开关，仍把第四输出作为 objectness。
   `YoloV8PostProcessing.cc` 的 `YoloV8NewPostProcess::DoPostProcess` 仍用它筛选候选框，
   并计算 `sigmoid(angle) * sigmoid(class)`；Python 的 `detection_values` 也保留该公式。
   例如类别 3 的 logit 为 2、角度 logit 为 -4，正确置信度应为 0.880797，旧公式得到
   0.015842，并可能错误选择另一锚点的类别 14。
2. 该分支新增的 `tests/use_case/yolov8_detection/LayoutTests.cc` 没有提供 Catch 的
   `CATCH_CONFIG_MAIN`，现有测试公共代码也没有补充入口。直接构建该测试出现
   `main` 和 Catch 实现符号未定义。其 Python 测试通过，但旧置信度测试仍接受错误的 0.25。

## 已确认的模型接口

| 模型 | 输入 | 三个检测头的通道拆分 | 第四输出 |
| --- | --- | --- | --- |
| 现有 `best.onnx` | `[1,3,256,256]` | 79 = 4 × 16 DFL + 15 类别 logits | `[1,1,1344]` 原始角度 logits |
| 灰度参考 `best_bifpn_gray_qop_int8_boardcal_20260819.onnx` | `[1,1,256,256]` | 47 = 4 × 8 DFL + 15 类别 logits | `[1,1,1344]` 原始角度 logits |

以上拆分来自终端 Conv / QLinearConv 权重的输出通道数。
两张图的角度输出均由三个单通道卷积分支展平后拼接，图内没有角度 sigmoid。

```text
类别 = argmax(class_logits)
置信度 = sigmoid(该类别的 logit)
角度（弧度） = (sigmoid(angle_logit) - 0.25) × pi
DFL bins = (检测头通道数 - 已配置类别数) / 4
```

当前部署仍使用现有 RGB 模型的 INT8 TFLite/PTE。灰度参考用于确认格式和验证解码；
本次没有替换部署权重。QOperator 图可以在 ONNX Runtime 上运行，本文不宣称已完成其
onnx2torch 转换链。

## 修改位置

路径均相对于仓库根目录。函数名可以直接搜索定位。

| 文件 | 定位 | 修改内容 |
| --- | --- | --- |
| `source/lib/mlek/use_case/yolov8_detection/YoloV8PostProcessing.cc` | `YoloV8NewPostProcess::DoPostProcess` | 第四输出按角度处理；类别独立 sigmoid；根据类别数推导 8/16 等 DFL bin；每个输出使用自身量化参数；旋转中心偏移并保留 OBB 几何。 |
| 同上 | `CalculateRotatedIoU` / `ApplyClassAwareNms` | 有角度时按 OBB probabilistic IoU 进行同类别 NMS。三输出普通框及原有单输出路径继续使用普通框 IoU。 |
| `source/lib/mlek/use_case/yolov8_detection/YoloV8PostProcessing.hpp` | `YoloV8PostProcessParams` | `angleIsLogits` 默认 true；false 可接收已解码的弧度。 |
| `source/lib/mlek/use_case/yolov8_detection/DetectionResult.hpp` | `DetectionResult` | 统一为 score、classId、centerX、centerY、width、height、angleRadians 共 7 个字段，28 字节。普通框默认角度 0，移除重复包围矩形和 rotated 标记。 |
| `source/lib/mlek/use_case/yolov8_detection/YoloV8PreProcessing.cc` | `GetInputDescription` / `DoPreProcess` | 支持 1/3 通道、NCHW/NHWC；模型需要单通道时才转灰度；按 tensor scale/offset 写 FP32/INT8/UINT8。 |
| `source/app/use_case/yolov8_detection/src/UseCaseHandler.cc` | `GetInputImageDimensions` / 推理循环 | 识别单通道输入；外层以 InferenceAndPostProcessing 合并记录推理和后处理，直接调用 model.RunInference；结果行清除区域按 HDLCD 9×15 字符格对齐。 |
| `source/app/use_case/yolov8_detection/src/DetectionDrawing.cc` | `DrawDetectionBoxes` / `DrawLine` | 中心、宽高、弧度统一转换四个顶点；线段裁剪后逐像素连成旋转框，角度 0 走相同流程；仅裁剪显示，不改变后处理几何。 |
| `source/hal/source/components/display/source/hdlcd_mps4/hal_display_mps4.c` | `publish_frame_buffer` | 图片、文字、填充矩形和清屏共用缓存发布，确保 HDLCD 从 DDR 看到更新。 |
| `source/hal/source/platform/mps4/source/timer_mps4.c` | `platform_get_counters` / `get_tstamp_milliseconds` / `get_mps4_core_clock` | CPU TOTAL 和 DURATION 共用一次 64 位 SysTick 快照；取消超过 1 秒切换 100 Hz 来源；SCC 为零时 FVP 使用启动代码的 SystemCoreClock，默认 25 MHz；修正 SysTick 周期的额外一拍。 |
| `scripts/model_export/qrcode/contract.py` | `describe_model` | 从原始及 QOperator ONNX 图读取真实 DFL、类别和角度接口，拒绝已激活角度等不一致格式。 |
| `scripts/model_export/qrcode/onnx_utils.py` | `detection_values` / `decode_boxes` / `detection_metrics` | Python 校验采用相同的类别置信度和旋转框公式；类别数明确传入；记录量化后的最高类别。 |
| `scripts/model_export/yolo/utils.py` | `load_image` | 居中裁剪匹配 `gen_rgb_cpp.py`；裁剪后按模型通道转换 RGB/L。 |
| `scripts/model_export/qrcode/onnx.py`、`scripts/model_export/yolo/pte.py`、`scripts/model_export/tools/compile_fastcache.py` | 样本加载、校准、接口读取 | 传递模型输入通道和类别数，共用经过检查的模型接口。 |
| `scripts/model_export/qrcode/onnx.py`、`scripts/model_export/qrcode/onnx2tf.py` | `validate_common_options` / 转换错误处理 | 兼容工作区并行新增的导出入口；拆分公共参数校验以通过 PyLint；转换器 SystemExit 转为 RuntimeError，确保失败状态写入报告。 |
| `tests/processing/yolov8_detection/`、`CMakeLists.txt` | `yolov8-processing` | 独立 C++ 前后处理测试，验证统一结果、普通框角度 0 和 NMS，接入 native CTest。 |
| `tests/hal/display/DetectionDrawingTests.cc` | `test_detection_drawing` | 调用真实 MPS4 HAL，核对 0°、±45°、90°、缩放、裁剪、图像区域边界及缓存发布；覆盖 RGB24/ARGB32 两种 framebuffer。 |
| `tests/hal/timer/`、`CMakeLists.txt` | `mps4-timer-fvp` / `mps4-timer-fpga` | 核对 SCC 和回退频率、999/1000/1001 ms 连续性、长时 64 位换算、SysTick 待处理溢出和中断屏蔽恢复；验证板载 100 Hz 值不影响 DURATION。 |
| `scripts/tests/test_qrcode_onnx_export.py`、`scripts/tests/test_qrcode_contract.py` | 回归测试 | 角度不影响分数/类别；8/16 bin；模型图接口；居中裁剪及灰度；原有导出校验。 |
| `scripts/model_export/tools/validate_processing.py` | `reference_decode` / `check_detections` | 真实模型输出送入实际 C++ 后处理，与 Torch softmax 和 Ultralytics `dist2rbox`、`batch_probiou` 独立核对。 |
| `docs/sections/yolov8_model_export.md` | QR-code ONNX export | 更正原 objectness 说明及置信度公式，注明旧报告的分数指标需要重新验证。 |

现有文件还按仓库 `clang-format` 规则格式化，可用 `git diff -w` 减少空白变化的干扰。
`scripts/model_export/yolo/pte.py` 原有的导出说明注释保留。

## 验证和复现

### 自动测试

```bash
PYTHONPATH=scripts resources_downloaded/env/bin/python \
    -m unittest discover -s scripts/tests -v

resources_downloaded/env/bin/cmake -S tests/processing/yolov8_detection \
    -B build-qrcode-processing-tests
resources_downloaded/env/bin/cmake --build build-qrcode-processing-tests --parallel 4
resources_downloaded/env/bin/ctest --test-dir build-qrcode-processing-tests --output-on-failure
```

Python 17 项测试通过。C++ 覆盖 FP32/INT8/UINT8 × NCHW/NHWC × 8/16 bin，
以及原始角度/弧度、三输出无角度、错误类别数、RGB 矩形输入、灰度和旋转 NMS。
native 工程中的 6 项 CTest 通过，包含两种 MPS4 framebuffer 像素格式和两种计时平台测试。

### 真实模型交叉核对

```bash
PYTHONPATH=scripts resources_downloaded/env/bin/python -m model_export.tools.validate_processing \
    --model resources_downloaded/yolo_QRcode/best.onnx \
    --model resources_downloaded/yolo_QRcode/best_bifpn_gray_qop_int8_boardcal_20260819.onnx \
    --decoder build-qrcode-processing-tests/yolov8_processing_tests \
    --tflite resources_downloaded/yolo_QRcode/tflite/best_int8.tflite \
    --pte-graph resources_downloaded/yolo_QRcode/pte_Dedicated_Sram/best_graphs/quantized.pt2 \
    --report resources_downloaded/yolo_QRcode/postprocessing_validation.json
```

每条链检查 6 张仓库样本，共 24 组。逐个返回框检查最高类别、对应 sigmoid 分数、
旋转框中心/宽高/角度和同类别 NMS；同时检查 Python 的框包围矩形。
量化输出可能出现最高分并列，此时检查每个返回框对应锚点的正确类别，而非要求
不同排序算法在并列候选中选择同一锚点。

报告记录模型 SHA-256、接口、每张图的类别、分数和最大误差。
24 组核对全部通过：类别不匹配数为 0，C++ 置信度最大误差约 `5.99e-8`，
旋转框中心/宽高/角度的最大数值误差约 `3.05e-5`。
当前标签文件为 `class_0` 至 `class_14` 占位名称，类别索引按模型通道顺序。
这些核对验证解码正确性；样本没有真实标注，不能用于证明检测准确率。

### 灰度 BMP 与合并性能计时

新样本位于 `resources/qrcode/smaples/`：`01.bmp` 至 `07.bmp`，均为 256×256 的
Pillow `L` 灰度图片。测试沿用现有 RGB INT8 部署模型，因此生成器把灰度值复制成
三个 RGB 通道；本报告不是单通道灰度参考模型的 NPU 性能。

将上述主机核对命令增加 `--samples resources/qrcode/smaples`，并将报告路径改为
`resources_downloaded/yolo_QRcode/grayscale_bmp_validation.json`，可复现 28 组 BMP 解码核对。
原 PNG 24 组与新增 BMP 28 组均通过 C++ 解码检查。

计时边界为：

```cpp
profiler.StartProfiling("InferenceAndPostProcessing");
model.RunInference();
postProcess->DoPostProcess();
profiler.StopProfiling();
```

失败路径也停止计时。预处理、结果打印和 LCD 绘制在计时之外。
每张图只出现一个 `InferenceAndPostProcessing` 记录。

两条 FVP 部署链在 U85 Z1024、Dedicated SRAM、393216 字节缓存、TA Mid 下各运行
全部 7 张 BMP。CPU 周期数包含推理及后处理：

| 框架 | 平均 CPU 周期 | 最小 | 最大 |
| --- | --- | --- | --- |
| TFLM | 40,699,294 | 40,147,263 | 40,797,754 |
| ExecuTorch | 16,602,101 | 16,598,200 | 16,606,928 |

这是计时修正前保存的 FVP 周期测量；原日志的毫秒值不能用作延迟结论。
毫秒换算需要与 SysTick 相同的 CPU 频率：`ms = cycles / (MHz × 1000)`。
逐图结果、检测框、模型与固件 SHA-256 以及原始日志位置保存在
`resources_downloaded/yolo_QRcode/grayscale_bmp_performance.json`。
七图测试固件已保存为两个 FVP 构建目录中的 `bin/mlek_qrcode_bmp_benchmark.axf`。
在相同参数下运行这些快照可以复现日志。

旧 TFLM 的编译模型在部分 BMP 上与主机 LiteRT 有数值差异。例如 `02.bmp`：
主机最高分约 0.402189，NPU 原始输出复算最高分约 0.170032，低于 0.25 阈值。
设备推理前的 INT8 输入与主机量化输入的 FNV-1a 校验和均为 713616725；源 BMP
与固件样本数组逐字节一致。提取原始 NPU 输出后，独立 C++ 解码同样返回零框。
差异发生在编译模型输出，本次没有改动部署权重，也不把它宣称为精度验证通过。
审计证据在 TFLM FVP 目录的 `input-audit-fvp.log`、`02-raw-audit-fvp.log` 和
`02-npu-outputs.bin`。ExecuTorch 的 BMP 输出与主机 PT2E 核对一致。

### MPS4 计时修正

原代码在短时间内按 SysTick 换算，超过 1 秒改为 `CLK100HZ × 10`，
使同一测量的 CPU 周期和毫秒来自不同计数器。FVP 的 SCC 时钟寄存器返回零时，
原回退频率 32 MHz 也与 Cortex-M85 启动代码和标准 SSE-320 FVP 的 25 MHz 不一致。

现在 `CPU TOTAL` 与 `DURATION` 使用同一次 64 位 SysTick 快照，所有时长均按
核心频率换算。优先读取 SCC；SCC 返回零时 FPGA 使用 FI101 的 50 MHz，
FVP 使用 `SystemCoreClock`（默认 25 MHz）。如果手动修改 FVP 的核心时钟，
应同步配置启动代码中的 `SystemCoreClock`。换算按整数商和余数分开计算，避免
`cycles × 1000` 溢出，也不预先截断 Hz 到 kHz。

计时范围为 `model.RunInference()` 开始到 `DoPostProcess()` 结束，包含等待 NPU
和框架内部拷贝；图片转换、量化并写入 input tensor、打印和 LCD 绘制位于范围外。
这里的 CPU TOTAL 是经过的核心时钟周期，包含 CPU 等待时间，不代表 CPU 忙碌周期。

修复后实测：TFLM 的 `01.bmp` 为 40,147,748 cycles / 1605 ms；
ExecuTorch 的旋转框 `1.png` 为 16,606,634 cycles / 664 ms。
两个结果满足 25 MHz 的换算关系。日志保存为各 FVP 构建目录中的 `timing-fvp.log`。
这次修正测量方式，尚未优化 DFL、逐元素张量读取或 NMS 的运行速度。
FVP 用于功能与趋势验证，最终延迟仍需在 FPGA 或目标硬件上测量。

计时回归测试也可独立运行：

```bash
resources_downloaded/env/bin/cmake -S tests/hal/timer -B build-mps4-timer-tests
resources_downloaded/env/bin/cmake --build build-mps4-timer-tests --parallel 4
resources_downloaded/env/bin/ctest --test-dir build-mps4-timer-tests --output-on-failure
```

### 旋转框 LCD 验证

MPS4 FPGA 和 SSE-320 FVP 的 TFLM/ExecuTorch 共四个固件构建通过。
使用旧 PTE 模型和 `1.png` 专门核对可见旋转角度：类别 3、分数 0.869708，
`centerX=131.268219`、`centerY=85.630623`、`width=104.586731`、
`height=107.160233`、`angleRadians=0.338686`（约 19.4°）。

实际帧缓冲有 199 个被改成绿色的边框像素。按 Ultralytics 四顶点公式独立核对：
所有这些像素到真实四条边的最大距离约 0.736 像素，边线上采样点到绘制像素的
最大距离约 0.815 像素，四条边完整；边框之外的原图逐像素一致。
FVP 日志无 ERROR。绿色 RGB565 展开后为 RGB `(0,252,0)`。

- `build-qrcode-processing-tests/rotated-display-validation.json`
- `build-qrcode-fvp-pte-dedicated-z1024-ta-mid/rotated-demo-fvp.log`
- `build-qrcode-fvp-pte-dedicated-z1024-ta-mid/rotated-lcd.png`

上述 LCD 验证来自 FVP 的实际 HDLCD 帧缓冲；本次没有实体开发板执行记录。
