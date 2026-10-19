<!--
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
-->

# QR-code 灰度模型前后处理优化

本说明对应独立 `qrcode` use case，以及共享的 `YoloV8NewPostProcess`。
输入为 NHWC `[1,256,256,1]`；三个检测头各包含 32 个 DFL 通道和 15 个类别
logit，另一个输出包含 1,344 个角度 logit。置信度只由类别输出决定。

## 修改位置

| 文件 | 修改内容 |
| --- | --- |
| `source/app/use_case/qrcode/include/QrCodePreProcessing.hpp` | 灰度输入量化查表，缓存输入量化参数 |
| `source/app/use_case/qrcode/src/UseCaseHandler.cc` | 调用新的前处理，记录总耗时与各阶段耗时 |
| `source/lib/mlek/use_case/yolov8_detection/YoloV8PostProcessing.cc` | 缓存输出访问，类别先筛选，量化激活查表，拆分解码与 NMS |
| `source/lib/mlek/use_case/yolov8_detection/YoloV8PostProcessing.hpp` | 保留 `DoPostProcess()`，增加两个分阶段调用接口 |
| `source/app/profiler/Profiler.cc`、`include/Profiler.hpp` | 从同一计数周期的快照记录分段耗时，无需再次复位 PMU |
| `tests/processing/yolov8_detection/` | 全部灰度值的输入量化核对，量化输出与浮点参考解码比较 |

`DetectionResult` 继续使用原来的中心、宽高和角度字段，大小保持 28 字节。
模型、量化参数、类别数量、阈值和旋转 NMS 公式均沿用现有设置。
`source/lib/` 的后处理没有引入 HAL 依赖；读取 PMU 的代码位于应用层。

## 前处理：256 项输入映射表

BMP 中的一个像素 `p` 是 `uint8`，模型对应的浮点输入为 `p / 255`。
输入为 INT8 或 UINT8 时，转换公式为：

```text
q = clamp(lround((float(p) / 255.0F) / scale) + zero_point)
```

输入像素只有 256 种取值。新的前处理在第一次使用时计算全部映射，并在后续
图片中直接查表写入输入张量。保留原先浮点运算的顺序、`lround` 舍入规则和
饱和范围。输入类型、scale 或 zero point 改变时重新生成表。

表占 256 字节，随前处理对象保存在应用的栈帧中。INT8 表按 `q + 128` 保存，
写入时恢复为有符号值；UINT8 直接保存 `q`。FP32 输入仍按像素除以 255 写入。
每帧都会重新获取输入数据指针，允许框架在推理之间改变张量存储位置。

例如当前模型的 `scale=0.0037831603549420834`、`zero_point=-128`，不能直接
用 `p-128` 替代量化。新映射表遵循该模型的实际参数；像素超过 246 时会饱和。

## 后处理：缓存张量访问

旧的 `ReadTensorValue()` 每读取一个元素，都会查询元素数量、类型、数据指针
以及量化参数。TFLM 的 `GetNumElements()` 通过 `Shape()` 构造临时 vector。
四个输出完整读取一次有 `1344 × (47 + 1) = 64,512` 个元素。

新的原始头解码在每次调用开始时缓存每个张量的指针、元素数量、类型与量化参数。
循环内不再调用这些虚函数，也不再为单个元素构造 shape vector。下标检查及
FP32 有限值检查仍保留。每次解码重新获取指针，避免缓存失效的框架输出存储。
旧的单输出 `YoloV8PostProcess` 不使用这条新解码路径。

## 后处理：类别筛选提前

处理顺序从：

```text
所有位置的角度解码 → DFL → 类别 sigmoid → 阈值筛选 → NMS
```

调整为：

```text
类别筛选 → 通过阈值的位置才解码角度和 DFL → NMS
```

每个检测头使用一个正的 per-tensor scale，因此其整数类别 logit 的大小关系
与反量化后的大小关系一致。sigmoid 单调递增，只有遇到新的最大 logit 时才
需要更新类别分数；INT8/UINT8 分数直接查表。比较仍使用严格 `>`，保留类别
分数相等时优先选择前一个类别的规则，包括 sigmoid 浮点舍入到相同值的情况。

置信度低于阈值的位置直接跳过。旧实现每帧固定执行 43,008 次 DFL 指数运算、
20,160 次类别 sigmoid 和 1,344 次角度 sigmoid。新实现只对保留的位置读取
DFL 和角度，并保持 class-aware 旋转 NMS 的阈值与公式。

## 后处理：INT8/UINT8 查表

同一个输出头内，DFL softmax 的指数项可以写为：

```text
x = scale × (q - zero_point)
exp(x - max(x)) = exp(-scale × (max(q) - q))
```

`max(q)-q` 的范围是 0～255。每个头生成 256 项指数表，并按该头自己的 scale
查表。类别 sigmoid 也为每个头生成 256 项表；角度 sigmoid 使用角度输出自己的
量化参数生成独立表。不同输出的 scale/zero point 不混用。

三个检测头的指数表和类别表，加上角度表，共占约 7 KiB 堆内存，位于普通 C++
堆中，不占模型 tensor arena。表在后处理对象内复用，仅在首次使用或对应输出的
类型、scale、zero point 改变时重新生成。第一帧的 `Decode` 时间包含表分配和
1,792 次指数运算的初始化成本；后续参数不变时无需重新计算这些表。没有把
第一帧的初始化时间移到计时区间外。

候选框 vector 的容量在后处理对象内复用，本模型最多预留约 36.75 KiB。
FP32 输出沿用实际浮点 softmax 和角度计算，不套用 8 位表。

DFL 的整数差乘 scale 与原来先反量化再相减的表达式，在实数上等价，但可能
存在浮点舍入差异。验证时比较最终类别、置信度、角度、框和 NMS 结果，不要求
框坐标逐位相同。极接近 NMS 阈值的样本仍需按实际结果核对。

## 计时范围

| 日志名称 | 范围 |
| --- | --- |
| `PreProcessing` | 写入输入张量；第一张还包含输入查找表初始化 |
| `Inference` | 框架 `RunInference()`，含框架内部执行和数据搬运 |
| `Decode` | 输出检查、缓存属性、必要的表初始化、类别筛选、DFL/角度解码和候选框生成 |
| `NMS` | 候选框排序与类别感知的旋转 NMS |
| `InferenceAndPostProcessing` | 保留原来的推理加后处理总计时，包含分段快照开销 |

前处理使用独立的 PMU 计数周期。推理、解码与 NMS 共用一个连续周期，只在
阶段边界读取快照；`RecordProfiling()` 记录快照差值，不复位 PMU、不释放 NPU
计数引用。总计时在全部后处理结束后停止。打印和 LCD 绘图位于计时区间外。
分段读取快照也有开销，总计时与各段的和可能略有差异；毫秒值还存在取整误差。

## 验证与复现

处理测试覆盖 INT8、UINT8、FP32、NCHW/NHWC 输出和 8/16 个 DFL bin。
输入测试遍历全部 256 种像素，比较原量化公式与查表结果，并检查同一对象在
量化参数变化后重建映射。DFL 测试使用非二进制整数幂 scale 和非零零点，将
量化输出与同一输出反量化后的 FP32 解码结果比较。

```bash
cd /home/xx/ml-embedded-evaluation-kit
resources_downloaded/env/bin/cmake \
  -S tests/processing/yolov8_detection \
  -B /tmp/qrcode-processing-tests -DCMAKE_BUILD_TYPE=Release
resources_downloaded/env/bin/cmake --build /tmp/qrcode-processing-tests -j4
resources_downloaded/env/bin/ctest \
  --test-dir /tmp/qrcode-processing-tests --output-on-failure
```

已有的 FVP 单图构建目录可以直接增量编译，无需重新校准或运行 Vela：

```bash
resources_downloaded/env/bin/cmake \
  --build build-qrcode-fvp-single01 --target mlek_qrcode -j8
source /home/xx/FVP_Corstone_SSE-320/scripts/runtime.sh
/home/xx/FVP_Corstone_SSE-320/models/Linux64_GCC-9.3/FVP_Corstone_SSE-320 \
  -a /home/xx/ml-embedded-evaluation-kit/build-qrcode-fvp-single01/bin/mlek_qrcode.axf \
  -C mps4_board.subsystem.ethosu.num_macs=1024 \
  -C mps4_board.telnetterminal0.start_telnet=0 \
  -C mps4_board.uart0.out_file=- \
  -C mps4_board.uart0.shutdown_on_eot=1 \
  -C mps4_board.visualisation.disable-visualisation=1 \
  -C vis_hdlcd.disable_visualisation=1 --stat
```

### 实测记录

实测结果在同一个 `01.bmp`、同一模型、同一 U85 Z1024 Dedicated SRAM 配置下比较。
FVP 日志中的 CPU 时间与宿主机运行模拟器的 wall time 是不同指标；本文以固件
`CPU TOTAL` 和 `DURATION` 为比较依据。真实 MPS4 板卡耗时需另行实测。

前后处理优化不改变量化模型本身，也不代表原有 INT8 精度校验问题已经解决。

2026-10-15，Release 构建，FVP CPU 时钟 25 MHz，U85 Z1024，TA Mid，Dedicated SRAM，
NPU cache 393,216 字节；只编入 `01.bmp`。使用同一个 Vela 模型，第一帧包含查找表
初始化。下表毫秒数由 `CPU TOTAL / 25,000` 换算，保留三位小数：

| 阶段 | 优化前 CPU cycles | 优化后 CPU cycles | 优化后 CPU 时间（ms） |
| --- | ---: | ---: | ---: |
| 推理加后处理总计 | 24,932,238 | 1,993,742 | 79.750 |
| 前处理 | 原先未单独记录 | 405,517 | 16.221 |
| 推理 | 原先未单独记录 | 985,002 | 39.400 |
| 解码（含首次初始化） | 原先未单独记录 | 1,000,734 | 40.029 |
| NMS | 原先未单独记录 | 4,948 | 0.198 |

总计从 997.290 ms 降至 79.750 ms，约加速 **12.5 倍**，减少约 **92%**。
总计的 `DURATION` 整数日志分别为 997 ms 和 79 ms。NPU TOTAL 前后均为
972,089 cycles。前处理仍在该总计之外；目前没有优化前的独立前处理实测值。

两次均检出一个 `BARCODE`，置信度 `0.780681`，中心和角度一致。FVP 打印的
框宽由 `41.393066` 变为 `41.393059`，框高由 `22.080032` 变为 `22.080030`，
差异来自 DFL 浮点舍入。

另外，对原来的全部 7 张 BMP 提取同一模型的真实 INT8 输出，用旧版 C++ 浮点
解码器处理反量化数据，与新版 C++ INT8 查表解码器比较：检测数量、顺序、类别
和置信度一致，最大框坐标差约 `1.526e-5` 像素，角度一致。此项比较用于验证
后处理优化前后的一致性，不用于证明量化模型与原始 FP32 模型精度一致。

原始日志、固件及模型 SHA256、七张图的逐图比较报告保存在本地忽略目录：

```text
resources_downloaded/qrcode/processing_optimization/fvp_before.log
resources_downloaded/qrcode/processing_optimization/fvp_after.log
resources_downloaded/qrcode/processing_optimization/fvp_comparison.json
resources_downloaded/qrcode/processing_optimization/real_int8_parity.json
```

## ExecuTorch Conv/ReLU 量化边界修复

此项修改用于 PT2E 到 Ethos-U 的编译阶段，与前面的固件前后处理优化分开验证。
原始 PT 校准图中，8 个 C3Ghost 的 `cv2` 分支包含如下模式：

```text
DQ → Conv → ReLU → Q
```

这些分支后面的拼接涉及有符号数值，共享量化范围可能使 ReLU 输出零点为
`-28`、`-40` 等，而不是 INT8 最小值 `-128`。当前 ExecuTorch 后端识别
Conv/ReLU 量化融合时要求零点等于量化最小值；上述模式因此被分区器拒绝。
拒绝原因分别为卷积输出和 ReLU 输入未满足量化边界要求。卷积和 ReLU 算子
本身具备后端支持，单纯放宽精度验证阈值不能解决此问题。

### 修改位置与方法

| 文件 | 修改内容 |
| --- | --- |
| `scripts/qrcode_pte_passes.py` | `separate_quantized_relu()` 复用 ReLU 输出的 scale、zero point、范围和类型，在卷积与 ReLU 之间补充 Q/DQ 边界 |
| `scripts/qrcode_export_backends.py` | `lower_integer_pte()` 在分区前调用上述变换；保留最终 PTE 的全 NPU、整数 I/O 和内存检查 |
| `scripts/tests/test_qrcode_pte_passes.py` | 覆盖负值、零、半步舍入、饱和值，以及重复调用不再插入边界 |
| `scripts/tests/test_qrcode_pt_export.py` | 在端到端测试图中增加带 ReLU 的卷积分支和有符号拼接，验证最终序列化 PTE 无 CPU 算子 |

修复后的捕获图为：

```text
DQ → Conv → Q → DQ → ReLU → Q
            └─复用原 ReLU 输出的量化参数─┘
```

零点位于量化范围内，因此浮点零能精确表示。对于此模式，将数值先量化再
截断到零，与先经过 ReLU 再量化具有相同的整数结果；使用相同的量化网格，
避免引入一套新的校准参数。插入的 Q/DQ 用于明确编译图的量化边界，后端会
将其降低到整数 NPU 计算，并不要求在 CPU 上新增反量化操作。
仅修改用于编译的捕获图，原始权重、已保存的校准图和精度验证阈值保持原值。

### 2026-10-15 验证结果

复用 `direct_pt/pte_Dedicated_Sram/best_graphs/quantized.pt2`，该图来自
`best-bifpn.pt`、820 张校准图片、灰度 NHWC 256 输入、per-channel 权重量化。
编译目标为 U85 1024、`Ethos_U85_SYS_DRAM_Mid_1024`、Dedicated SRAM，
cache 为 393,216 字节。本次没有重跑校准。

- 实际插入 8 处边界；原始 7 张 BMP 在修改前后，四个主机输出头逐元素完全一致。
- 分区报告为 `No nodes rejected`。
- Vela 报告 CPU 算子 0、NPU 算子 156；最终 PTE 只有 1 个 `EthosUBackend`
  delegate，执行计划中的 CPU 算子列表为空。
- 序列化模型大小为 1,318,960 字节。此项结果证明编译与分区通过，尚未验证该
  PTE 在 FVP 或 FPGA 上的输出精度、实际耗时。

正常导出继续运行原来的 `scripts/export_model.py --to pte` 命令即可，修复自动
在校准后的编译阶段生效。七张图的一致性检查用于确认本次图变换的影响，不能
代替原始 FP32 对量化模型的完整验证集精度评估。
