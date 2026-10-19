<!--
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
-->

# QR-code 模型内输出反量化与独立计时

输入继续使用 INT8 NHWC `[1,256,256,1]`，BMP 前处理继续读取模型的输入 scale
和 zero point，使用原有映射表。不向模型添加输入量化算子。

输出可选择两种形式，网络计算均在 Ethos-U 上：

| 配置 | 模型输出 | 输出反量化 | 后处理 |
| --- | --- | --- | --- |
| `--output-type int8`，默认 | 四个 INT8 原始输出头 | 沿用现有解码器中的反量化和查表 | INT8 优化解码、旋转 NMS |
| `--output-type float32` | 四个 FP32 原始输出头 | 模型内四个独立反量化算子 | 读取 FP32、DFL/角度解码、旋转 NMS |

没有新增应用层全量反量化模式。四个头均完整反量化，涵盖全部 1,344 个位置，
不根据置信度、候选框数量或最终目标数量裁剪算子的工作量。三个检测头各有
47 通道，角度头有 1,344 项，总计 64,512 个输出元素。INT8 输出数据为
64,512 字节，FP32 输出数据为 258,048 字节，框架还需要保留相应的中间缓冲。

## 导出或复用已有模型

### ExecuTorch

原来的原生 PT 到 PTE 命令增加 `--output-type float32`，并使用新的输出目录。
例如沿用此前的 820 张校准图、543 张验证图和用户设置的验证阈值：

```bash
resources_downloaded/env/bin/python scripts/export_model.py \
  --model resources_downloaded/yolo_QRcode/best-bifpn.pt \
  --to pte --image-size 256 --io-layout nhwc --output-type float32 \
  --preprocessing mlek-crop \
  --calibration-dir resources_downloaded/qrcode/calibration_merged/images/calibration \
  --calibration-limit 0 \
  --validation-dir resources_downloaded/qrcode/calibration_merged/images/validation \
  --validation-limit 543 --threads 4 \
  --target ethos-u85-1024 --system-config Ethos_U85_SYS_DRAM_Mid_1024 \
  --memory-mode Dedicated_Sram --arena-cache-size 393216 \
  --max-score-error 0.9 --min-box-iou 0.1 \
  --output-dir resources_downloaded/qrcode/direct_pt_float_outputs
```

上述误差阈值沿用先前用户的实验配置，不代表推荐的部署精度要求。
默认 INT8 输出的命令及验证要求保持原状。

`lower_integer_pte()` 继续移除输入量化边界，并保留输入的量化配置方法。
FP32 输出模式保留输出反量化，将其替换为 Cortex-M 运行库的
`cortex_m::dequantize_per_tensor.out`。导出及序列化检查只允许这四个末端 CPU
算子，要求它们直接连接 NPU 输出与最终输出；卷积、激活等网络计算回退到 CPU
仍然报错。输出 JSON 区分算子种类数、实际调用次数和网络 CPU 算子数。

### TFLite

现有 QR-code ONNX 到 TFLite 命令同样支持 `--output-type float32`，适用于
ONNX2TF 和 LiteRT 两种转换方式。原始 INT8 文件保留，浮点输出文件增加
`_float_outputs` 标识。该参数对应量化网络的输出类型，独立的全浮点参考模型
仍作为原来的精度校验基准。

已有 Vela 模型可直接追加四个输出反量化，无需再次校准或编译：

```bash
PYTHONPATH=scripts resources_downloaded/env/bin/python -m model_export.qrcode.output_io \
  --model resources_downloaded/qrcode/merged_onnx2tf/tflm_Dedicated_Sram/best_gray_full_integer_quant_vela.tflite \
  --output resources_downloaded/qrcode/output_dequantization/tflm_Dedicated_Sram/best_float_outputs_vela.tflite
```

原来的输入、INT8 输出张量、量化参数和 NPU 数据保留，新增四个 FP32 张量和
四个 `DEQUANTIZE` 节点。重新序列化时维持命令流和权重所需的 16 字节对齐。
Vela 的 `OfflineMemoryAllocation` 元数据同步扩展，
新增张量使用 `-1`，由 TFLM 的在线内存规划器分配，原有离线偏移保持不变。
结构校验要求一个或多个 Ethos-U 节点，以及直接连接最终输出的四个反量化节点。

把新的模型路径传给 `-Dqrcode_MODEL_PATH=...`，重新执行 CMake 配置，再编译
`mlek_qrcode`。模型内容改变后也需要重新配置，以重新生成模型 C++ 数组。
也可使用下面的完整矩阵脚本，自动生成和选择浮点输出模型。

## 完整矩阵构建脚本

新增两个入口，复用原有矩阵脚本的编译、构建和 FVP 校验流程，固定使用
`--output-type float32`；原有两个入口仍默认 INT8 输出。两种输出模式默认写入
不同的构建目录，便于保留对照固件。

| 入口 | 输入文件 | 默认输出目录 |
| --- | --- | --- |
| `scripts/qrcode/build_qrcode_tflite_dequant_matrix.sh` | 未经 Vela 编译的 INT8 TFLite | `build-qrcode-tflite-dequant-matrix/` |
| `scripts/qrcode/build_qrcode_pte_dequant_matrix.sh` | 原始 OBB PT 权重 | `build-qrcode-pte-dequant-matrix/` |

在项目根目录运行：

```bash
bash scripts/qrcode/build_qrcode_tflite_dequant_matrix.sh \
  resources_downloaded/qrcode/merged_onnx2tf/tflite/onnx2tf_outputs/best_gray_full_integer_quant.tflite

bash scripts/qrcode/build_qrcode_pte_dequant_matrix.sh \
  resources_downloaded/yolo_QRcode/best-bifpn.pt
```

每条命令生成 Shared SRAM，以及 Dedicated SRAM 64/128/256/384/768 KiB 六种模型，
各自构建 FPGA 和 FVP 固件，共 12 份。使用全部样本，当前原始目录为七张 BMP；
Release、非交互运行、CPU profiling 开启。FPGA 固件仍需另行下载到开发板。
各变体输出 `configure.log`、`build.log`，固件位于 `fpga/配置/bin/mlek_qrcode.axf`
或 `fvp/配置/bin/mlek_qrcode.axf`。

共同支持 `--dry-run` 预览命令、`--models-only` 仅生成并检查模型，
以及 `--run-fvp` 在构建后运行六份 FVP 固件并保存 `fvp.log`。
样本目录、标签、输出目录和并行度沿用原脚本的 `QRCODE_SAMPLE_PATH`、
`QRCODE_LABELS_FILE`、`QRCODE_MATRIX_ROOT` 和 `BUILD_JOBS`。

TFLite 先编译全 INT8 模型，检查 NPU 配置和 Vela CSV，然后追加并检查四个
末端 `DEQUANTIZE`；网络出现其他 CPU 算子仍会失败。已有六种 INT8 Vela 模型
可以复用，无需重编译网络：

```bash
bash scripts/qrcode/build_qrcode_tflite_dequant_matrix.sh \
  --npu-model-root build-qrcode-tflite-matrix/models --run-fvp
```

PTE 默认只校准一次，再用公共 PT2 生成六种编译配置；只允许四个末端 CPU
反量化调用，校验序列化执行计划和输出 FP32 类型。已有原矩阵的公共 PT2 可复用：

```bash
bash scripts/qrcode/build_qrcode_pte_dequant_matrix.sh \
  resources_downloaded/yolo_QRcode/best-bifpn.pt \
  --quantized-pt2 build-qrcode-pte-matrix/common/quantized.pt2 --run-fvp
```

公共缓存必须保留同目录的 `pt2_report.json`、`references.npz`，并且源权重、
校准和验证数据、预处理及量化配置必须相同；复用时仍执行缓存一致性和量化误差
校验。`--npu-model-root` 则只接受已经导出为 FP32 输出的完整六种 PTE 及各自的
`export_report.json`，不会把原来的 INT8 PTE 误用为本次对比模型。

本次已实际生成并检查六种 TFLite 浮点输出模型，以及使用原有公共 PT2 导出的
六种 PTE 浮点输出模型。PTE 结果在
`build-qrcode-pte-dequant-matrix/models/`，可直接复用构建固件：

```bash
bash scripts/qrcode/build_qrcode_pte_dequant_matrix.sh \
  --npu-model-root build-qrcode-pte-dequant-matrix/models
```

两套命令预览均检查了 12 份固件配置的模型路径、Release 和 CPU profiling 开关；
没有在本次脚本验证中批量构建全部 24 份固件。

PTE 的校准目录、验证目录和阈值沿用原脚本环境变量。当前默认
`QRCODE_MAX_SCORE_ERROR=0.9`、`QRCODE_MIN_BOX_IOU=0.1`，属于此前实验采用的
宽松阈值，不代表精度评估通过。

后处理的 `OutputReader` 按实际张量类型选择路径：FP32 直接读取 `float`，
不应用输出 scale/zero point，也不创建 INT8 sigmoid/DFL 查表；继续计算类别
sigmoid、DFL softmax、角度映射和旋转 NMS。FP32 输出仍是原始检测头，
并非已解码的框或最终置信度。

## 计时边界

需要 `-DCPU_PROFILE_ENABLED=ON`。固件日志按图片输出：

| 计时名称 | 范围 |
| --- | --- |
| `PreProcessing` | BMP 到输入 INT8 张量，沿用现有实现 |
| `Inference` | 完整框架调用，包括 NPU 推理、模型内输出反量化，以及框架自己的搬运/调度 |
| `OutputDequantization` | 四个模型内反量化算子的实际运行区间之和，属于 `Inference` 的子集 |
| `Decode` | 输出访问、类别筛选、DFL 和角度解码 |
| `NMS` | 旋转框抑制 |
| `InferenceAndPostProcessing` | 从框架调用开始，到解码和 NMS 结束 |

仅 `Inference` 记录 NPU 周期和事件计数。`PreProcessing`、
`OutputDequantization`、`Decode`、`NMS` 和 `InferenceAndPostProcessing`
仅记录 CPU 周期与耗时；这些阶段的快照不读取 NPU PMU 寄存器。
完整 PMU 快照仅在框架调用前后各读取一次，供 `Inference` 使用。
外层计时仍统一初始化和释放 PMU，保证嵌套的 NPU 快照与 CPU 计时不被重置。

**不要把 `OutputDequantization` 再加到 `Inference` 上。**
默认 INT8 输出没有独立的模型内反量化调用，其反量化/查表融合于 `Decode`，
不会打印一个虚假的独立反量化耗时。比较整体性能应使用
`InferenceAndPostProcessing`，同时参考 `Decode`：INT8 查表路径和 FP32 路径的
解码实现不同，不能仅用反量化耗时推导两者总耗时。

TFLM 通过 `MicroProfilerInterface::BeginEvent/EndEvent` 选择 `DEQUANTIZE`。
ExecuTorch 通过 `EventTracer` 的算子调用事件，结合序列化执行计划判断实际算子
名称，只采集输出反量化。两者均在应用层读取 PMU 快照，不在回调中打印或分配
统计容器；外层 PMU 计时期间不重置计数器。推理完成后再合计四段差值，每张图
记录一次 `OutputDequantization`，要求恰好采集到四次调用。

此计时包含算子调用边界与 PMU 采样开销。毫秒计数器分辨率较粗，各头的整数
毫秒差值相加可能出现舍入误差，精确比较优先使用 `CPU TOTAL` cycles 除以
实际 CPU 时钟。FVP 的主机 wall time 不能作为固件耗时。

## 修改位置

| 文件 | 职责 |
| --- | --- |
| `scripts/model_export/qrcode/onnx.py` | `--output-type` 选项 |
| `scripts/model_export/qrcode/output_io.py` | TFLite 输出包装、离线内存元数据扩展、边界执行检查和独立 CLI |
| `scripts/model_export/qrcode/backends.py` | PTE 保留输出反量化、整数输入配置及序列化检查 |
| `scripts/model_export/qrcode/onnx_utils.py` | 输入 INT8、输出 INT8/FP32 的形状及类型契约 |
| `scripts/model_export/qrcode/tflite_utils.py`、`scripts/model_export/qrcode/onnx2tf.py` | 转换、验证、编译和发布输出类型变体 |
| `scripts/qrcode/build_qrcode_{tflite,pte}_dequant_matrix.sh` | 固定 FP32 输出的完整矩阵入口，使用独立默认构建目录 |
| `scripts/qrcode/build_qrcode_{tflite,pte}_matrix.sh` | 共享矩阵流程，按输出类型验证，PTE 支持复用公共 PT2 |
| `source/app/use_case/qrcode/include/OutputDequantizationProfiler.hpp` | 两套框架的算子回调、四段 PMU 区间汇总 |
| `source/app/use_case/qrcode/src/MainLoop.cc` | 注入框架 profiler/tracer，管理生命周期 |
| `source/app/use_case/qrcode/src/UseCaseHandler.cc` | 每图复位、记录模型内反量化指标，保留其他阶段 |
| `source/app/profiler/` | 选择 CPU 计时，支持完整快照与 CPU 快照之间的阶段差值 |
| `source/hal/{include,source}/` | 提供 CPU 快照入口；MPS3/MPS4 跳过 NPU 寄存器读取，native 复用主机时间 |
| `source/lib/mlek/fwk/{tflm,executorch}/` | 接收调用方的可选 profiler/tracer；没有引入 HAL 依赖 |
| `scripts/cmake/executorch.cmake` | QR-code 开启 CPU profiling 时启用运行时算子回调，统一宏定义 |

## 验证范围

- 四个头使用不同量化参数、非零零点和完整 INT8 值域，验证模型反量化与原有
  数值恢复相符；允许浮点乘加顺序产生的末位舍入差异。
- 真实七张 BMP 的主机 TFLite 输出，追加反量化前后四个头逐元素一致，类别相同。
- 检查 TFLite 的四个反量化输出连接与离线内存计划，拒绝无关 CPU 操作。
- 小型真实 PT 图执行捕获、校准、Vela 和 PTE 序列化，分别覆盖 INT8、FP32
  输出接口，保持输入 INT8。
- 复用 `direct_pt` 保存的真实校准图，生成 PTE 浮点输出模型；没有重跑校准。

FVP 运行记录与实际计时见本地
`resources_downloaded/qrcode/output_dequantization/`，FPGA 仍需单独实测。

### 2026-10-16 FVP 计时记录

MPS4/SSE-320，U85 1024，Dedicated SRAM，cache 393,216 字节，Release，
CPU 时钟 25 MHz；单独编入原始 `02.bmp`，每个模型运行一帧。各框架的 INT8
和 FP32 输出版本使用相同的量化网络。TFLite 与 PTE 来自各自原有的量化路线，
不是同一套量化参数，不将跨框架检测结果差异归因于反量化方式。

下表毫秒数均由实际 `CPU TOTAL / 25,000` 换算，不使用日志中整数毫秒的累加值：

| 框架/输出 | 推理 ms | 模型内反量化 cycles / ms | 解码 ms | 推理加后处理 ms |
| --- | ---: | ---: | ---: | ---: |
| TFLM / INT8 | 39.403 | 无独立模型算子 | 40.702 | 80.392 |
| TFLM / FP32 | 60.336 | 519,990 / 20.800 | 42.970 | 103.593 |
| ExecuTorch / INT8 | 41.753 | 无独立模型算子 | 40.599 | 82.608 |
| ExecuTorch / FP32 | 46.307 | 109,185 / 4.367 | 42.834 | 89.397 |

各框架的两种输出方式在该图上均检出两个结果，类别、顺序、打印的置信度一致，
框尺寸只有末位浮点舍入差异。这里只确认两种输出方式的一致性，不证明原模型
的类别预测或重复框符合真实标注。两种 FP32 输出固件都实际捕获了四次反量化
事件并成功结束，默认 INT8 版本没有额外的反量化算子计时。

对应模型、主机七图比较、编译/运行日志和测试固件保存在：

```text
resources_downloaded/qrcode/output_dequantization/
  pte_Dedicated_Sram/best.pte
  tflm_Dedicated_Sram/best_float_outputs_vela.tflite
  host_parity.json
  fvp_comparison.json
  evidence/
```
