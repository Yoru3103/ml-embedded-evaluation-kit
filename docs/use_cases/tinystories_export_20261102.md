# TinyStories-42M U85-1024 导出实验

<!-- SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

本实验只覆盖第一阶段：量化、Vela 编译与 PTE 产物检查。FVP 连续生成、
MLEK 用例接入及 FPGA 部署属于第二阶段，不能由导出成功推断已经验证。

## 配置与产物位置

- ExecuTorch 源码：`dependencies/executorch`，标签 `v1.5.1`，
  commit `3b60683923245cf472b7323426920e15623ba361`。
- 独立 WSL venv：`/home/mokuroo/.venvs/mlek-tinystories-et151`。
- 资源与日志：工程根目录下 `resources_downloaded/tinystories42m/`。
  该目录用于实验产物，不作为受版本控制的源代码发布。
- PyTorch `2.14.0+cpu`、ExecuTorch `1.5.1+cpu`、TorchAO `0.18.0+cpu`、
  Vela `5.2.0`、TOSA tools `2026.5.0`。
- 固定 batch=1、单 token 调用、64-token 上下文、FP32 KV Cache、
  `ethosu_16a8w`、`quantize_scope=linear`。
- Vela：`ethos-u85-1024`、`Ethos_U85_SYS_DRAM_Mid`、`Sram_Only`。
  这是上游配置的 FVP 基线，不能直接当作 Dedicated/Shared SRAM 板级产物。
- 官方 Wikitext 校准任务，`calibration_limit=1`，窗口长度 64；
  limit 计数的是文档，不保证仅执行 64 个 token。

模型权重来自 `karpathy/tinyllamas` 的 `stories42M.pt`；tokenizer 来自
`karpathy/llama2.c` 的 `tokenizer.model`。输入文件 SHA256、关键导出源码
SHA256 和完整 pip freeze 均保存在产物目录。

## 本机复现

在 WSL、工程根目录执行：

```bash
cp dependencies/executorch/schema/{program,scalar_type}.fbs \
  dependencies/executorch/exir/_serialize/
bash resources_downloaded/tinystories42m/run_export.sh
source /home/mokuroo/.venvs/mlek-tinystories-et151/bin/activate
python resources_downloaded/tinystories42m/enrich_report.py
python resources_downloaded/tinystories42m/verify_pte.py
```

`run_export.sh` 依赖该目录中的 `params.json`、模型文件和
`export_with_report.py`；后者优先使用本地 ExecuTorch Python 源码，
动态库使用独立 venv 中与源码版本匹配的官方 wheel。
完整依赖见产物目录 `environment.txt`，安装时 CPU wheel 索引为
`https://download.pytorch.org/whl/cpu`。

## 已处理的问题

1. `lm_eval==0.4.5` 与自动安装的 Transformers 5.x 不兼容，
   报错 `AutoModelForVision2Seq` 不存在。独立环境固定为
   Transformers `4.51.3`、PEFT `0.15.2`、Datasets `3.6.0` 后通过导入和校准。
2. Vela 已编译全部子图，但转 ExecuTorch 时缺少
   `quantized_decomposed::quantize_per_tensor.out` 和
   `dequantize_per_tensor.out`。实验包装器显式加载官方 wheel 中的
   `kernels/quantized/libquantized_ops_aot_lib.so`，没有伪造算子实现。
3. Windows 构建脚本实际已有 1.5.1 的包版本映射；部分现有文档仍写 1.3.1，
   不能仅根据旧文档判定构建入口不支持 1.5.1。
4. 直接从 checkout 运行 Python 源码时，`exir/_serialize` 缺少构建时复制的
   `program.fbs`、`scalar_type.fbs`。从同一 checkout 的 `schema/` 复制这两个
   资源后，小型 ReLU 模型的完整 PTE 序列化预检通过（888 bytes）。
   导出包装器在大模型编译前执行该预检。

## 验证边界

CPU FP32 参考已经生成，固定 prompt 为
`Once upon a time in a small village,`，16-token greedy 续写为：

> there was a little girl named Lily. She loved to play outside with her

参考 Token ID 与位置见 `fp32_reference.json`。这不是量化 PTE 的输出。
校准期间单篇 Wikitext 的 perplexity 不作为正式质量评估；
尚需量化前后 logits/Token 对比和多 prompt 生成验证。

KV Cache 逻辑容量为 `2 × 8 × 64 × 8 × 64 × 4 = 2,097,152 bytes`。
最终 PTE 中 16 个初始化 KV 张量均分配在 planned memory ID 1，
已计入静态 planned buffer，不能再次累加。
主机导出进程的 RSS 是 PC 内存，不能作为 MCU RAM 指标。

## 实测导出结果

最终完整导出正常退出，耗时 8 分 33.97 秒（包含 CPU 校准与 Vela 编译）。

| 项目 | 实测结果 |
| --- | --- |
| PTE | `tinystories42m_u85_1024_ctx64_kv_w8a16.pte` |
| 文件大小 | **105,240,640 bytes / 100.3653 MiB** |
| SHA256 | `38cf292e5d6295b75e1010335940a1b1d60dd93ba73084917bca106c7712ac24` |
| forward planned buffer | **4,584,448 bytes / 4.3721 MiB**，含 2 MiB KV |
| 最大单子图 Vela scratch | **132,128 bytes / 129.03 KiB** |
| 各子图 scratch 需求之和 | 1,245,440 bytes，仅诊断统计，不代表并发分配 |
| NPU 委托子图 | **41** |
| Linear 节点 | **57 个全部委托** |
| 分区后节点数（runtime 转换前） | 委托 660、非委托 805 |
| runtime 转换后 | 委托 660、非委托 1,409；额外含 604 个 `alloc` 节点 |
| forward 输入 | INT32 `[1,1]` Token、INT32 `[1]` position |
| forward 输出 | FP32 `[1,32000]` logits |

CPU 回退的主要算子包括 Embedding（1）、Attention BMM（16）、
Softmax（8）、mean/rsqrt（各 17）、sigmoid（8），以及部分索引、
状态更新、形状变换和量化/反量化。节点数量不是 FLOPs、耗时或加速比例。

检查已通过：`pip check`、实验 Python/Bash 脚本语法检查、完整导出、
最终 ET12 PTE 反序列化、方法输入输出核对、41 个 Vela stream 的 scratch 提取。

原始证据位于产物目录：`export_report.json`、`pte_interface.json`、
`memory_profile.json`、`lowered_graph.txt`、`export.log`。
`memory_profile.json` 是静态分配轨迹，时间轴不表示真实 MCU 执行时间。

**尚未验证**：量化后生成质量、跨调用 KV 正确性、运行峰值 RAM、
FVP/FPGA 吞吐和 CPU/NPU 实际耗时。
若模型必须整体常驻单个 64 MB 存储区，本产物无法装入；如有独立只读模型
存储及 NPU 可访问的地址映射，需要第二阶段另行验证。

## 第二阶段

先用匹配此 PTE 的上游 semihosting runner 做 FVP smoke test，
再扩展为连续 16～32 Token。检查 Token/位置输入、同一 Method 的 Cache
跨调用保持、新 prompt 的状态重置、CPU 回退内核和内存区域。
之后再新增 MLEK `tinystories` 用例及 UART Token 协议。
平台相关操作留在应用/HAL 层，`source/lib` 继续保持平台无关。

参考：[官方 TinyStories 示例](https://github.com/pytorch/executorch/blob/main/examples/arm/tinystories42m_example_ethos_u/README.md)。
