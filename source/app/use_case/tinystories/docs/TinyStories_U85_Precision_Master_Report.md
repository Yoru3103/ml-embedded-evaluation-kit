# TinyStories 42M U85 精度排查与 LUT 尝试总报告

文档日期：2026-11-13\
适用模型：TinyStories 42M，Ethos-U85 Z1024，MPS4 / Corstone-320\
默认配置：context 64，W8A16 body + INT8 embedding + INT16 persistent KV

---

## 索引

### 章节索引

1. [结论速览](#1-结论速览)
2. [目标与基线](#2-目标与基线)
3. [实验总表](#3-实验总表)
4. [诊断工具与方法](#4-诊断工具与方法)
5. [关键问题与根因](#5-关键问题与根因)
6. [逐项实验记录](#6-逐项实验记录)
7. [性能数据](#7-性能数据)
8. [当前产物索引](#8-当前产物索引)
9. [代码与脚本索引](#9-代码与脚本索引)
10. [日志索引](#10-日志索引)
11. [未解决问题](#11-未解决问题)
12. [下一步建议](#12-下一步建议)
13. [附录：命令与术语](#13-附录命令与术语)
14. [2026-11-13 阶段性更新：ctx128、FVP demo、归档](#14-2026-11-13-阶段性更新ctx128fvp-demo归档)

### 产物索引

| 类型 | 索引位置 |
|---|---|
| PTE | [8.1 PTE](#81-pte) |
| FPGA 固件 | [8.2 FPGA 固件](#82-fpga-固件) |
| TOSA / JSON / payload | [8.3 TOSA 与中间产物](#83-tosa-与中间产物) |
| 工具脚本 | [9. 代码与脚本索引](#9-代码与脚本索引) |
| FVP / FPGA 日志 | [10. 日志索引](#10-日志索引) |
| 相关文档 | [8.4 相关文档](#84-相关文档) |

### 快速结论

- 当前唯一稳定达到 **10/10 prompt、16/16 generation** 的方案：
  - **rsqrt-only CPU**，18 delegate；
  - **RMSNorm-only CPU**，18 delegate。
- 单 delegate 的最佳结果仍是 full 的 **6/10、11/16**。
- 单 delegate LUT 尝试全部失败：3/10、1/10、6/10。
- 主要精度瓶颈：**RMSNorm rsqrt TABLE 的低值区量化/查表**。
- 主要工程阻碍：**TOSA TABLE 精确 index/fraction/output scale 语义尚未确认**。
- ctx128 已修复：floor 1e-3 FVP 为 **9/10、13/16**，生成 118 token。
- 已提供 DEMO ctx64/ctx128 流式 prompt 演示，见第 14 节。

---

## 1. 结论速览

### 1.1 精度结果

| 版本 | delegate | prompt | generation | 结论 |
|---|---:|---:|---:|---|
| full 单 delegate | 1 | 6/10 | 11/16 | 基线 |
| softmax-only CPU | 9 | 7/10 | 13/16 | softmax 有贡献，不够 |
| eps=1e-3 additive | 1 | 10/10 | 5/16 | 改变数学，生成坏 |
| floor 1e-3 | 1 | 10/10 | 13/16 | 最好的单 delegate floor |
| floor 3e-4 | 1 | 7/10 | 14/16 | 生成最好，prompt 掉 |
| floor 3e-3 | 1 | 3/10 | 4/16 | 比 full 还差 |
| qspec rsqrt 固定 scale | 1 | 8/10 | 6/16 | 破坏共享 qspec |
| **RMSNorm CPU** | 18 | **10/10** | **16/16** | 精度上界，4.09 s/token |
| **rsqrt-only CPU** | 18 | **10/10** | **16/16** | 最小切分金标准 |
| linear | 多个 | 10/10 | 16/16 | 诊断上界 |
| LUT warp v1 | 1 | 3/10 | 3/16 | 输入 scale 失败 |
| LUT 校准 v2 | 1 | 1/10 | 0/16 | 输出 scale 假设错 |
| LUT per-node F v3 | 1 | 6/10 | 5/16 | 方向对，语义仍错 |

### 1.2 核心判断

1. **softmax 不是主因**：softmax-only 只到 7/10、13/16。
2. **RMSNorm 是主因**：把 RMSNorm 的 `add/rsqrt/mul` 放 CPU，直接 10/10、16/16。
3. **rsqrt 是 RMSNorm 里的关键**：只把 `rsqrt` 放 CPU，add/mul 留 NPU，仍然 10/10、16/16。
4. **floor 只能局部缓解**：floor 1e-3 修 prompt，但生成 13/16；floor 3e-3 反而更差。
5. **单 delegate LUT 还没有成功**：输入 warp、输出 scale、table index 三者尚未同时对齐。

---

## 2. 目标与基线

### 2.1 目标

在保持 **1 个 Ethos-U delegate、0 个 CPU op** 的前提下，恢复：

```text
prompt 10/10
generation 16/16
```

### 2.2 基线

Full NPU 单 delegate：

```text
prompt 6/10
generation 11/16
```

Linear（全部非线性 CPU）：

```text
prompt 10/10
generation 16/16
```

### 2.3 当前精度金标准

```text
rsqrt-only CPU
18 delegate
prompt 10/10
generation 16/16
```

该版本用于判断任何单 delegate 方案是否“保住了精度”。

---

## 3. 实验总表

| # | 实验 | 图改动 | delegate | prompt | gen | 备注 |
|---:|---|---|---:|---:|---:|---|
| 1 | full | 无 | 1 | 6/10 | 11/16 | 基线 |
| 2 | softmax-only | 去 softmax Q/DQ | 9 | 7/10 | 13/16 | 需 terminal KV |
| 3 | eps=1e-3 | `norm_eps=1e-3` | 1 | 10/10 | 5/16 | 数学被改坏 |
| 4 | floor 1e-3 | `relu(mean-floor)+floor` | 1 | 10/10 | 13/16 | 可用中间版 |
| 5 | floor 3e-4 | 同上 | 1 | 7/10 | 14/16 | 生成最好 |
| 6 | floor 3e-3 | 同上 | 1 | 3/10 | 4/16 | 更差 |
| 7 | floor 1e-2 | 同上 | — | — | — | TOSA 转换失败 |
| 8 | qspec | 固定 rsqrt scale | 1 | 8/10 | 6/16 | 共享 qspec 被破坏 |
| 9 | RMSNorm CPU | add/rsqrt/mul CPU | 18 | 10/10 | 16/16 | 4.09 s/token |
| 10 | rsqrt-only CPU | 只 rsqrt CPU | 18 | 10/10 | 16/16 | 最小切分 |
| 11 | LUT warp v1 | 输入 S + 新表 | 1 | 3/10 | 3/16 | 输出饱和 |
| 12 | LUT cal v2 | p99 s_tab | 1 | 1/10 | 0/16 | 输出 scale 假设错 |
| 13 | LUT F v3 | per-node F | 1 | 6/10 | 5/16 | 仍未对齐 |

---

## 4. 诊断工具与方法

### 4.1 FVP UART

问题：`uart0.out_file` 每次正好 4096 字节，卡住后日志不落盘。

解决：使用 telnet raw 端口：

```text
-C mps4_board.telnetterminal0.mode=raw
-C mps4_board.telnetterminal0.start_port=5020
```

再配合 `build/tinystories42m/fvp_raw_client.py` 抓完整 UART。

### 4.2 Delegate 日志

临时加入：

```text
TINYSTORIES_DELEGATE begin/end
TINYSTORIES_RUN begin
TINYSTORIES_INFER begin/inputs set/run begin/run end
```

用于区分“没进 RunInference”和“卡在 delegate”。

### 4.3 Delegate I/O 校验

`source/app/use_case/tinystories/tools/validate_delegate_io.py`

检查：

- ET args 与 Vela inputs/outputs 数量；
- ET/Vela 输入 shape + dtype 对齐；
- 16 个 KV 输入/输出；
- in-place alias。

### 4.4 TOSA/Vela 检查

流程：

```text
TOSA dump -> flatc JSON -> 修改 -> flatc 回二进制
  -> Vela compile_payload.py -> repack_pte.py
```

注意：

- JSON -> 二进制必须用简单 `flatc -b -o . tosa.fbs file.json`；
- 不要加 `--strict-json`、`--raw-binary`，否则会输出 JSON 文本或崩溃；
- terminal KV 由 `terminalize_tosa_outputs.py` / `terminalize_delegate.py` 完成。

### 4.5 host 参考

- `quantized.pt2` 在 WSL 里加载；
- 需要先 import：
  - `executorch.exir.passes._quant_patterns_and_replacements`
  - `torchao.quantization.pt2e.quantize_pt2e`
- 否则会报 `quantized_decomposed.embedding_byte` 无法反序列化。

---

## 5. 关键问题与根因

### 5.1 多 delegate 的 KV 输出被折叠

现象：

```text
Expected 16 KV inputs and 16 KV outputs, found 16 and 0
```

根因：Vela 把完整 KV 输出折叠成 in-place alias。

解决：为每个完整 KV 输出加 terminal transpose，强制独立输出。

### 5.2 ET/Vela 输入顺序错位

现象：

```text
Input 9 expected Integer (4 byte), Char (1 byte) or Bool (1 byte)
got ScalarType id Int size 2
```

根因：delegate 4 的输入 9/10 顺序错位：

```text
ET 9  = (1,) INT32
ET 10 = (1,8,1,64) INT16
Vela 9 = (1,8,1,64) INT16
Vela 10= (1,) INT32
```

解决：`reorder_from_pte.py` 按 shape + dtype 重排。

### 5.3 RMSNorm CPU 第一版的 INT32 边界

现象：

```text
TOSA output [1,1,1,1] INT32
ET output   [1,1,1,1] FP32
```

根因：把 `mean/sum` 也放到了 CPU，暴露 INT32 `reduce_sum`。

解决：保留 `mean/sum` 在 NPU，只把 `add/rsqrt/mul` 放 CPU。

### 5.4 epsilon 被量化掉

RMSNorm 统计量 scale 1/4096：

```text
1e-5 / (1/4096) = 0.041 code
```

量化后 epsilon 基本为 0。

直接改 `norm_eps=1e-3`：

```text
prompt 10/10
gen 5/16
```

说明不是“补回 epsilon”，而是改变了模型数学。

### 5.5 单 int16 scale 的范围/分辨率互斥

- 保留 eps 1 code：scale≈1e-5，mean 上限 0.327，会饱和；
- 保留 mean ±8：scale=1/4096，eps 只有 0.041 code。

所以 `mean + eps` 不能靠单个 int16 scale 同时解决。

### 5.6 floor 非单调

| floor | prompt | gen |
|---|---:|---:|
| 3e-3 | 3/10 | 4/16 |
| 1e-3 | 10/10 | 13/16 |
| 3e-4 | 7/10 | 14/16 |

floor 会改变 `mean < floor` 的 token，太大/太小都不行。

### 5.7 TOSA TABLE 的真实行为

对真实 RMSNorm 输入观察：

```text
table_index = clamp(round(mean(x²) * 2) + 256, 0, 512)
table_output = table[table_index] / F
```

`F` 是 per-node 输出 factor，不是全局 2266：

```text
F 范围约 886 到 16,383
```

低值区：

```text
mean=0.193 -> index 256 -> table=32767 -> 输出 14.46
exact rsqrt=2.2756
```

说明低值区被 index 256 的 32767 饱和值截断，`mean < 0.25` 全部映射到同一个 code。

### 5.8 LUT 三次失败

| 版本 | 做法 | prompt | gen | 失败原因 |
|---|---|---:|---:|---|
| v1 | 输入 S，新表 | 3/10 | 3/16 | 输出 32767 饱和未处理 |
| v2 | p99 s_tab | 1/10 | 0/16 | 全局 2266 输出 scale 假设错 |
| v3 | per-node F | 6/10 | 5/16 | table index/输入语义仍未完全对齐 |

### 5.9 TOSA reference 未跑通

最小图：

```text
test_in INT16 -> TABLE(rsqrt) -> RESCALE -> output
```

`tosa_reference_model` 在 WSL 存在，但：

- 不带 `--operator_fbs` 报 schema 缺失；
- 带 schema 后不产生 `min_out.npy`，没有有效错误信息。

因此精确 TABLE index/fraction/output scale 语义还没有参考输出。

---

## 6. 逐项实验记录

### 6.1 softmax-only

- 去掉 softmax Q/DQ；
- 9 delegate；
- terminal KV + delegate 4 输入重排；
- FVP：7/10、13/16。

首次分叉：generated 13，`1708 vs 5377`。

### 6.2 eps=1e-3

- 全量单 delegate；
- terminal KV；
- FVP：10/10、5/16；
- host 也在 generated 1 分叉（471 -> 10600）。

### 6.3 floor 扫描

1e-3：

```text
10/10、13/16
text: ... loved to play outside and explore
```

3e-4：

```text
7/10、14/16
```

3e-3：

```text
3/10、4/16
```

1e-2：

TOSA 转换 >30 min，失败。

### 6.4 qspec

- FixedQParamsObserver；
- rsqrt input 1/8192、output 1/1024；
- FVP：8/10、6/16；
- 破坏共享 qspec 平衡。

### 6.5 RMSNorm CPU

- 18 delegate；
- FVP：10/10、16/16；
- FPGA TA：4.09 s/token，NPU 322 ms，CPU Quantize 2.15 s。

### 6.6 rsqrt-only CPU

- partitioner 拒绝 `aten.rsqrt`；
- add/mul 留 NPU；
- 18 delegate；
- FVP：10/10、16/16。

生成文本：

```text
Once upon a time in a small village, there was a little girl named Lily.
She loved to play outside with her friends. One day, while playing,
she saw a big, red ball.
```

### 6.7 LUT 尝试

- identity clone：17 个独立 table constant，单 Vela network，665 NPU op，100% NPU；
- v1/v2/v3 均未达到 10/10、16/16。

---

## 7. 性能数据

### 7.1 RMSNorm CPU，FPGA TA ON

日志：

```text
build/logs/tinyStory_logs/TA_split/tinystory_test_TA_split_64tok_dedicated_1_2M.log
```

25 次前向：

```text
prompt 40.936 s
decode 61.413 s
合计 102.349 s
```

单 token：

| 项目 | 耗时 |
|---|---:|
| E2E | 4,093.945 ms |
| NPU hardware | 322.20 ms（7.9%） |
| invoke | 4,046.96 ms |
| CPU.Quantize | 2,147.81 ms |
| CPU.OtherKernel | 1,141.46 ms |
| CPU.Dequantize | 209.64 ms |
| Delegate.Total | 413.54 ms |
| NPU.Driver | 332.97 ms |

对比 full TA ON：

```text
451.947 ms/token
```

RMSNorm CPU 慢约 9.1 倍。

### 7.2 floor 1e-3，54-token TA

日志：

```text
build/logs/tinyStory_logs/TA/tinystory_test_TA_1e3floor_64tok_dedicated_1_2M.log
```

- E2E 约 2,777 ms/token；
- NPU 约 317 ms；
- CPU.Quantize 约 2,125 ms。

### 7.3 DEMO TA OFF

日志：

```text
build/logs/tinyStory_logs/TA/tinystory_demo_TA_1e3floor_64tok_dedicated_1_2M.log
```

- E2E 约 3,642 ms/token；
- NPU 约 1,190 ms；
- CPU.Quantize 约 2,123 ms。

差异原因：TA OFF 时 NPU cycle 约 3.75 倍；CPU 相同。

---

## 8. 当前产物索引

### 8.1 PTE

| 版本 | 路径 |
|---|---|
| full 单 delegate | `resources_downloaded/tinystories42m/npu_full_16a8w_embedding8_dedicated_protected/` |
| softmax-only | `resources_downloaded/tinystories42m/npu_softmax_cpu_dump_ctx64/tinystories42m_u85_1024_ctx64_softmax_cpu_terminal3.pte` |
| eps=1e-3 | `resources_downloaded/tinystories42m/npu_eps1e3_dump_ctx64/tinystories42m_u85_1024_ctx64_eps1e3_terminal.pte` |
| floor 1e-3 | `resources_downloaded/tinystories42m/npu_relu1e3_ctx64/tinystories42m_u85_1024_ctx64_relu1e3_terminal.pte` |
| floor 3e-4 | `resources_downloaded/tinystories42m/npu_relu3e4_ctx64/tinystories42m_u85_1024_ctx64_relu3e4_terminal.pte` |
| floor 3e-3 | `resources_downloaded/tinystories42m/npu_relu3e3_ctx64/tinystories42m_u85_1024_ctx64_relu3e3_terminal.pte` |
| qspec | `resources_downloaded/tinystories42m/npu_qspec_ctx64/tinystories42m_u85_1024_ctx64_qspec_terminal.pte` |
| RMSNorm CPU | `resources_downloaded/tinystories42m/npu_rmsnorm_cpu2_ctx64/tinystories42m_u85_1024_ctx64_rmsnorm_cpu2_terminal.pte` |
| rsqrt-only CPU | `resources_downloaded/tinystories42m/npu_rsqrt_only_ctx64/tinystories42m_u85_1024_ctx64_rsqrt_only_terminal.pte` |
| LUT warp v1 | `resources_downloaded/tinystories42m/npu_lut_probe_ctx64/tinystories42m_u85_1024_ctx64_warp.pte` |
| LUT cal v2 | `resources_downloaded/tinystories42m/npu_lut_probe_ctx64/tinystories42m_u85_1024_ctx64_cal2.pte` |
| LUT F v3 | `resources_downloaded/tinystories42m/npu_lut_probe_ctx64/tinystories42m_u85_1024_ctx64_fix.pte` |

### 8.2 FPGA 固件

| 版本 | 路径 |
|---|---|
| RMSNorm CPU，TEST，TA ON | `build/tinystories42m/test/fpga-rmsnorm-cpu2-ta/tinystories_test_ctx64_rmsnorm_cpu_fpga_ta.zip` |
| floor 1e-3，TEST 54 token，TA ON | `build/tinystories42m/test/fpga-relu1e3-54-ta/tinystories_test_ctx64_relu1e3_54tok_ta_fpga.zip` |
| floor 1e-3，TEST 54 token，TA OFF | `build/tinystories42m/test/fpga-relu1e3-54/tinystories_test_ctx64_relu1e3_54tok_fpga.zip` |

### 8.3 TOSA 与中间产物

| 内容 | 路径 |
|---|---|
| 干净 full TOSA | `resources_downloaded/tinystories42m/npu_lut_probe_ctx64/delegate_00.tosa` |
| 干净 full JSON | `resources_downloaded/tinystories42m/npu_lut_probe_ctx64/delegate_00.json` |
| 17 table identity clone | `resources_downloaded/tinystories42m/npu_lut_probe_ctx64/delegate_00_split.tosa` |
| LUT v1 | `delegate_00_warp.tosa` |
| LUT v2 | `delegate_00_cal.tosa` |
| LUT v3 | `delegate_00_fix.tosa` |
| 最小 rsqrt 图 | `min_rsqrt.json`、`min_in.npy` |
| 校准配置 | `cal_configs.json` |

### 8.4 相关文档

| 文档 | 说明 |
|---|---|
| `source/app/use_case/tinystories/docs/TinyStories_Precision_Bisect_Report.md` | 详细 bisect 记录 |
| `source/app/use_case/tinystories/docs/TinyStories_Rsqrt_LUT_Plan.md` | LUT 单 delegate 方案 |
| `source/app/use_case/minimind2_small/docs/261111-MiniMind2_U85_排查与部署报告.md` | MiniMind 参考，13.7/13.9/13.10 |

---

## 9. 代码与脚本索引

### 9.1 图变换

| 脚本 | 用途 |
|---|---|
| `source/app/use_case/tinystories/tools/softmax_cpu_transform.py` | 去 softmax Q/DQ |
| `source/app/use_case/tinystories/tools/rmsnorm_cpu_transform.py` | RMSNorm CPU 切分 |
| `source/app/use_case/tinystories/tools/terminalize_delegate.py` | 多 delegate terminal KV |
| `source/app/use_case/tinystories/tools/reorder_delegate_inputs.py` | 硬编码输入重排 |
| `source/app/use_case/tinystories/tools/reorder_from_pte.py` | 按 PTE args 通用重排 |
| `source/app/use_case/tinystories/tools/repack_pte_multi.py` | 多 delegate 重打包 |

### 9.2 导出与校验

| 脚本 | 用途 |
|---|---|
| `resources_downloaded/tinystories42m/export_with_protected_embedding.py` | 导出入口，支持 env 开关 |
| `source/app/use_case/tinystories/tools/validate_delegate_io.py` | delegate I/O 校验 |
| `source/app/use_case/tinystories/tools/profile_contract.py` | profile 阶段校验 |
| `build/tinystories42m/fvp_raw_client.py` | telnet raw UART 抓取 |

### 9.3 构建

| 脚本 | 用途 |
|---|---|
| `source/app/use_case/tinystories/tools/build_test.ps1` | TEST 固件 |
| `source/app/use_case/tinystories/tools/build_fpga_demo.ps1` | DEMO/TEST 固件 |

### 9.4 导出环境变量

| 变量 | 作用 |
|---|---|
| `TINYSTORIES_DUMP_TOSA=1` | dump `delegate_XX.tosa` |
| `TINYSTORIES_CONTEXT=64` | context |
| `TINYSTORIES_SOFTMAX_CPU=1` | softmax CPU |
| `TINYSTORIES_RMSNORM_CPU=1` | RMSNorm CPU |
| `TINYSTORIES_RSQRT_ONLY_CPU=1` | 只 rsqrt CPU |
| `TINYSTORIES_RSQRT_QSPEC=1` | rsqrt 固定 qspec |
| `TINYSTORIES_RMSNORM_CLAMP=...` | floor 形式 |
| `TINYSTORIES_NEG_INF=...` | mask |

---

## 10. 日志索引

| 日志 | 内容 |
|---|---|
| `build/logs/tinyStory_logs/TA_split/tinystory_test_TA_split_64tok_dedicated_1_2M.log` | RMSNorm CPU FPGA TA，54 token |
| `build/logs/tinyStory_logs/TA/tinystory_test_TA_1e3floor_64tok_dedicated_1_2M.log` | floor 1e-3 FPGA TA |
| `build/logs/tinyStory_logs/TA/tinystory_demo_TA_1e3floor_64tok_dedicated_1_2M.log` | DEMO TA OFF |
| `build/tinystories42m/test-fvp-warp-raw.log` | LUT v1 |
| `build/tinystories42m/test-fvp-cal2-raw.log` | LUT v2 |
| `build/tinystories42m/test-fvp-fix-raw.log` | LUT v3 |
| `build/tinystories42m/test-fvp-rsqrt-only-raw.log` | rsqrt-only CPU |

---

## 11. 未解决问题

### 11.1 TOSA TABLE 精确语义

仍未确认：

- table index 是 `code + 256`、`(code >> 7) + 256` 还是其他；
- 低 7 位 fraction 是否参与插值；
- table 输出 rescale 与下游 dequant 的精确关系；
- 为什么 per-node `F` 会从 886 到 16,383 变化。

`tosa_reference_model` 最小图没有跑通，是当前最大阻碍。

### 11.2 低值区饱和

table index 256 的 32767 是饱和哨兵：

```text
mean < 0.25 -> 同一个 code
```

单靠输入 warp 无法恢复，需要同时改 table 输出 scale 或表值。

### 11.3 单 delegate LUT 未达标

当前最好 6/10、5/16，仍会生成重复 token。

### 11.4 性能

RMSNorm CPU 精度达标但 4.09 s/token；rsqrt-only CPU 尚未测 FPGA 性能，预计仍受 CPU Q/DQ 影响。

---

## 12. 下一步建议

### 12.1 首选：跑通 TOSA reference 最小图

1. 用 `tosa_verif_run_ref` 或 TOSA conformance 的 TABLE 用例；
2. 输入已知 code，拿到参考输出；
3. 确认 index/fraction/output rescale；
4. 再重写 17 个 table。

### 12.2 次选：放弃单 delegate LUT

保留 **rsqrt-only CPU**：

```text
10/10、16/16
18 delegate
```

优化 CPU Q/DQ 开销，作为工程交付。

### 12.3 长期

- QAT；
- 或重新设计 RMSNorm 的量化表示，避免 epsilon 与 mean 共用 int16 scale。

---

## 13. 附录：命令与术语

### 13.1 常用命令

抓 FVP UART：

```bash
python build/tinystories42m/fvp_raw_client.py 5020 out.log 20
```

Vela 编译：

```bash
python resources_downloaded/tinystories42m/_tosa_diag/compile_payload.py \
  --tosa delegate_00.tosa --out payload \
  --config resources_downloaded/tinystories42m/_tosa_diag/vela52_cache_sizes.ini
```

PTE 重打包：

```bash
python source/app/use_case/tinystories/tools/repack_pte_multi.py \
  source.pte payload_dir output.pte
```

### 13.2 术语

| 术语 | 含义 |
|---|---|
| delegate | Ethos-U 子图 |
| KV | key/value cache |
| terminal KV | 为 KV 输出加 transpose，避免 alias |
| floor | `max(mean, floor)` 替代 `mean+eps` |
| qspec | quantization specification |
| LUT | 513 点查找表 |
| s_tab | table 输出 code 的物理 scale |
| F | table 值与原 rsqrt 的比例因子 |
| TA | timing adapter |

---

## 14. 2026-11-13 阶段性更新：ctx128、FVP demo、归档

### 14.1 ctx128 修复

ctx128 之前失败的原因不是 terminal transpose 本身，而是导出配置仍是 64：

```text
max_seq_length: 64
max_context_length: 64
calibration_seq_length: 64
```

所以 PTE 的 KV cache 实际是 `[1,8,64,64]`，与 context 128 的 terminal 输出不匹配。

修复：

```text
max_seq_length: 128
max_context_length: 128
calibration_seq_length: 128
```

重新导出后：

```text
delegate_00.tosa                         43,652,032 B
tinystories42m_..._ctx128_relu1e3.pte   58,299,360 B
```

完整 KV 输出使 PTE 超出 64 MiB 静态预算，因此又用：

```text
retain_int16_kv.py --cache-shape 1,8,128,64 --compact-arena
```

压缩为：

```text
tinystories42m_u85_1024_ctx128_relu1e3_terminal_int16.pte
56,201,856 B
```

当前 DEMO 构建使用 7 MiB method pool。CMake 报告的静态 DDR 占用为：

```text
66,832,723 / 67,108,864 B
```

### 14.2 FVP 结果

| 版本 | context | 生成 token | prompt | generation |
|---|---:|---:|---:|---:|
| floor 1e-3 ctx64 | 64 | 54 | 10/10 | 13/16 |
| floor 1e-3 ctx128 | 128 | 118 | 9/10 | 13/16 |

ctx64 文本：

```text
Once upon a time in a small village, there was a little girl named Lily.
She loved to play with her friends and explore the woods near her house.
One day, while playing, she found a shiny crystal on the ground.
It was so pretty and sparkly that she wanted to keep it
```

ctx128 文本：

```text
Once upon a time in a small village, there was a little girl named Lily.
She loved to play with her friends and eat yummy food. One day, Lily's mom
made a big pot of soup for dinner. The soup was very hot, so Lily had to wait
for it to cool down before she could eat it.
```

### 14.3 FVP demo 产物

```text
build/tinystories42m/fvp-demo/
  README.md
  ctx64/bin/mlek_tinystories.axf
  ctx128-int16/bin/mlek_tinystories.axf
  demo-ctx64/bin/mlek_tinystories.axf
  demo-ctx128/bin/mlek_tinystories.axf
  tinystories_floor1e3_fvp_demos.zip
```

- `ctx64` / `ctx128-int16` 是 TEST 回归固件（固定 54/118 token）。
- `demo-ctx64` / `demo-ctx128` 是 DEMO 固件（可输入 prompt、流式输出）。

DEMO 使用原有输入输出接口，但 floor PTE 没有 `aten::clamp` cache fast path，
因此 `demo_cache_contract.py` 增加了 fallback：

- 扫描所有 INT16 `[1,8,context,64]` 且 `memory_id=1` 的 allocation；
- 重置这些 KV offset（ctx64 为 32 个 alias，ctx128 为 16 个 cache）。

### 14.4 流式 console

修改：

```text
source/app/use_case/tinystories/tools/fvp_console.py
```

现在会过滤：

- `INFO -` / `WARN -` / `ERROR -` / `E [` / `I [`；
- `TINYSTORIES_*`；
- 启动行 `Processor internal clock`、`MPS4 build` 等；
- `Prompt> `、`Story:`；
- 用户 prompt 的回显行。

终端流式显示故事文本，并在故事结束后显示一条 `DONE` 摘要。摘要中的
`total_ms` 取自客户端主机时钟，标记为 `time_source=host_wall`。

### 14.5 运行方式（Windows，无 WSL）

终端 1：

```powershell
.\source\app\use_case\tinystories\tools\run_fvp_demo_floor1e3.ps1 -Context 64
```

终端 2：

```powershell
.\source\app\use_case\tinystories\tools\connect_fvp_demo_floor1e3.ps1 `
  -TcpPort <FVP 打印的 telnetterminal0 端口> `
  -Prompt "Once upon a time"
```

ctx128 把 `-Context 64` 改成 `128`。

脚本：

```text
source/app/use_case/tinystories/tools/run_fvp_demo_floor1e3.ps1
source/app/use_case/tinystories/tools/connect_fvp_demo_floor1e3.ps1
```

它们只是调用原有：

```text
run_fvp_demo.ps1
connect_fvp_demo.ps1
```

### 14.6 工作区归档

失败/非关键模型：

```text
resources_downloaded/tinystories42m/_archive/2026-11-13/failed_models/
```

索引：

```text
resources_downloaded/tinystories42m/_archive/2026-11-13/README.md
resources_downloaded/tinystories42m/KEY_ARTIFACTS.md
```

保留的关键模型：

```text
npu_relu1e3_ctx64/
npu_relu1e3_ctx128/
npu_rsqrt_only_ctx64/
npu_rmsnorm_cpu2_ctx64/
npu_full_16a8w_embedding8_dedicated_protected/
```

失败的 ctx128-raw FVP 构建：

```text
build/tinystories42m/archive/fvp-ctx128-raw/
```

没有执行 git 操作，也没有删除关键产物。

### 14.7 LUT 状态

- identity clone：17 个独立 table constant 编译成 1 个 Vela network、665 NPU op、100% NPU；
- warp v1：3/10、3/16；
- 校准 v2：1/10、0/16；
- per-node F v3：6/10、5/16；
- TOSA reference 最小图仍未跑通，TABLE 的精确 index/fraction/output scale 未确认。

结论不变：单 delegate LUT 还没有达到 10/10、16/16；`rsqrt-only CPU` 仍是精度金标准。

### 14.8 当前建议

1. 精度交付：使用 `rsqrt-only CPU`（10/10、16/16，18 delegate）。
2. 演示交付：使用 floor 1e-3 `demo-ctx64` / `demo-ctx128`，可输入 prompt、流式故事输出。
3. 性能对比：统一使用 TA ON；TA OFF 的 NPU cycle 会差约 3.75 倍。
4. LUT 单 delegate：先解决 TOSA TABLE reference 语义，再继续 per-node table。

---

文档结束。
