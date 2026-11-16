# TinyStories 42M U85 精度回归排查报告

日期：2026-11-12\
范围：TinyStories 42M，Ethos-U85 Z1024，MPS4 / Corstone-320，context 64，W8A16 body + INT8 embedding + INT16 persistent KV。

## 1. 背景与目标

Full NPU 单 delegate 版本在 FVP 上只能达到：

```text
prompt top1: 6/10
generated:   11/16
```

而把全部非线性放回 CPU 的 linear 版本可以达到：

```text
prompt top1: 10/10
generated:   16/16（前 12 个已逐 token 确认）
```

目标是定位 full 版本精度损失来自哪个算子，并尽量把该算子放回 NPU delegate，而不是长期保留 CPU 小岛。

## 2. 结论摘要

| 版本 | prompt | gen | 说明 |
|---|---:|---:|---|
| full NPU | 6/10 | 11/16 | 基线 |
| softmax-only CPU | 7/10 | 13/16 | softmax 有贡献，但不够 |
| eps=1e-3 additive | 10/10 | 5/16 | 改变 RMSNorm 数学，生成被改坏 |
| relu floor 1e-3 | 10/10 | 13/16 | 解决 prompt，生成仍差 3 个 token |
| relu floor 1e-2 | — | — | TOSA 转换 >30 min，失败 |
| relu floor 3e-3 | 3/10 | 4/16 | 比 full 更差 |
| relu floor 3e-4 | 7/10 | 14/16 | floor 里生成最好，但 prompt 低于 1e-3 |
| FixedQParamsObserver | 8/10 | 6/16 | 固定 rsqrt 输入/输出 scale，反而更差 |
| **rsqrt-only CPU** | **10/10** | **16/16** | add/mul 回 NPU，仅 rsqrt CPU；18 delegate |
| **RMSNorm-only CPU** | **10/10** | **16/16** | 精度上界；4.09 s/token |
| linear | 10/10 | 16/16 | 诊断上界 |

核心结论：

- **RMSNorm 是 full 量化的主要精度瓶颈**，不是 softmax；
- 只把 RMSNorm 的 `add(eps) / rsqrt / mul` 放 CPU，就能恢复 10/10、16/16；
- 但 RMSNorm CPU 小岛把单 token 从 451.9 ms 拉到 4,093.9 ms，慢 9.1 倍；
- 单 delegate 的 `max(mean, eps_floor)` 能把 prompt 拉回 10/10，但生成只到 13/16；
- floor 形式在 TOSA lowering 中会产生大量 RESCALE，1e-2 超过 30 分钟无法完成转换。

## 3. 问题现象与定位过程

### 3.1 最初以为是 softmax exp LUT

Full 图的 softmax 被 lower 成：

```text
amax -> sub -> exp(TABLE) -> sum -> reciprocal -> mul
```

exp 使用 513 项 int16 LUT，128 codes/step，scale 0.0081，步长约 1.04。最初假设 LUT 插值误差是主要问题。

host 侧做了两组对比：

- host exact；
- host + 同样的 exp LUT。

结果两组都是 10/10、16/16，最大 logit 差异 0.807。板端却是 6/10、11/16，position=0 的 logit 差 -1.182，position=5 差 -6.955。

**结论：exp LUT 不是板端差距的主因。**

### 3.2 softmax-only 拆分与 FVP 卡死

把 softmax 的 Q/DQ 去掉，形成 9 个 delegate，导出后 FVP 首帧卡住。定位过程：

1. 在 `TinyStoriesProfiler::DelegateBegin/End` 加日志；
2. 在 `EtModel::RunInference` 加日志；
3. 在 `TinyStoriesRuntime::Infer` 加日志；
4. 发现 UART 文件每次正好 4096 字节，后续日志被 UART retarget 缓冲截断；
5. 改用 FVP telnet raw 端口抓完整 UART，才看到真实错误：

```text
Input 9 expected Integer (4 byte), Char (1 byte) or Bool (1 byte) integer inputs,
got ScalarType id Int size 2
CALL_DELEGATE execute failed at instruction 96
```

根因是 **delegate 4 的输入 9/10 顺序错位**：

```text
ET input 9  = (1,) INT32        attn_options_input_pos
ET input 10 = (1,8,1,64) INT16
TOSA/Vela 9 = (1,8,1,64) INT16
TOSA/Vela 10= (1,) INT32
```

重排 delegate 4 的 9/10 后，FVP 首帧正常，softmax-only 得到：

```text
prompt 7/10
gen    13/16
```

首次分叉在 generated 13：`1708 vs 5377`。

### 3.3 多 delegate 的 KV 输出被折叠

拆分后 Vela 把 16 个 KV 输出折叠成 in-place alias，导致：

- ET args 与 Vela inputs/outputs 数量不一致；
- validator 报 `Expected 16 KV inputs and 16 KV outputs, found 16 and 0`；
- FVP 运行时 stage 校验失败。

解决方法是 terminal 化：对每个完整 KV 输出追加一对 `TRANSPOSE([0,1,3,2])`，强制 Vela 生成独立输出。

softmax-only 的 9 个 delegate 中 8 个 KV 输出/alias 处理完后，validator 通过。

### 3.4 RMSNorm-only CPU 的 INT32 边界

第一版 RMSNorm 图手术把 `mean/sum` 也放到了 CPU，导致 delegate 输出暴露：

```text
TOSA output: [1,1,1,1] INT32  tosa_reduce_sum_default
ET output:   [1,1,1,1] FP32
```

不是顺序问题，而是 **边界 dtype 不匹配**。改为方案 1：

- 保留 `mean/sum` 在 NPU；
- 只把 `mul(mean const) / add(eps) / rsqrt / mul(embedding, rsqrt)` 放 CPU；
- 边界是 sum 后的 Q/DQ（int16）。

这样 18 个 delegate 的输入/输出重排全部通过，FVP 得到 10/10、16/16。

## 4. 为什么 RMSNorm 是主因

### 4.1 epsilon 被量化掉

RMSNorm 的统计量 scale 是 1/4096：

```text
mean + 1e-5
```

`1e-5 / (1/4096) = 0.041` code，量化后基本为 0。也就是说 NPU 实际算的是：

```text
rsqrt(mean)
```

而不是 `rsqrt(mean + eps)`。

### 4.2 直接把 eps 改成 1e-3 会改坏模型

尝试 `norm_eps = 1e-3` 全量 PTE：

```text
prompt 10/10
gen     5/16
```

host 侧同样分叉：

```text
eps=1e-3 gen: 727, 10600, 263, ...
reference gen: 727, 471, 263, ...
```

说明 `1e-3` 已经改变了 RMSNorm 的数学行为，不是“补回被量化掉的 epsilon”。

### 4.3 mean 的范围与 eps 分辨率互斥

单个 int16 tensor 的 scale 为 `s`：

- 要保留 `eps=1e-5` 至少 1 code：`s ≈ 1e-5`，mean 上限只有 0.327，正常 mean≈1 会饱和；
- 要保留 mean 范围 ±8：`s = 1/4096`，eps 只有 0.041 code。

因此 **单个 int16 scale 无法同时保留 mean 动态范围和 eps 分辨率**。这不是 API 问题，是信息论限制。

### 4.4 rsqrt LUT 本身不是主因

- rsqrt 输入 scale 1/4096；
- 表步长约 0.031；
- 输出 scale 0.001765；
- LUT 插值误差很小。

主要误差来自输入 mean 被量化、eps 被抹掉，而不是 rsqrt 表分辨率。

## 5. 尝试过的“放回 delegate”方案

### 5.1 additive eps = 1e-3

结果 10/10、5/16。失败，原因见 4.2。

### 5.2 max(mean, eps_floor) = 1e-3

公式：

```python
u = torch.relu(mean - floor) + floor
x * torch.rsqrt(u)
```

`torch.clamp` 和 `torch.maximum` 都在 TOSA 转换卡住；`relu` 形式可以完成导出。

结果：

```text
prompt 10/10
gen    13/16
```

故事：

```text
Once upon a time in a small village, there was a little girl named Lily.
She loved to play with her friends
```

比 full 好，但比 RMSNorm CPU 少 3 个 token。

### 5.3 floor = 1e-2

TOSA 转换日志：

```text
FuseConsecutiveRescalesPass: removed 16 identity pairs (20 RESCALES: 819 -> 799)
Converting ExportedProgram to TOSA
```

超过 30 分钟没有生成 `delegate_00.tosa` 或 `.pte`，失败。

### 5.4 floor = 3e-3

3e-3 在 1/4096 scale 下约 12.3 code，不会像 3e-4/1e-4 那样有 floor 被量化成 0 的风险。

但实测同样卡在 TOSA 转换：

```text
Converting ExportedProgram to TOSA
```

等待约 12 分钟后，3e-3 最终完成导出（`delegate_00.tosa` 43,650,396 B，PTE 56,182,960 B），说明 3e-3 只是转换很慢，不是完全不可完成。

但 FVP 结果很差：

```text
prompt 3/10
gen    4/16
```

故事：

```text
Once upon a time in a small village, there was a little boy. He was very curious and he wanted to know what
```

floor 3e-3 比 full 6/10、11/16 还差，不能使用。

### 5.4b floor = 3e-4

3e-4 在 1/4096 scale 下约 1.23 code，接近量化下限，但导出仍能完成（约 20 分钟）。

FVP：

```text
prompt 7/10
gen    14/16
```

故事：

```text
Once upon a time in a small village, there was a little girl named Lily.
She loved to play outside and explore
```

这是 floor 系列里生成最好的值（14/16），但 prompt 从 1e-3 的 10/10 降到 7/10，仍不是 16/16。

floor 结果不是单调的：

| floor | prompt | gen |
|---|---:|---:|
| 3e-3 | 3/10 | 4/16 |
| 1e-3 | 10/10 | 13/16 |
| 3e-4 | 7/10 | 14/16 |

1e-3 偏向 prompt，3e-4 偏向生成，两者都不能同时到 10/10 和 16/16。

### 5.5 local qspec / FixedQParamsObserver

Arm quantizer 支持：

```text
set_node_name / set_node_target / set_module_name
```

但 `get_symmetric_a16w8_quantization_config` 只接受 `act_range`，不能直接固定物理 scale；scale 仍由 observer 决定。

FixedQParamsObserver 可以固定 scale，但仍受 4.3 的范围/分辨率互斥限制；它更可能改善 rsqrt 输出/乘积，而不是彻底解决 epsilon。

本轮已经把 FixedQParamsObserver 跑通。当前环境是 torch 2.14.0+cpu，实际导入路径是：

```python
from executorch.backends.arm.quantizer.quantization_config import QuantizationConfig
from torchao.quantization.pt2e.quantizer import QuantizationSpec
from torchao.quantization.pt2e.observer import FixedQParamsObserver
```

不能用 `torch.ao.quantization.quantizer`（不存在），也不能用 `torch.ao.quantization.observer.FixedQParamsObserver`（Dynamo 会报 `Attempted to inline function marked as skipped`）。

实际配置：

```text
rsqrt input  scale = 1/8192
rsqrt output scale = 1/1024
固定到所有 aten.rsqrt.default
```

导出成功：

```text
resources_downloaded/tinystories42m/npu_qspec_ctx64/
  tinystories42m_u85_1024_ctx64_qspec_terminal.pte
```

FVP：

```text
prompt 8/10
gen    6/16
```

比 full 6/10、11/16 还差，说明单纯固定 rsqrt 输入/输出 scale 不能解决 RMSNorm 的误差，反而破坏了共享 qspec 的平衡。qspec 方向暂不继续。

## 5.6 rsqrt-only CPU：只留 rsqrt 在 CPU

图手术不再改 RMSNorm 数学，而是在 partitioner 层拒绝 `aten.rsqrt`：

```python
class _NoRsqrtSupport(OperatorSupportBase):
    def is_node_supported(self, submodules, node):
        return 'aten.rsqrt' not in str(node.target)
```

这样：

- `mean/sum`、`add(eps)`、最后的 `mul` 都留在 NPU；
- 只有 `rsqrt` 落 CPU；
- 不需要 floor，也不需要改 `norm_eps`。

导出结果：

```text
resources_downloaded/tinystories42m/npu_rsqrt_only_ctx64/
  tinystories42m_u85_1024_ctx64_rsqrt_only.pte
  56,198,368 B
  18 个 Ethos-U delegate
```

terminal KV、输入/输出重排、Vela 重编译、PTE 重打包后，FVP：

```text
TINYSTORIES_SMOKE matches=10 total=10 success=1
TINYSTORIES_GENERATION matches=16 count=16 success=1
```

故事：

```text
Once upon a time in a small village, there was a little girl named Lily.
She loved to play outside with her friends. One day, while playing, she saw a big, red ball.
Lily was very happy and wanted to play with it.
```

这是目前最小切分的 16/16 方案：

```text
mean/sum       NPU
add(eps)       NPU
rsqrt          CPU
mul(embedding) NPU
```

结论：**rsqrt 的查表近似（以及它的输入量化）就是 floor/RMSNorm 实验里剩下的主要误差；add(eps) 和末端乘法可以安全留在 NPU。**

## 5.7 LUT 方案执行进展

本轮已经完成：

1. 导出干净的 full TOSA：

```text
resources_downloaded/tinystories42m/npu_lut_probe_ctx64/
  delegate_00.tosa   43,590,484 B
  quantized.pt2      73,880,463 B
```

2. 解析出 **17 个 rsqrt TABLE**，其中两个节点共享：

```text
b_aten_rsqrt_default_5_table_constant_common
```

其余节点各自有独立 table constant。

3. 确认 `rsqrt-only CPU` 是当前最小切分金标准：

```text
18 delegate
prompt 10/10
generation 16/16
```

4. host rsqrt 表探针已经开始，但发现 TOSA `TABLE` 的 index 语义不能按普通 `code >> 7` 处理：
   - TABLE 输入是 INT16，表长 513；
   - 输入前还有一次 `RESCALE`（multiplier 2^30、shift 41）；
   - 当前 probe 对部分节点的相对误差达到 100%+，说明 index/offset 还没有和 TOSA 语义对齐，**这组数字不能作为表误差结论**。

结论：LUT 方案的结构基础已经就绪，但还没有生成 per-node warp 的 TOSA/PTE/FVP 结果。

下一步严格按 `TinyStories_Rsqrt_LUT_Plan.md`：

1. 用 TOSA 1.0.1 的 TABLE index 规则重写 host probe；
2. 先做 **identity clone**（17 个表值相同但常量分开），确认 Vela 仍为 1 delegate；
3. 再做 per-node alpha，最后 per-node table fit；
4. 18-delegate 的 rsqrt-only CPU 继续作为精度金标准。

## 5.8 identity table clone：17 个独立 TABLE 可以单 delegate

为了验证“拆 table 不等于拆 delegate”，先做了一次 identity clone：

- 保留 17 个 rsqrt TABLE 的原始 513 点值；
- 为每个节点克隆独立的 table constant；
- 不改任何数值，只改常量名和 TABLE 引用。

结果：

```text
resources_downloaded/tinystories42m/npu_lut_probe_ctx64/
  delegate_00_split.tosa        42 MB
  payload_split/out.tosa        43,616,936 B
  payload_split/vela_payload.bin 37,502,960 B
```

Vela：

```text
NPU operators = 665 (100.0%)
input 45
```

也就是 17 个独立 table constant 仍然编译成 **1 个 Vela network、100% NPU operator、0 CPU op**。这证明：

- 拆表不会产生多 delegate；
- 当前 multi-delegate 的唯一来源是 `rsqrt` 被放到 CPU；
- 下一步只需要把 per-node table 值填进去，不需要改 partition。

这次 identity clone 没有做 terminal KV 和 PTE 重打包，因此还不是可上板 PTE；它只验证了 Vela 对 17 个独立 TABLE 常量的支持。

## 5.9 TOSA rsqrt TABLE 的实测语义

用真实 RMSNorm 输入对照后，当前表的行为可以确定为：

```text
table_index = clamp(round(mean(x^2) * 2) + 256, 0, 512)
table_output = table[table_index] / 2266
```

实测样本：

| mean(x²) | table index | table value | table 输出 | exact rsqrt |
|---:|---:|---:|---:|---:|
| 1.5557 | 259 | 1850 | 0.8164 | 0.8018 |
| 1.5132 | 259 | 1850 | 0.8164 | 0.8129 |
| 11.7444 | 279 | 668 | 0.2948 | 0.2918 |
| 11.7973 | 280 | 654 | 0.2886 | 0.2911 |
| 0.1931 | 256 | 32767 | 14.46 | 2.2756 |
| 0.00244 | 256 | 32767 | 14.46 | 20.24 |

关键发现：

- 表输入是 `mean(x²) * 2` 的整数 code，索引是 `code + 256`；
- 表值输出 scale 约 `1/2266`；
- 中高值段（mean≈1.5、11.7）误差很小；
- **低值段被 index 256 的 32767 饱和值截断**：`mean < 0.25` 全部映射到 14.46，而真实 rsqrt 可以到 20–37。

这就是 floor/qspec 都只能部分改善的原因：低值区不是 scale 不够细，而是表在 index 256 处有一个饱和哨兵，且 `mean*2` 的整数化把 `mean < 0.25` 全部压到同一个 code。

per-node warp 必须同时做两件事：

1. 把低值区 spread 到多个 table index；
2. 重写 index 256 附近的表值，不能保留 32767 哨兵。

identity clone 已经证明 17 个独立 table 可以单 network 编译；下一步是把上述 warp 和表值填进去。

## 5.10 第一版 per-node table warp：FVP 3/10、3/16

第一版 warp 的做法：

- 为每个 rsqrt 节点选择 `S = 256 / max(mean(x²))`；
- 把 table 前的 RESCALE 改成 `S/4096`；
- table index 仍为 `code + 256`；
- table 值改写为 `2266 / sqrt((index-256)/S)`。

结构上：

```text
1 个 Vela network
665 NPU operators，100% NPU
0 CPU op
```

但 FVP：

```text
prompt 3/10
generation 3/16
```

生成退化到 token 301 重复。

根因不在输入映射，而在 **table 输出 code 饱和**：

- table 输出是 INT16，最大 32767；
- 按原输出 scale，最大只能表示 rsqrt ≈ 14.46；
- 低能量节点的真实 rsqrt 可以到 20–37；
- 第一版 host 模拟直接用了 `1/sqrt(phys)`，没有模拟 32767 饱和，所以 host MAE 看起来很小，FVP 却很差。

结论：per-node 输入 scale 只能解决“低值区被压到同一个 code”，不能解决“低值区 rsqrt 超过 table 输出范围”。下一步必须同时改：

1. table 输出 scale / 后置 rescale，让 20–37 可表示；
2. 或者给低值区单独的 output scale，再在 table 后恢复。

## 5.11 校准 scale 版 per-node table：FVP 仍失败

按 7 步校准流程做了第二版：

1. 17 节点采样（主 prompt、生成、BOS-only，共 33 样本/节点）；
2. 统计 max/p99/p99.9；
3. 选 `s_tab = p99 * 1.1 / 32767`；
4. 输入 `S = min(256/max(mean), (32767*s_tab)^2)`，避免 table 输出饱和；
5. 重写 17 个 table；
6. table 后 RESCALE 改为 `s_tab * 2266 / 128`；
7. Vela 单 network、665 NPU op、100% NPU。

FVP：

```text
prompt 1/10
generation 0/16
```

文本退化为：

```text
Once upon a time in a small village,,,,,,,,,,,,,..............
```

说明第二版仍然把 rsqrt 表语义改坏了。当前最可能的问题：

- 全局 `s_tab = 1/2266` 只对第一个节点成立；
- 其他节点的 table 输出 rescale / 下游 scale 不同，不能共用 2266；
- 需要从每个节点的 TOSA 后置 RESCALE 和下游 dequant 反推各自的 `s_tab`，而不是假设全局 2266。

当前没有单 delegate 的可用 LUT 版本；18 delegate 的 rsqrt-only CPU 仍是唯一 10/10、16/16。

## 5.12 修正 per-node table 输出 factor 后：FVP 6/10、5/16

第三版修正了两个问题：

1. 每个节点的 table 输出 factor `F` 不再假设全局 2266，而是从原表值反推：

```text
F = median(table[i] * sqrt((i-256)/2))
```

2. table 后 RESCALE 改为：

```text
r_post = s_tab * F / 128
```

17 个节点的 F 从约 886 到 16,383 不等，证明全局 2266 确实是错误的。

FVP：

```text
prompt 6/10
generation 5/16
```

文本仍然退化（重复 `ally` / `bounce today`），但比第二版的 1/10、0/16 好，说明 factor 修正方向正确，table 的 index/输入映射仍未完全对齐。

当前结论：

- per-node table 结构、Vela 单 delegate、100% NPU 已经成立；
- 表值/scale 还没有和 TOSA TABLE 的真实 index 语义完全一致；
- 18 delegate 的 rsqrt-only CPU 仍是唯一 10/10、16/16。

## 6. 性能数据（FPGA TA ON，RMSNorm CPU 拆分版）

日志：

```text
build/logs/tinyStory_logs/TA_split/tinystory_test_TA_split_64tok_dedicated_1_2M.log
```

精度：

```text
10/10 prompt
16/16 generated
```

25 次前向：

```text
prompt 10 次: 40.936 s
decode 15 次: 61.413 s
合计:        102.349 s
```

单 token：

| 项目 | 耗时 |
|---|---:|
| E2E | 4,093.945 ms |
| NPU hardware TOTAL | 322.20 ms（7.9%） |
| invoke | 4,046.96 ms |
| CPU.Quantize | 2,147.81 ms |
| CPU.OtherKernel | 1,141.46 ms |
| CPU.Dequantize | 209.64 ms |
| CPU.StateCopy | 22.58 ms |
| Delegate.Total | 413.54 ms |
| NPU.Driver | 332.97 ms |
| Runtime.Other | 109.95 ms |

对比 full：

```text
full:        451.947 ms/token
RMSNorm CPU: 4,093.945 ms/token（9.1× 慢）
```

NPU 时间几乎不变，CPU Q/DQ 和 RMSNorm 浮点计算是主要开销。

## 7. 当前产物

### 精度上界

```text
resources_downloaded/tinystories42m/npu_rmsnorm_cpu2_ctx64/
  tinystories42m_u85_1024_ctx64_rmsnorm_cpu2_terminal.pte
  56,218,112 B，18 delegates
```

FPGA 固件：

```text
build/tinystories42m/test/fpga-rmsnorm-cpu2-ta/
  tinystories_test_ctx64_rmsnorm_cpu_fpga_ta.zip
```

### floor 1e-3

```text
resources_downloaded/tinystories42m/npu_relu1e3_ctx64/
  tinystories42m_u85_1024_ctx64_relu1e3_terminal.pte
```

### softmax-only

```text
resources_downloaded/tinystories42m/npu_softmax_cpu_dump_ctx64/
  tinystories42m_u85_1024_ctx64_softmax_cpu_terminal3.pte
```

## 8. 经验与注意事项

1. **UART 文件有 4 KiB 缓冲**：FVP 卡住时后续日志不会落盘；用 telnet raw 端口抓完整 UART。
2. **多 delegate 必须做 terminal KV**：否则 Vela 会把 KV 输出折叠成 alias，ET/Vela 接口不一致。
3. **输入重排必须匹配 shape + dtype**：只匹配 shape 会漏掉 INT32/INT16 的 `(1,)` 标量错位。
4. **RMSNorm 拆分要保留 sum 在 NPU**：否则暴露 INT32 `reduce_sum`，ET 侧期望 FP32。
5. **floor 形式会触发大量 RESCALE**：clamp/max/relu 都可能在 TOSA 转换阶段病态变慢。
6. **eps 不是越大越好**：1e-3 会改变模型数学；floor 1e-3 比 additive 1e-3 安全。
7. **临时文件要放 gitignored 目录**：Hydra 默认会在仓库根生成 `outputs/`，本轮已移动到 `build/tinystories42m/tmp/root-artifacts/`，并在 `.gitignore` 中忽略 `/outputs/`。

## 9. 下一步

1. rsqrt-only CPU 已经达到 10/10、16/16，说明 add(eps) 和末端乘法可以留在 NPU，CPU 只需要 rsqrt；
2. 下一步是 host rsqrt 表探针：用真实 RMSNorm 输入对比 exact rsqrt 与 513 点表，量化表误差；如果表误差是主因，再做低值区加密的分段表；
3. 分段表先在 host 验证，再编回 TOSA/Vela，目标是消掉 rsqrt-only CPU，回到单 delegate；
4. 如果自定义表无法在 Vela 落地，保留 rsqrt-only CPU（18 delegate）作为最终精度方案；
5. 长期方案：QAT。
