# TinyStories rsqrt 查表修复方案（单 delegate）

日期：2026-11-13\
目标：把 RMSNorm 的 `rsqrt` 从 CPU 收回 NPU，恢复 **1 个 Ethos-U delegate、0 个 CPU op**，同时保持 **prompt 10/10、generation 16/16**。

## 1. 当前事实

| 版本 | delegate | prompt | gen | 说明 |
|---|---:|---:|---:|---|
| full 单 delegate | 1 | 6/10 | 11/16 | 基线 |
| floor 1e-3 | 1 | 10/10 | 13/16 | 改数学，生成不够 |
| qspec | 1 | 8/10 | 6/16 | 固定 scale 破坏共享 qspec |
| **rsqrt-only CPU** | **18** | **10/10** | **16/16** | 精度金标准，但不可接受 |
| RMSNorm CPU | 18 | 10/10 | 16/16 | 更重的 CPU 版本 |

结论：

- `add(eps)` 和末端 `mul` 可以留在 NPU；
- 只把 `rsqrt` 放 CPU 就能恢复 10/10、16/16；
- 因此剩余误差集中在 **rsqrt TABLE 的输入量化/查表近似**；
- 目标是只改 TABLE 和它的输入映射，不拆 delegate。

## 2. TinyStories 的 rsqrt 结构

full TOSA 中每个 RMSNorm 的 rsqrt 是一条：

```text
mean/sum -> RESCALE -> TABLE(rsqrt) -> RESCALE -> mul
```

关键事实：

- TinyStories 有 **17 个 rsqrt TABLE**（16 个 layer norm + final norm）；
- 多个节点共享同一个 table constant，例如 `b_aten_rsqrt_default_5_table_constant_common`；
- TABLE 固定 513 个 INT16 knot，输入 code 范围 [-32768, 32767]；
- 低值区曲线最陡，BOS/低能量 token 与普通 token 的输入范围差异很大；
- 当前 floor/qspec 的失败说明：只改全局 scale 不够，必须改 **输入 code 到物理值的映射**。

## 3. 总体路线

```text
A. 采集真实 rsqrt 输入 code + 节点 qparams
B. host 上拟合分段输入 warp（全局 alpha -> per-node alpha）
C. 在 TOSA 里插入 warp 并重写 513 点 TABLE
D. Vela 编译成单 delegate + FVP 回归
E. 若 alpha 不够，直接拟合 per-node table 值
```

## 4. Phase A：采集与解析

### A1. 导出干净 full TOSA

用未改数学的 full 配置导出，打开 `TINYSTORIES_DUMP_TOSA=1`，得到单 delegate 的 `delegate_00.tosa`。

### A2. 解析 17 个 rsqrt TABLE

从 TOSA JSON 中提取：

- `TABLE` 的输入 tensor 名；
- 输入前的 `RESCALE` qparams（multiplier/shift/input_zp/output_zp）；
- TABLE constant 名和 513 个 INT16 值；
- TABLE 后的 `RESCALE`（output_scale/output_zp/qmin/qmax）；
- 每个节点的 `input_scale`、`low_max`、`high_max`。

输出：

```text
rsqrt_nodes.json
```

格式参考 MiniMind：

```json
{
  "node_configs": {
    "rsqrt": {
      "input_scale": 0.000244140625,
      "low_max": 0.25,
      "high_max": 8.0,
      "output_scale": 0.001765,
      "output_zp": 0,
      "output_qmin": -32767,
      "output_qmax": 32767
    }
  },
  "samples": [
    {"node": "rsqrt", "case": "prompt0", "qcode": 1234, "pt2_rsqrt": 0.9123}
  ]
}
```

### A3. 采样真实输入

对以下输入做 hook，记录每个 `aten.rsqrt` 的输入 code：

- 当前 10 个 prompt token；
- 16 个贪心生成 token；
- BOS-only；
- 额外 holdout：不同故事开头、短句、低能量输入。

每个节点至少 200–300 个样本；记录：

```text
node, case, position, qcode, input_scale, exact_rsqrt, table_rsqrt
```

### A4. 量化当前表误差

对每个节点计算：

- 当前 513 点表 + 线性插值 vs exact rsqrt；
- 误差分布：MAE、p95、p99、max；
- 按输入区间分组：低值（`< low_max`）、中值、高值（BOS）；
- 端到端影响：host 上只替换 rsqrt 为当前表，看 token 从 10/10 掉到多少。

验收：如果“只换表”能把 10/10 打到 6/10 左右，说明表误差确实是主因，继续；如果几乎不变，停止查表路线。

## 5. Phase B：host 分段 warp 拟合

### B1. 映射形式

采用 MiniMind 的 piecewise-affine：

```text
u = 归一化输入 code [0,1]
alpha = 低值区占用的 code 比例

physical(u) =
    low_max * u / alpha                     , u <= alpha
    low_max + (high_max - low_max) *
        (u - alpha) / (1 - alpha)           , u > alpha
```

TABLE 的 513 个 knot 用 `1/sqrt(physical(u))` 生成，再按 output_scale 量化。

输入侧用 TOSA 整数算子实现：

```text
CLAMP(low)  -> RESCALE(alpha * 65536 / low_code)
CLAMP(high) -> RESCALE((1-alpha) * 65536 / (high_code-low_code))
            -> SUB/ADD 组合成 table code
            -> RESCALE 回 INT16
```

所有 RESCALE 用 `SINGLE_ROUND`，host 模拟必须按 TOSA 的 64-bit `value*multiplier + (1<<(shift-1)) >> shift` 实现，不能用简单的 `np.rint` 近似。

### B2. 全局 alpha 扫描

先对全部节点用同一个 alpha：

```text
alpha = 0.60, 0.65, ..., 0.95
```

评估：

- 每个节点的 code MAE / p99；
- host 端 10 prompt + 16 generation token 匹配；
- BOS-only、低能量 holdout 是否退化。

### B3. per-node alpha

对每个节点独立选 alpha：

- 训练集：每节点 200+ 样本；
- holdout：BOS、短句、英文/中文不同开头；
- 目标函数：先最小化 p99 code error，再最小化端到端 token error；
- 输出 `per_node_alpha_fit.json`。

### B4. host 验收

必须同时满足：

1. 17 个节点的 p99 code error 明显低于当前表；
2. host 端 10 prompt 至少 10/10；
3. 16 generation 至少 16/16；
4. BOS-only 和低能量 holdout 不退化；
5. 与 rsqrt-only CPU 金标准逐 token 对比，无分叉。

只有 host 达标才进入 TOSA 改写。

## 6. Phase C：TOSA 改写

### C1. 定位 TABLE

只处理 table constant 名包含 `rsqrt` 的 17 个 TABLE，不要碰：

- `exp` / `reciprocal`（softmax）；
- `silu`；
- RoPE 的 sin/cos 表。

### C2. 插入 warp

对每个 rsqrt TABLE：

1. 在 TABLE 前插入 `CLAMP + RESCALE + SUB/ADD` 的整数 warp；
2. 用 warp 后的 table code 作为 TABLE 输入；
3. 生成新的 513 点 table constant；
4. 如果多个节点共享 table constant，**为每个节点克隆一份**，避免 per-node alpha 互相污染。

### C3. 保持单 delegate

warp 只使用 TOSA 基础算子：

```text
CLAMP, RESCALE, SUB, ADD
```

这些 Vela 都支持；不要引入 SELECT/动态 shape。

检查：

- TOSA 中 17 个 rsqrt TABLE 都已被改写；
- 没有新增 CPU op；
- Vela 报 0 CPU op、1 个 delegate；
- KV 16 输入/16 输出契约不变。

## 7. Phase D：Vela 与 FVP

### D1. 编译

```text
U85-1024, Dedicated_Sram 1.2 MiB
Vela 5.2.0
```

检查：

- 1 个 delegate payload；
- 17 个 rsqrt 全部在 delegate 内；
- 16 个 KV 输出完整；
- scratch 不超限。

### D2. FVP 回归

必须跑：

- 10 prompt；
- 16 generation；
- zero-cache position-1；
- 生成文本。

验收：

```text
prompt 10/10
generation 16/16
zero-cache 与 host 一致
单 delegate
```

### D3. 若 FVP 不达标

按顺序排查：

1. 第一个分叉位置的 17 个 rsqrt 节点：输入 code、warp code、TABLE 输出、后续 logit；
2. 对比 host warp 模拟与 FVP 的 code；
3. 检查 `SINGLE_ROUND` 舍入；
4. 检查是否有节点输入超出 `high_max` 被 CLAMP。

## 8. Phase E：per-node table 直接拟合

如果 per-node alpha 仍不够，直接拟合每个节点的 513 点 table：

- 目标：最小化 `table(code) - exact_rsqrt(physical(code))`；
- 约束：单调、非负、输出范围；
- 可用最小二乘 + 单调回归；
- 低值区增加 knot 密度，高值区保 BOS；
- 仍然只改 TABLE，不改 delegate 结构。

这是比 alpha 更强的修复，但需要更多样本，且要防止过拟合到当前 10+16 token。

## 9. 验收矩阵

| 阶段 | 指标 | 通过条件 |
|---|---|---|
| A | 当前表 host 误差 | 只换表能复现 full 的 6/10、11/16 |
| B | host 10+16 | 10/10、16/16 |
| B | holdout | BOS、低能量、不同开头不退化 |
| C | TOSA/Vela | 1 delegate、0 CPU op、17 rsqrt 全在 NPU |
| D | FVP | 10/10、16/16、zero-cache 一致 |
| E | FPGA | 文本连贯、性能可测 |

## 10. 交付物

```text
source/app/use_case/tinystories/tools/
  probe_rsqrt_inputs.py        # 采集 17 节点输入 code
  fit_rsqrt_warp.py            # 全局/per-node alpha 拟合
  warp_rsqrt_table.py          # TOSA TABLE 改写
  validate_rsqrt_warp.py       # host 逐 token 对比
```

```text
resources_downloaded/tinystories42m/
  rsqrt_probe.json
  rsqrt_alpha_fit.json
  tiny_stories_rsqrt_warp.tosa
  tiny_stories_rsqrt_warp.pte
```

## 11. 风险

1. **Vela 对自定义 TABLE 的支持**：需要确认 TABLE constant 可以任意 513 点，而不是由 qparams 自动生成。
2. **共享 table constant**：per-node alpha 必须克隆 table，否则节点互相污染。
3. **BOS 高能量**：`high_max` 必须覆盖 BOS，否则高值被 CLAMP。
4. **低值分辨率**：alpha 太小时低值区仍粗；需要 per-node alpha 或直接 table 拟合。
5. **过拟合**：10+16 token 样本太少，必须加 holdout。
6. **TOSA/Vela 版本差异**：host 的 `SINGLE_ROUND` 必须与 TOSA 1.0.1 一致。

## 12. 推荐执行顺序

1. Phase A：导出干净 full TOSA，采集 17 节点输入，量化当前表误差；
2. Phase B：全局 alpha -> per-node alpha，host 先到 10/10、16/16；
3. Phase C：改写 TOSA，Vela 单 delegate；
4. Phase D：FVP 10+16；
5. 若不够，Phase E：直接拟合 per-node table；
6. 最终 FPGA test/demo 固件。
