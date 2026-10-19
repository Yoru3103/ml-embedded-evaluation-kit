<!-- SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com> -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# U85 双通道 PSRAM 的 Vela / TA 基线

这套配置用于当前 U85 Z1024 的性能对比：两条独立 x16 PSRAM 通道，假设每通道
400 MHz 时钟、双沿传输；目标 NPU 为 400 或 800 MHz。Vela 使用
`scripts/vela/u85_psram_dual_x16.ini`，固件使用
`scripts/cmake/timing_adapter/ta_config_u85_psram_dual_x16.cmake`。

## 参数依据

| 参数 | 基线 | 依据 |
|---|---|---|
| PSRAM 理论总峰值 | 3.2 GB/s | 2 通道 × 2 B × 400 MHz × 2 |
| 数据阶段带宽预算 | 2.4 GB/s，读写共享 | 暂取峰值的 75%，用于初始对比 |
| 读 / 写响应延迟 | 100 / 100 ns | 暂定的 AXI 侧等效延迟，包含控制器开销 |
| 每个 EXT AXI 的最大挂起读 / 写 | 2 / 2 | 暂定的有限流水能力，避免沿用 DDR 的 64 / 32 |
| Vela 最小高效外存突发长度 | 128 B | 沿用当前 U85 参考假设 |
| SRAM 时序 | 每端口 16 B/周期，读写延迟 32 周期 | 沿用当前参考值，单独对比 PSRAM 影响 |
| SRAM cache 预算 | 默认 384 KiB，可覆盖 | 沿用现有 Dedicated_Sram 预算，并非芯片容量结论 |

75% 用于给持续数据传输留出余量，并不是实测的端到端效率；起始响应延迟另行建模，
实际模型的有效带宽可能更低。100 ns 与并发数同样是工程假设，后续优先用器件和
控制器资料校准这三项。若“400M”实际指 400 MT/s，理论峰值和带宽预算应减半。

两条 PSRAM 通道不自动意味着两个 NPU AXI；本配置另外保留当前 Z1024 的
两个 SRAM AXI、两个 EXT AXI 的参考拓扑，并假设两个 EXT AXI 能均衡使用两条
PSRAM 通道。两端口的带宽预算合计为 2.4 GB/s，不是每端口 2.4 GB/s。

## 配对数值

U85 AXI 宽度为 16 B。令目标 NPU 频率为 f，外存 AXI 数量为 2，则：

```text
Vela 总带宽 = 16 × f × Dram_ports_used × Dram_clock_scale
TA 每端口带宽预算 = 16 × f × EXT_BWCAP / (EXT_PULSE_ON + EXT_PULSE_OFF)
等效响应延迟 = EXT_RLATENCY / f
```

| 参数 | NPU 800 MHz | NPU 400 MHz |
|---|---:|---:|
| Vela System_Config | Ethos_U85_PSRAM_Dual_X16_800 | Ethos_U85_PSRAM_Dual_X16_400 |
| `core_clock` | 800e6 | 400e6 |
| `Dram_clock_scale` | 0.09375 | 0.1875 |
| `Dram_ports_used` | 2 | 2 |
| `Dram_read_latency` / `Dram_write_latency` | 80 / 80 | 40 / 40 |
| `Dram_max_reads` / `Dram_max_writes` | 2 / 2 | 2 / 2 |
| `EXT_RLATENCY` / `EXT_WLATENCY` | 80 / 80 | 40 / 40 |
| `EXT_MAXR` / `EXT_MAXW` / `EXT_MAXRW` | 2 / 2 / 0 | 2 / 2 / 0 |
| `EXT_PULSE_ON` / `EXT_PULSE_OFF` | 4095 / 1 | 4095 / 1 |
| `EXT_BWCAP` | 384 | 768 |
| `EXT_MODE` | 1 | 1 |

本文 GB/s 按十进制计算。当前 Vela 5.1.0 的带宽报告以 2^30 B/s 换算但标注为
GB/s，因此 2.4 GB/s 的预算在报告中显示为约 2.24 GB/s。

`MAXRW=0` 表示不再额外限制读写合计并发，并不表示零事务。
带宽窗口的 cap 都是 16 beats 的整数倍，对齐现有驱动的最大 256 B 突发，
减少 TA 按事务限额时的舍入影响；窗口内仍可能出现突发传输和等待。

## 编译与固件选择

以下命令从仓库根目录执行。先用现有 VWW 原始模型检查编译流程；换成其他模型时，
保留这些目标参数，并使用该模型原有的导出 / 校准流程。

```bash
resources_downloaded/env/bin/python -m ethosu.vela \
  resources_downloaded/vww/vww4_128_128_INT8.tflite \
  --config scripts/vela/u85_psram_dual_x16.ini \
  --accelerator-config ethos-u85-1024 \
  --system-config Ethos_U85_PSRAM_Dual_X16_800 \
  --memory-mode Dedicated_Sram --arena-cache-size 393216 \
  --optimise Performance --verbose-config --verbose-performance \
  --output-dir /tmp/mlek-psram-vww-800
```

已有 YOLO PTE 导出命令对应增加或替换以下选项：

```text
--vela-config scripts/vela/u85_psram_dual_x16.ini
--system-config Ethos_U85_PSRAM_Dual_X16_800
--memory-mode Dedicated_Sram
--arena-cache-size 393216
```

在原来的完整固件 CMake 配置命令中使用一个新的 build 目录，增加或替换：

```text
-DETHOS_U_NPU_ID=U85
-DETHOS_U_NPU_CONFIG_ID=Z1024
-DETHOS_U_NPU_MEMORY_MODE=Dedicated_Sram
-DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=ON
-DTA_CONFIG_FILE=/home/xx/ml-embedded-evaluation-kit/scripts/cmake/timing_adapter/ta_config_u85_psram_dual_x16.cmake
-DPSRAM_TARGET_NPU_MHZ=800
```

固件必须加载这次重新编译的模型；只切换 TA 不会改变模型的编译调度。
400 MHz 组合需要同时把 Vela 的 System_Config 后缀改为 `_400`，并设置
`PSRAM_TARGET_NPU_MHZ=400`。该变量只选择时序模型，不修改 FPGA / FVP 主时钟。
配置文件使用普通 CMake 变量确定 TA 值，避免旧 DDR cache 值覆盖本文件；固件启动时
检查 `TA_EXT readback` 的 cap、延迟和并发数是否与上表一致。

权重和主 arena 通过 EXT，fast scratch 通过 SRAM。保持编译时的 cache 预算与
固件实际提供的 SRAM 分配一致；两种频率的对比使用相同 cache 容量、模型和输入。
若要测试 256 KiB 等其他容量，在 Vela 和固件两侧同步调整。

## 周期与时间

这里的 400 / 800 MHz 是目标芯片的模型参考频率。保持目前 FPGA 50 MHz、FVP
默认启动时钟即可先比较 `NPU TOTAL`。在 TA 和 NPU 计数时钟关系成立的前提下，
50 MHz FPGA 上对应的两端口总带宽预算分别为 300 / 150 MB/s；这两个值只是周期
比例换算，不是板上 DDR 的带宽测量。

对匹配 `_400` / `_800` 的模型与 TA 测量 NPU 周期 C 后，目标时间近似为：

```text
400 MHz: NPU 时间 [ms] = C / 400000
800 MHz: NPU 时间 [ms] = C / 800000
```

分别记录 Vela 估计、FVP NPU 周期、FPGA NPU 周期。CPU TOTAL 和 DURATION
保留各自平台含义；这套 NPU 侧 TA 参数没有模拟 CPU 直接访问 PSRAM 的全部开销。
报告中同时注明 75% 带宽预算、100 ns 延迟和 384 KiB 默认 cache 等假设。

## 配置验证记录

2026-10-15，使用项目共享环境中实际安装的 Vela 5.1.0，分别以两组配置编译
`resources_downloaded/vww/vww4_128_128_INT8.tflite`，两组均成功生成模型，
74 个算子全部放置到 NPU；报告中的加速器频率分别为 400 / 800 MHz，外存设计
带宽均显示为 2.24 GB/s。编译日志及模型分别位于
`/tmp/mlek-psram-vww-400/` 和 `/tmp/mlek-psram-vww-800/`。

通过项目现有 TA 模板生成头文件，核对了两组参数的带宽、延迟和寄存器范围，
并验证同一 CMake build cache 从 800 切换到 400 时正确更新参数，旧的 DDR
`EXT_BWCAP=3750` 不覆盖新配置，非法频率被拒绝。这些检查确认配置可用且数值
配对；本记录不包含新固件的 FVP / FPGA 性能测量。
