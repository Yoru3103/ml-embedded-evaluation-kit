# KWS_ZH Dedicated 256 KiB：板上故障修复与固件配置

Shared 已通过开发板执行。此版修正 Dedicated 的 scratch 地址，并将 Vela 编译预算和
固件专用缓存统一为 256 KiB；本次构建和 FVP 检查结果记录在固件包的 validation 目录。

## 故障定位

板上首个窗口报 `STATUS=0x5804`、`QREAD=456`、`cmd_end_reached=0`。
根据 core-driver 的 U85 寄存器定义，该状态展开如下：

| 字段 | 值 | 含义 |
|---|---:|---|
| bus_status，bit 2 | 1 | 总线访问失败，NPU 停止 |
| faulting_interface，bit 11 | 1 | EXT 接口 |
| faulting_channel，bits 12–15 | 5 | RD_IFM_STREAM，输入特征流读取 |
| cmd_parse_error，bit 4 | 0 | 命令解析正常 |
| ecc_fault，bit 8 | 0 | ECC 状态正常 |
| branch_fault，bit 9 | 0 | 分支状态正常 |
| cmd_end_reached，bit 5 | 0 | 命令流中途停止 |

上一版包装脚本将两种模式的临时分配区统一设置成 `0x31100000`。
ExecuTorch 从这个池分配普通 scratch，而 Vela 产物的输入使用 region 1、offset 0。
HAL 为 Dedicated 配置 `NPU_REGIONCFG_1=3`，通过 AXI EXT 访问该区域；
Shared 配置 `NPU_REGIONCFG_1=0`，通过 AXI SRAM 访问。
因此同一个 SRAM 地址在 Shared 中匹配，在 Dedicated 中形成接口与地址区域错配，
与板上 EXT 输入读取总线错误相符。

修复将 Dedicated 的普通 scratch 移到 DDR `0x72000000`，与既有 ASR、二维码、Pose
Dedicated 样例的临时区基址一致。fast scratch 继续使用 SRAM `0x31000000`。

## 256 KiB 配置与模型内存

| 项目 | 新配置 |
|---|---|
| NPU | Ethos-U85，1024 MAC |
| Vela system / memory | Ethos_U85_SYS_DRAM_Mid / Dedicated_Sram |
| Vela arena cache budget | 262144 bytes，256 KiB |
| 固件 cache_arena | 262144 bytes，基址 0x31000000 |
| 普通 scratch | 7920 bytes，region 1，DDR 临时池 |
| fast scratch | 232480 bytes，region 2，SRAM 专用缓存 |
| 权重数据 | 314320 bytes，DDR |
| 命令数据 | 5876 bytes，DDR |
| Vela SRAM memory used | 232480 bytes，227.03125 KiB |
| Vela DRAM memory used | 322240 bytes，314.6875 KiB |
| PTE 文件 | 322848 bytes |
| 方法池 | 1 MiB，DDR |
| 临时池 | 3 MiB，0x72000000–0x722fffff |
| 显示帧缓冲基址 | 0x74000000 |

Vela 的 SRAM/DRAM 数字描述模型分配；固件池容量另列。
原 384 KiB 版的 fast scratch 同样为 232480 bytes，因此故障修复点是普通 scratch 的
存放位置。256 KiB 版重新执行 Vela 编译，量化参考在 64 个评估窗口上与原版逐元素一致。

新 PTE SHA-256：
`fe2f05fd06fac8f620b7ffeddd194e1e7edd8ffb926f71382d5a215a852e50ac`。

编译后仍为一个 Ethos-U delegate，边界由 Cortex-M 执行 quantize / dequantize。
Vela 调度保持 30 个 pass；slice、cat 和布局算子的消除、转换流程沿用此前的模型分析。

## 验证结果

- Dedicated 256 KiB FVP 完成全部3段音频、5个窗口，共1980个输出分数。
- 与 PT2E 参考逐帧比较，最大绝对误差为 `4.6566e-10`，峰值误差和阈值判定分歧均为0。
- Hi_Xiaowen 在1.145 s触发，Nihao_Wenwen 在1.195 s触发，背景片段无触发。
- FVP 日志确认 cache 为262144 bytes，临时区基址为0x72000000。
- 配置回归检查确认：Dedicated 误设0x31100000临时区时，CMake直接报告地址错配。

## 重建与上板

```powershell
./scripts/windows/build-kws-zh.ps1 -MemoryModes Dedicated_Sram -CacheSizeKiB 256 -Package
```

模型：`resources_downloaded/kws_zh/pte_Dedicated_Sram_256k/wekws_dstcn_u85.pte`。
固件：`build/board-test-packages/kws_zh/pte-dedicated-256k-20261027.zip`。

替换包内完整 SOFTWARE 和 images.txt，匹配本次 AXF、模型、缓存与临时区配置。
启动日志应出现 `fast cache size: 262144` 和
`temporary allocator pool. Base: 0x72000000`。
两个正样本应分别产生 Hi_Xiaowen、Nihao_Wenwen 的 WAKE，背景样本保持未触发。

代码依据：

- [U85 STATUS 位定义](../../dependencies/core-driver/src/ethosu_interface_u85.h)
- [HAL region 接口配置](../../source/hal/source/components/npu/CMakeLists.txt)
- [ExecuTorch scratch 分配](../../dependencies/executorch/backends/arm/runtime/EthosUBackend.cpp)
- [构建脚本](../../scripts/windows/build-kws-zh.ps1)
- [使用指南](kws_zh.md)
