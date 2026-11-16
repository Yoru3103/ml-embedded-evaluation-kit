# MiniMind2 Small 部署小结

MiniMind2 Small 已部署到 MPS4 SSE-320 / Ethos-U85 Z1024。当前 UART demo 使用 128-token 上下文和 512 KiB Dedicated SRAM cache；64-token 版本保留作对照。模型量化和 Vela 编译在 WSL 的 `/home/mokuroo/documents/python/minimind/mlek_minimind2` 完成。Windows 工程负责嵌入 PTE 和构建固件。

| 关键版本 | 验证结果 | 用途 |
| --- | --- | --- |
| 初版 512 KiB terminal PTE | 保留 16 路 KV cache 输出 | 早期部署基线 |
| linear PTE | FVP prompt 32/32，生成 16/16，49 次 delegate 调用 | 工程默认及精度基线 |
| RMSNorm 分段回收三版 | FVP 均为 prompt 27/32，生成 4/16 | 保留 CPU/NPU 切分对照 |
| per-node rsqrt PTE，ctx64 | 单个 NPU delegate，FVP 和 FPGA 均为 prompt 28/32、生成 16/16，零缓存两次 top-1 均为 2 | 64-token 性能对照 |
| per-node rsqrt PTE，ctx128 | 单个 NPU delegate，INT16 KV，512 KiB cache；新增流式 UART 回复 | 当前 128-token demo |

per-node 版逐节点调整 RMSNorm 的 rsqrt 查表输入范围。固定生成样例恢复到 16/16，prompt 仍在位置 10、12、19、26 有差异。FPGA 开 TA 时，prompt 平均 E2E 939 ms，decode 平均 E2E 963 ms（15 次调用）。NPU 每次约 624 万 cycles，按配置的 50 MHz 折算约 125 ms。后续可对 `RunInference` 内未由 NPU 计数解释的耗时做分段测量。

## 128-token 流式对话

当前 128-token PTE 使用已验证的 per-node rsqrt 查表缩放、INT16 持久 KV 和 512 KiB Dedicated SRAM。PTE 为
`resources_downloaded/minimind2_small/minimind2_small_u85_1024_ctx128_w8a16_embedding8_per_node_warp_int16kv_512k.pte`，大小 28,651,392 B。Vela fast scratch 为 395,472 B。

客户端从固件 `READY` 帧读取 context 长度，因此同一套工具可连接 64 或 128-token 固件。128-token 固件最多接收 127 个 prompt token，剩余位置用于回复；过长输入仍由客户端从较早的用户文本开始截断。设备每生成一个 token 就发送带字节长度的 `CHUNK` 帧，Windows 客户端用增量 UTF-8 解码器立即显示；EOS 或上下文用尽时发送 `DONE`。

### FPGA TA 固件

FPGA 客户端使用单字节 `?` 握手，收到 `HELLO` 后才开放输入。连接时会在等待期限内重试握手。固件收到完整的 token 行后会先回 `ACCEPT`，随后开始推理。请将新版客户端与本节构建的固件配套使用。

从仓库根目录构建：

```powershell
.\source\app\use_case\minimind2_small\tools\build_test.ps1 `
  -Target FPGA `
  -BuildDirectory build/minimind2_small/test/fpga_uart_demo_ctx128 `
  -ModelPath resources_downloaded/minimind2_small/minimind2_small_u85_1024_ctx128_w8a16_embedding8_per_node_warp_int16kv_512k.pte `
  -CacheOffsets 1048576,1081344,1114112,1146880,1179648,1212416,1245184,1277952,1310720,1343488,1376256,1409024,1441792,1474560,1507328,1540096 `
  -ContextLength 128 -Demo -ProfileStages -Int16Kv -TimingAdapter
```

固件路径：`build/minimind2_small/test/fpga_uart_demo_ctx128/bin/mlek_minimind2_small.axf`。加载到 MPS4 后，关闭其他占用 UART0 的串口程序，再运行：

最终版本的计时口径、验收记录和文件哈希见 [阶段交付清单](docs/Closeout.md)。运行客户端时加 `-Measure` 可显示计时。

```powershell
.\source\app\use_case\minimind2_small\tools\fpga_console.ps1 -Port COM5
```

### FVP 快速演示

构建后可用功能 FVP 检查交互流程（FVP `--fast` 时间不代表 FPGA 性能）：

```powershell
.\source\app\use_case\minimind2_small\tools\build_test.ps1 `
  -Target FVP `
  -BuildDirectory build/minimind2_small/test/fvp_demo_fast_ctx128 `
  -ModelPath resources_downloaded/minimind2_small/minimind2_small_u85_1024_ctx128_w8a16_embedding8_per_node_warp_int16kv_512k.pte `
  -CacheOffsets 1048576,1081344,1114112,1146880,1179648,1212416,1245184,1277952,1310720,1343488,1376256,1409024,1441792,1474560,1507328,1540096 `
  -ContextLength 128 -Demo -FastDemo -ProfileStages -Int16Kv
```

```powershell
.\source\app\use_case\minimind2_small\tools\run_fvp_test.ps1 `
  -BuildDirectory build/minimind2_small/test/fvp_demo_fast_ctx128 -UartPort 15660
```

另一个 PowerShell 窗口运行：

```powershell
.\resources_downloaded\windows-executorch\env\Scripts\python.exe -B `
  .\source\app\use_case\minimind2_small\tools\fvp_console.py `
  --tokenizer .\resources_downloaded\minimind2_small\hf_tokenizer --port 15660
```

## 64-token 对照固件

### FVP 单轮对话

在 Windows PowerShell 的仓库根目录启动演示固件：

```powershell
.\source\app\use_case\minimind2_small\tools\run_fvp_test.ps1 -BuildDirectory build/minimind2_small/test/fvp_demo_fast
```

在另一个 Windows PowerShell 窗口，从仓库根目录运行客户端。默认 UART0 端口为 15650。如果 FVP 输出的 `telnetterminal0` 端口不同，在客户端追加 `--port <实际端口>`：

```powershell
.\resources_downloaded\windows-executorch\env\Scripts\python.exe -B .\source\app\use_case\minimind2_small\tools\fvp_console.py --tokenizer .\resources_downloaded\minimind2_small\hf_tokenizer
```

64-token 目录可继续作为旧版对照。每轮清空 KV cache，回复预算为上下文长度减去实际 prompt token 数，遇到 EOS 提前结束。

### FPGA 串口单轮对话

64-token 旧版固件位于 `build/minimind2_small/test/fpga_uart_demo/bin/mlek_minimind2_small.axf`。128-token 固件路径和命令见上文。关闭占用同一 COM 口的串口工具，在 Windows PowerShell 从仓库根目录运行客户端，将 `COM5` 换成板卡 UART0 对应端口。

```powershell
.\source\app\use_case\minimind2_small\tools\fpga_console.ps1 -Port COM5
```

客户端使用 115200、8N1、无流控。输入中文后回车即可逐 token 看到回复；输入 `/quit` 关闭连接。Windows 负责聊天模板和 tokenizer，板端接收 token ID 并推理，再以长度帧回传 token 原始字节。客户端增量解码 UTF-8，避免多字节字符被 token 边界切开时出现乱码。固件启用 TA，使用 INT16 KV PTE 和原地限幅快路径。

### 部署产物

以下 Windows 路径相对仓库根目录。

- 模型和 tokenizer 在 `resources_downloaded/minimind2_small/`。128-token demo 使用 `minimind2_small_u85_1024_ctx128_w8a16_embedding8_per_node_warp_int16kv_512k.pte`；64-token PTE 保留作对照。
- 128-token FPGA TA demo 在 `build/minimind2_small/test/fpga_uart_demo_ctx128/bin/mlek_minimind2_small.axf`。
- FPGA UART 日志在 `build/logs/minimind_logs/fpga/minimind_u85_1024_dedicated_512k.log`。FVP 主候选和三阶段对照日志仍在 `build/logs/minimind_logs/`。
- [排查与部署报告](docs/261111-MiniMind2_U85_排查与部署报告.md)记录了复现步骤和构建配置。WSL 保留源权重、导出脚本、`report_linear` 和 `report_rmsnorm_reclaim`。

## 归档

Windows 历史资源、诊断文件、日志和旧构建树在 `build/minimind_archive_20261113/`，`manifest.csv` 记录每项的原始路径和归档路径。旧 CMake/Ninja 构建树含绝对路径，如需复现，应先恢复原路径或重新配置。历史报告中引用的归档文件可按清单查找。

WSL 的早期掩码、校准和 softmax 实验报告及日志在 `/home/mokuroo/documents/python/minimind/mlek_minimind2/archive_20261113/`，原路径记录在同目录的 `manifest.tsv`。
