# MiniMind2 Small 阶段交付

MPS4 SSE-320 / Ethos-U85 Z1024，512 KiB Dedicated SRAM，静态 128-token 上下文，单 NPU delegate，INT16 持久 KV。每轮独立，prompt 与回复共享 128 token。

## 产物

以下路径以仓库根目录为基准。Windows 构建和运行命令见 [README](../README.md)。

| 文件 | SHA-256 |
| --- | --- |
| `build/minimind2_small/test/fpga_uart_demo_ctx128/bin/mlek_minimind2_small.axf` | `e387a92278409cc0ce6e0a4c057cd6adaef88174c4d81fd885ded730cc98656f` |
| `build/minimind2_small/test/fvp_demo_fast_ctx128/bin/mlek_minimind2_small.axf` | `5333e2f72196b6bcb107ae18314819c1c807c6f8636f217a7ba2ad61d505b1c7` |
| `resources_downloaded/minimind2_small/minimind2_small_u85_1024_ctx128_w8a16_embedding8_per_node_warp_int16kv_512k.pte` | `7994c3bec421b0f09bc127ce0c6d904b9714384e5a61a095f2d1c106a0d65955` |
| `resources_downloaded/minimind2_small/tokenizer_bytes.bin` | `364dc08665fb45f3a3440898135e86fdfc05921c8b206a7b130ea8412701c3ef` |

主机 tokenizer 位于同一资源目录的 `hf_tokenizer/`。模型量化、Vela 编译和 PTE 转换在 WSL 的 `/home/mokuroo/documents/python/minimind/mlek_minimind2` 执行，128-token 入口为 `40_export_minimind2_ctx128.sh` 至 `43_repack_ctx128_per_node.py`，INT16 KV 转换记录在 `report_ctx128_512k_warp/transform_int16_kv.json`。实际首次导出的 checkpoint 在 `report_ctx128/report_ctx128/quantized.pt2`。复用 64-token 校准输入和 per-node rsqrt 拟合参数。Vela fast scratch 395,472 B，最终 planned buffer 1,636,864 B，KV 偏移以最终压紧后的 PTE 为准。

## 验收与计时

- 用户提供的 COM5 重连记录显示，第一次连接中「你好！」（26 token）和「hello!」（27 token）均以 EOS 结束。退出客户端并再次连接后，「介绍下你自己」（28 token）也以 EOS 结束。该记录覆盖了完成回复后的重连。
- 原始实板计时日志位于 `build/logs/minimind_logs/fpga/minimind_128tok_u85_1024_dedicated_512k.log`。三条输入为 26、29、36 token，主机首段文本延迟分别为 11.92、13.16、16.31 秒，整轮分别为 15.47、30.81、43.01 秒。板端 decode 为 434.2–434.7 ms/次。
- 最终固件已构建。客户端加 `-Measure` 后，主机 `first_text_ms` 和 `total_ms` 分别量到首段可显示 UTF-8 文本和收到 DONE。板端 `prefill_infer_ms` 和 `decode_infer_ms` 是各阶段模型调用累计耗时，`decode_ms_per_call` 是后续调用均值，包含预测 EOS 的调用（若发生）。旧日志中跨 PMU 重置计算的板端 prefill、first_chunk 和 total 字段无效。最终计时帧待实板复测。

## 精度和归档

- 11 条同模板、同预算的对照保存在 `build/minimind2_small/fp32_ctx128_comparison.jsonl` 和 `build/minimind2_small/quantized_pt2_ctx128_comparison.jsonl`。FP32 已有答偏与重复。Python 无效代码和勇气故事反问在量化 PT2 已出现。量化 PT2 与 FPGA 的停止原因 10/11 一致，部分文本仍分叉。
- 阶段模型、64-token 性能对照、128-token 固件和实板日志保留在原路径。Windows 旧构建和辅助日志的移动记录见 `build/minimind_archive_closeout/manifest.csv`。恢复旧构建树后应重新配置 CMake。
- WSL 的 18 个旧 RMSNorm 实验 Vela 缓存移到项目内的 `archive_closeout/`，原路径见 `archive_closeout/manifest.tsv`。128-token 导出和重打包依赖的 TOSA、PT2、拟合参数和报告保留在原路径。
