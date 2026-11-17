# wav2letter 可选分段计时

默认构建不启用额外阶段计时，两种模式互斥：

| 模式 | CMake 选项 | 输出 |
|---|---|---|
| 简洁 | `ASR_SIMPLE_STAGE_PROFILING=ON`、`ASR_STAGE_PROFILING=OFF` | 每个窗口六段，逐次打印 |
| 详细 | `ASR_STAGE_PROFILING=ON`、`ASR_SIMPLE_STAGE_PROFILING=OFF` | 原有逐帧 MFCC、前后处理细项和官方 PMU |

两者均要求 `CPU_PROFILE_ENABLED=ON`。简洁档支持 MPS4、TFLM、Ethos-U、GNU/Clang 链接器。

## 六段的边界

按 `Inference 1/2`、六段记录、`Inference 2/2`、六段记录的顺序输出。
保留官方 `Profile for ...` / `CPU TOTAL` / `DURATION` 风格，阶段时间以微秒打印。
没有跨窗口统计、均值或极值，也不重复打印整段 Invoke 或 pipeline 合计。

| 输出名 | 范围 |
|---|---|
| Preprocess | 前处理整体减去输入量化与排列 |
| Quantize | 输入特征量化、排列和写入 INT8 张量 |
| NPU.Driver | 同步 `ethosu_invoke_v3()` 从进入到返回，包括提交、驱动、等待和相关缓存维护 |
| Dequantize | 148 个胜出字符分数的反量化，不是整张概率张量反量化 |
| Invoke.Other | `model.RunInference()` 减去 `NPU.Driver`，包含其余框架调度和包装开销 |
| Postprocess | 窗口后处理整体减去分数反量化，主要为上下文屏蔽、Argmax、标签与结果分配 |

六段均来自同一个 CPU elapsed 时钟，互不重复；NPU.Driver 不是 NPU PMU 的纯硬件周期。
量化发生在 Invoke 之前，反量化发生在 Invoke 之后，不从 Invoke 重复扣除。
最终跨窗口 token 合并、文本解码、结果展示及音频采集不在上述六段内。
阶段采样有少量开销，串口打印在本窗口计算完成后执行。

简洁档不编译逐帧 MFCC 钩子，不保留详细档的 4096 条事件缓冲。
为避免对 148 个短标量表达式分别采样，只有简洁档会把胜出分数还原移到一个独立循环；
保留同样的 Argmax、标签顺序和 double 反量化表达式。详细档及默认构建保留原循环结构。
六段收集器每窗清空；缺少阶段、出现重复 NPU 调用或子阶段大于父阶段时明确报错。

## Windows 构建

输入是已由 Vela 编译、与内存模式和 cache 匹配的模型。

```powershell
./scripts/windows/build.ps1 -Profile AsrStageProfile `
    -ModelPath <dedicated-model.tflite> -MemoryMode Dedicated_Sram `
    -CacheSizeKiB 256 -TimingMode Simple -Package

./scripts/windows/build.ps1 -Profile AsrStageProfile `
    -ModelPath <shared-model.tflite> -MemoryMode Shared_Sram `
    -TimingMode Simple -Package
```

`-TimingMode Detailed` 切换为原有详细模式；脚本默认 Simple，不影响普通构建默认关闭计时。
默认新构建/包目录名包含模式，防止两档相互覆盖；显式指定目录时按指定目录构建。
