# 中文唤醒词识别：WeKWS DS-TCN / PTE

`kws_zh` 在 Cortex-M 上计算 Kaldi Fbank，通过 ExecuTorch 调用 Ethos-U85，
从逐帧 Sigmoid 分数中检测“Hi 小问”和“你好问问”。默认运行三个内置 PCM 音频样本。

验证状态：Shared 已通过开发板执行；Dedicated 现改用 256 KiB 编译预算，
普通 scratch 放在 DDR，专用 fast scratch 放在 SRAM。前端和 FVP 数值验证见下文。
板上错误状态、地址错配原因和256 KiB配置见
[Dedicated 256 KiB修复记录](kws_zh_dedicated256.md)。

## 处理流程

```text
16 kHz / mono / PCM16
  → 31920样本固定窗口，步长16000样本
  → 400样本分帧，帧移160样本
  → 每帧去直流、0.97预加重、Povey窗
  → 512点FFT、功率谱、40维Mel滤波、自然对数
  → float32 [1,198,40]
  → PTE：CMVN、DS-TCN、逐帧Linear与Sigmoid
  → float32 [1,198,2]
  → 连续帧确认、重叠去重、每词冷却
  → UART输出WAKE事件
```

Fbank使用原始PCM16幅度、20–8000 Hz Mel范围、dither=0、snip_edges=true，
Povey窗采用400点对称Hann窗的0.85次幂。Mel功率的下限为float32 epsilon。
短音频和最后一个窗口在波形域补零。训练时的CMVN保留在模型内部。

输出标签顺序固定为 `Hi_Xiaowen`、`Nihao_Wenwen`。
两个输出分别表示各自关键词的Sigmoid分数，后处理直接使用这些分数。

## 时序与唤醒判定

每个窗口独立推理，模型内部因果卷积历史在该窗口内重建。窗口重叠提供上下文。

- 默认阈值0.5，连续3帧达到阈值后触发。
- 每个关键词独立设置1000 ms冷却时间。
- 事件时间取确认帧的帧尾音频时间。
- 后处理按绝对音频位置跳过重叠窗口已处理的帧。
- 只处理实际音频范围内的完整帧；短样本的最小补零由音频代码生成器完成。
- 帧时间不连续时重新开始连续帧计数，每个音频片段单独重置解码器。

UART示例：

```text
KWS_ZH peak clip=0 start=0 label=Hi_Xiaowen score=0.999512 frame=128
WAKE label=Hi_Xiaowen score=0.874083 time=1.145s
KWS_ZH peak clip=1 start=0 label=Nihao_Wenwen score=0.999512 frame=119
WAKE label=Nihao_Wenwen score=0.995592 time=1.195s
```

每个窗口输出两类峰值，Profiler分别记录Fbank、Inference和WakeWordDecode。

## 模型和输入资源

默认目录：

```text
resources_downloaded/kws_zh/
  pte_Dedicated_Sram_256k/wekws_dstcn_u85.pte
  pte_Shared_Sram/wekws_dstcn_u85.pte
  pt2e/int8.pt2                       # 数值验证用参考图
resources/kws_zh/
  labels/labels.txt
  samples/01_hi_xiaowen.wav
  samples/02_nihao_wenwen.wav
  samples/03_background.wav
  samples/manifest.json
```

两种PTE均使用U85-1024。Dedicated配置256 KiB专用缓存，Shared采用共享SRAM。
三个WAV来自Apache 2.0许可的OpenSLR87，来源和校验和见样本清单。

## Windows构建和FPGA打包

使用现有ATFE与Windows ExecuTorch环境，环境配置见
[Windows ExecuTorch构建](../sections/windows_executorch.md)。

```powershell
# 构建两种内存模式并生成固件ZIP
./scripts/windows/build-kws-zh.ps1 -Package

# 只构建Shared
./scripts/windows/build-kws-zh.ps1 -MemoryModes Shared_Sram -Package

# 更换输入样本
./scripts/windows/build-kws-zh.ps1 -InputPath C:/data/my_kws_wavs -Package
```

脚本复用 `build.ps1`、`build-executorch.ps1` 和 `package-fpga.ps1`。
构建配置为MPS4/SSE-320、FPGA、Release、PMU A、
`ta_config_u85_sys_dram_1024`。音频选择设置为 `USE_SINGLE_INPUT=OFF`。

本轮固件包：

```text
build/board-test-packages/kws_zh/pte-dedicated-256k-20261027.zip
build/board-test-packages/kws_zh/pte-shared-20261027.zip
```

包内包含 `SOFTWARE/boot.bin`、`bram.bin`、`ddr.bin`、`images.txt`、
匹配的AXF/map、PTE、构建配置、模型SHA-256和验证记录。
打包脚本逐段检查BIN与AXF的一致性。

将三段BIN复制到开发板 `SOFTWARE` 目录，将 `images.txt` 放到当前FPGA镜像目录，
如 `MB/HBI0376B/FI101`。三个加载地址依次为
`0x11000000`、`0x12000000`、`0x70000000`。

## CMake选项

| 选项 | 默认值 | 用途 |
|---|---|---|
| `kws_zh_MODEL_PATH` | 对应内存模式的默认PTE | 模型 |
| `kws_zh_FILE_PATH` | `resources/kws_zh/samples` | 单个WAV或目录 |
| `kws_zh_LABELS_TXT_FILE` | `resources/kws_zh/labels/labels.txt` | 两个标签 |
| `kws_zh_ACTIVATION_BUF_SZ` | `0x00100000` | ExecuTorch方法/规划内存池 |
| `kws_zh_SCORE_THRESHOLD` | `0.5` | Sigmoid阈值 |
| `kws_zh_MINIMUM_FRAMES` | `3` | 连续确认帧数 |
| `kws_zh_COOLDOWN_MS` | `1000` | 每词冷却时间 |
| `kws_zh_WINDOW_STRIDE` | `16000` | 波形窗口步长，按160样本对齐 |
| `kws_zh_DUMP_OUTPUT` | `OFF` | 打印198×2逐帧分数 |

包装脚本使用1 MiB方法内存池和3 MiB临时内存区。
Dedicated 临时区从 DDR `0x72000000` 开始，Shared 临时区从 SRAM `0x31100000` 开始。
Dedicated 的256 KiB专用缓存位于 `0x31000000`，与普通 scratch 分开。
`-CacheSizeKiB` 同时选择缓存容量和对应容量目录中的已编译模型，默认256。
模型、音频和内存模式由通用构建脚本记录于 `build-info.json`。

## 前后处理测试

```bash
cmake -S tests/processing/kws_zh -B build/kws_zh/processing-native
cmake --build build/kws_zh/processing-native -j4
ctest --test-dir build/kws_zh/processing-native --output-on-failure

python tests/processing/kws_zh/compare_fbank.py \
  --executable build/kws_zh/processing-native/kws_zh_processing_tests \
  --samples resources/kws_zh/samples \
  --output build/kws_zh/frontend-validation
```

数值比较脚本使用PyTorch、torchaudio和NumPy。
处理逻辑测试覆盖静音下限、补零、连续帧确认、冷却、NaN及重叠窗口去重。
WSL原生测试编译启用AddressSanitizer和UndefinedBehaviorSanitizer。

本轮三个样本的C++/Kaldi Fbank平均绝对误差分别为
`9.28e-6`、`9.79e-6`、`4.74e-5`，最大绝对误差分别为
`0.000708`、`0.000654`、`0.011987`。
构建生成的PCM数组与原始WAV逐样本一致，短样本尾部补零。

## FVP与PTE参考对齐

```powershell
./scripts/windows/build-kws-zh.ps1 -Target FVP -DumpOutput
```

在Corstone-320 FVP中加载对应 `bin/mlek_kws_zh.axf`，设置U85为1024 MAC，
将UART0保存为 `fvp.log`。运行库环境和参数可参考
[项目FVP运行脚本](../../scripts/qrcode/run_qrcode_fvp320.sh)。

```bash
python tests/processing/kws_zh/compare_fvp.py \
  --program resources_downloaded/kws_zh/pt2e/int8.pt2 \
  --samples resources/kws_zh/samples \
  --log build/kws_zh/pte-shared-fvp-20261027/fvp.log \
  --output build/kws_zh/pte-shared-fvp-20261027/fvp_comparison.json
```

默认三个样本对应5个推理窗口、1980个输出分数。
Dedicated 256 KiB和Shared FVP与PT2E参考的最大逐帧误差均为 `4.66e-10`，两条正样本分别触发对应唤醒词，
背景样本的三个窗口均保持未触发。

## 源码位置

- [KaldiFbank](../../source/lib/mlek/use_case/kws_zh/KaldiFbank.cc)：平台无关前端。
- [WakeWordDecoder](../../source/lib/mlek/use_case/kws_zh/WakeWordDecoder.cc)：平台无关后处理。
- [MainLoop](../../source/app/use_case/kws_zh/src/MainLoop.cc)：HAL音频、EtModel、Profiler与UART。
- [usecase.cmake](../../source/app/use_case/kws_zh/usecase.cmake)：模型/音频/标签生成和PTE算子库。
- [Windows包装脚本](../../scripts/windows/build-kws-zh.ps1)：构建和固件打包。
