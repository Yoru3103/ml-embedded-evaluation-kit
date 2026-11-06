# QR OBB 模型 `.pt` → 量化 `.pt2`

本目录可脱离 Arm ML Embedded Evaluation Kit 仓库单独使用。压缩包内有 `export_pt_to_pt2.py`（导出脚本）、`requirements.txt`（环境依赖）、`default_vela.ini`（量化器配置）和本说明。

**模型范围：**当前项目的非 end-to-end QR OBB Ultralytics checkpoint，要求 15 类、8 个 DFL bins、3 个检测头。首层可以是 RGB 三通道或灰度单通道；RGB 权重会折叠成单通道卷积。输出是三个原始检测特征图和角度 logits；脚本不适用于任意 Detect/OBB 架构。

## 1. 配置环境

建议在 Linux 或 WSL 中使用 Python 3.12。解压并进入 `trainer_pt2` 目录：

```sh
cd trainer_pt2
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

依赖版本固定为已验证的 CPU 导出环境；首次安装需要访问 PyPI 和 PyTorch CPU wheel 源。模型权重与校准图片由使用者提供，不在压缩包中。

## 2. 导出命令

```sh
.venv/bin/python export_pt_to_pt2.py \
  --model /path/to/best.pt \
  --calibration-dir /path/to/calibration_images \
  --validation-dir /path/to/validation_images \
  --output-dir /path/to/pt2_output
```

校准和验证目录会递归读取 `.jpg`、`.jpeg`、`.png`、`.bmp`。两组图片建议分开。脚本把图片转换为单通道灰度并归一化为 `float32 / 255`。默认的 PT2 输入和检测特征图输出边界是 NHWC；角度输出保持原形状。

例如使用 256×256 输入和全部校准图片：

```sh
.venv/bin/python export_pt_to_pt2.py \
  --model /path/to/best.pt \
  --calibration-dir /path/to/calibration_images \
  --validation-dir /path/to/validation_images \
  --output-dir /path/to/pt2_256 \
  --image-size 256 \
  --calibration-limit 0
```

也可以运行 `.venv/bin/python export_pt_to_pt2.py --help` 查看参数。

## 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | 必填 | Ultralytics `.pt` 权重文件。 |
| `--calibration-dir` | 必填 | 代表性校准图片目录，递归读取。 |
| `--validation-dir` | 校准目录 | 验证图片目录；建议显式传入独立数据。 |
| `--output-dir` | 必填 | 输出目录；已有完整匹配的缓存会验证后复用。 |
| `--image-size` | `320` | 输入的默认宽和高；可被下面两个参数分别覆盖。 |
| `--image-width` | `--image-size` | 输入宽度，须为正数且为 32 的倍数。 |
| `--image-height` | `--image-size` | 输入高度，须为正数且为 32 的倍数。 |
| `--io-layout` | `nhwc` | PT2 边界布局，可选 `nhwc` 或 `nchw`。 |
| `--preprocessing` | `mlek-crop` | `mlek-crop`：等比放大后居中裁剪；`letterbox`：等比缩放后以灰度值 114 居中填充。 |
| `--calibration-limit` | `128` | 最多使用的校准图片数；`0` 表示全部。 |
| `--validation-limit` | `8` | 最多使用的验证图片数，必须大于 0。 |
| `--seed` | `0` | 图片打乱种子；验证集使用 `seed + 1`。 |
| `--threads` | `4` | PyTorch CPU 线程数。 |
| `--max-score-error` | `0.15` | 最大类别分数绝对误差上限，范围 `[0, 1]`。 |
| `--min-box-iou` | `0.9` | 原模型最高分 anchor 的量化前后框 IoU 下限，范围 `[0, 1]`。 |
| `--per-channel` / `--no-per-channel` | 启用 | 是否按输出通道量化权重。 |
| `--target` | `ethos-u85-1024` | Ethos-U 量化器目标型号。 |
| `--vela-config` | 包内 `default_vela.ini` | Ethos-U 编译规范使用的 Vela 配置路径。 |
| `--system-config` | `Ethos_U85_SYS_DRAM_Mid_1024` | Vela 系统配置名称。 |
| `--memory-mode` | `Dedicated_Sram` | 可选 `Dedicated_Sram`、`Shared_Sram`、`all`。PT2 阶段的 `all` 按 `Dedicated_Sram` 处理，不生成两份 PT2。 |
| `--arena-cache-size` | `393216` | 传给 Ethos-U 编译规范的 SRAM cache 字节数，须大于 0。 |
| `--extra-flag` | 无 | 附加 Ethos-U/Vela flag，可重复传入，例如 `--extra-flag=--foo`。 |

图片按路径排序，再用固定种子打乱后取指定数量。图片尺寸不合规、目录没有图片、模型结构不符合上述契约，或验证结果超过阈值时，脚本会报错退出。

## 输出文件和复用规则

| 文件 | 内容 |
| --- | --- |
| `quantized.pt2` | PT2E 量化的 `ExportedProgram`。边界输入/输出仍为浮点张量，内部为量化计算图。 |
| `pt2_report.json` | 源模型及所选图片哈希、张量契约、量化配置、灰度折叠/浮点捕获/量化验证结果，以及 `completed` 或 `failed` 状态。 |
| `references.npz` | 验证输入和原始浮点输出，供缓存重新加载时复查。 |
| `labels.txt` | 按类别 ID 排列的 15 个类别名称。 |

再次运行时，脚本检查模型和所选图片哈希、影响 PT2 的选项以及工具版本，并重新验证缓存；不匹配时请指定新的 `--output-dir`。`--memory-mode`、cache 大小等后续编译调度参数不属于 PT2 缓存身份。此脚本只导出 PT2，不编译 `.pte`，也不代表已在 FVP、FPGA 或 NPU 上验证。
