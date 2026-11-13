<!--
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
-->

# 全类别手势模型：统一导出与 CMake 构建

模型转换统一使用 `scripts/export_model.py`，固件使用 CMake。
专用 Python 部署入口及其准备、导出、构建模块已移除。
已存在的模型、校准图片和报告保留，但新导出使用独立目录，避免混入旧产物。

## 当前输入

所有命令在 WSL 的工程根目录执行：

```bash
cd /home/xx/ml-embedded-evaluation-kit
unset PYTHONHOME
source resources_downloaded/env/bin/activate
```

- 模型：`resources_downloaded/gesture_detection/gesture_33.pt`。
- 类型：非 end-to-end YOLO Detect，320×320 RGB，33 类；不是图像分类模型。
- 本次训练数据：`resources_downloaded/gesture_all_source/dataset/`。
- 原始图片：`images/train` 2299 张、`images/val` 540 张、`images/test` 528 张。
- 原始 YAML 中 `path` 仍是服务器路径；PTE 命令直接指定本机图片目录，不依赖该字段。

## 量化和编译为 PTE

```bash
python scripts/export_model.py \
    --model resources_downloaded/gesture_detection/gesture_33.pt \
    --to pte \
    --output-dir resources_downloaded/gesture_detection/exported_33_fulltrain_320 \
    --calibration-dir resources_downloaded/gesture_all_source/dataset/images/train \
    --validation-dir resources_downloaded/gesture_all_source/dataset/images/val \
    --calibration-limit 0 \
    --validation-limit 540 \
    --seed 0 \
    --image-size 320 \
    --preprocessing letterbox \
    --io-layout nhwc \
    --target ethos-u85-1024 \
    --vela-config scripts/vela/default_vela.ini \
    --system-config Ethos_U85_SYS_DRAM_Mid_1024 \
    --memory-mode Dedicated_Sram \
    --arena-cache-size 393216 \
    --max-planned-memory 4194304 \
    --threads 4
```

`--calibration-limit 0` 使用目录内全部图片，递归读取 jpg/jpeg/png/bmp。
上面使用全部 2299 张 train 图片校准，全部 540 张 val 图片检查，test 保留评估。
不需要为 PTE 校准传入 YOLO 标注文件。验证是 FP32 与主机 PT2E 的数值检查，
不是 mAP，也不执行编译后的 NPU。默认 `--max-score-drop 0.20`；检查失败时先看
`pte_Dedicated_Sram/best.json`，不要将中间 `quantized.pt2` 当成成功的 `.pte`。
如果决定放宽门限，应明确记录；门限变化不改善量化质量。

原先 `gesture_detection_all/calibration` 的 280 张已预处理图片仍可使用；
切换到上面的原始 train 目录即可扩大校准集。上述 letterbox 是本次建议配置，
旧 280 张图片使用 mlek-crop，两种结果不能只归因于校准数量变化。
若只比较校准数量，应保持原有 `--preprocessing mlek-crop` 和固定验证集。
更多数据应补充不同人员、手形、距离、光照、背景及无手势场景，验证集保持独立。
模型看到的是输入像素，预处理还需与实际 VSI 相机路径核对。

成功产物：

```text
resources_downloaded/gesture_detection/exported_33_fulltrain_320/
  export_report.json
  labels.txt                         # 下方从 checkpoint 生成
  pte_Dedicated_Sram/
    best.pte
    best.json
    best_graphs/{float,quantized}.pt2
```

普通 Detect 路线内部进行 INT8 量化，输入/输出接口仍为 FP32：
输入 `[1,320,320,3]` RGB /255，输出 `[1,37,2100]` 为归一化 XYWH 与类别概率。
不要在固件重复 sigmoid。导出失败时旧 `.pte` 可能仍存在，应核对报告状态和生成时间。

## 按 checkpoint 顺序生成标签

使用模型本身的类别顺序，不能使用旧十类 `resources/gesture_detection/labels.txt`：

```bash
python - <<'PY'
from pathlib import Path
from ultralytics import YOLO
model = YOLO('resources_downloaded/gesture_detection/gesture_33.pt')
output = Path('resources_downloaded/gesture_detection/exported_33_fulltrain_320')
output.mkdir(parents=True, exist_ok=True)
(output / 'labels.txt').write_text(
    '\n'.join(model.names[i] for i in range(len(model.names))) + '\n', encoding='utf-8')
print('classes:', len(model.names))
PY
```

## 构建 VSI / FVP 的 ExecuTorch 固件

先确认本次导出成功，且 `best.json` 的 `planned_total_bytes` 不超过下方 4 MiB
method pool。该 pool 不等于整机 RAM；NPU scratch 单独分配。

```bash
cmake -S . -B build-gesture33-pte-vsi \
    -DCMAKE_TOOLCHAIN_FILE="$PWD/scripts/cmake/toolchains/bare-metal-gcc.cmake" \
    -DMLEK_PYTHON_VENV="$PWD/resources_downloaded/env" \
    -DTARGET_PLATFORM=mps4 \
    -DTARGET_SUBSYSTEM=sse-320 \
    -DFPGA_PLATFORM_SSE_320=OFF \
    -DML_FRAMEWORK=executorch \
    -DUSE_CASE_BUILD=yolov8_detection \
    -DUSE_SINGLE_INPUT=ON \
    -DETHOS_U_NPU_ENABLED=ON \
    -DETHOS_U_NPU_ID=U85 \
    -DETHOS_U_NPU_CONFIG_ID=Z1024 \
    -DETHOS_U_NPU_MEMORY_MODE=Dedicated_Sram \
    -DETHOS_U_NPU_CACHE_SIZE=393216 \
    -DETHOS_U_NPU_TIMING_ADAPTER_ENABLED=ON \
    -DTA_CONFIG_FILE="$PWD/scripts/cmake/timing_adapter/ta_config_u85_sys_dram_mid.cmake" \
    -DFVP_VSI_ENABLED=ON \
    -DBUILD_FVP_TESTS=OFF \
    -DINTERACTIVE_MODE=OFF \
    -DSEMIHOSTING_ENABLED=OFF \
    -Dyolov8_detection_CAMERA_INPUT=ON \
    -Dyolov8_detection_BUTTON_CAPTURE=ON \
    -Dyolov8_detection_MODEL_PATH="$PWD/resources_downloaded/gesture_detection/exported_33_fulltrain_320/pte_Dedicated_Sram/best.pte" \
    -Dyolov8_detection_LABELS_YAML_FILE="$PWD/resources_downloaded/gesture_detection/exported_33_fulltrain_320/labels.txt" \
    -Dyolov8_detection_FILE_PATH="$PWD/resources_downloaded/gesture_all_source/dataset/images/val" \
    -Dyolov8_detection_IMAGE_SIZE=320 \
    -Dyolov8_detection_NUM_CLASSES=33 \
    -Dyolov8_detection_INFER_LAYOUT_FROM_SHAPE=ON \
    -Dyolov8_detection_BOX_COORDINATES=normalized \
    -Dyolov8_detection_INPUT_IS_GRAYSCALE=OFF \
    -Dyolov8_detection_OBB=OFF \
    -Dyolov8_detection_SCORE_THRESHOLD=0.25 \
    -Dyolov8_detection_NMS_THRESHOLD=0.45 \
    -Dyolov8_detection_MAX_DETECTIONS=20 \
    -Dyolov8_detection_DISPLAY_DOWNSCALE=2 \
    -Dyolov8_detection_ACTIVATION_BUF_SZ=0x00400000 \
    -Dyolov8_detection_ET_METHOD_MEMORY_IN_DDR=ON

cmake --build build-gesture33-pte-vsi --target mlek_yolov8_detection -j8
```

`FILE_PATH` 满足当前无条件样本代码生成；VSI 运行时使用摄像头帧。
`INFER_LAYOUT_FROM_SHAPE=ON` 对本次原生 NHWC 输入必需，避免写成 NCHW。
标签数和顺序必须与 checkpoint 一致。编号与颜色由检测显示代码绘制，默认启用。

启动 Windows VSI 摄像头服务后：

```bash
GESTURE_BUILD_DIR="$PWD/build-gesture33-pte-vsi" bash scripts/vsi/run_gesture_fvp320.sh
```

WSL 镜像网络可添加 `VSI_VIDEO_HOST=127.0.0.1`。按 KEY0 捕获并识别一帧。
