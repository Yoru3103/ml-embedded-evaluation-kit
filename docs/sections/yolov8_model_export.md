<!--
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
-->

# Exporting YOLO detection models

Run these commands from the repository root. Supply your own checkpoint, calibration
images and dataset metadata. Exported models belong in `resources_downloaded/`, which
is excluded from Git.

The scripts support non-end-to-end Ultralytics detection checkpoints. The PTE adapter
requires a Detect head exposing `_get_decode_boxes` and checks its floating-point
output against the original model before quantization.

## Python environment

Use a Python environment containing the project's model-validation dependencies.
The exporters have been checked with Ultralytics 8.4.129, PyTorch 2.12.0,
ExecuTorch 1.3.1 and Vela 5.1.0. TFLite export additionally requires the dependencies
used by the Ultralytics LiteRT exporter. The scripts import these packages directly;
the ordinary firmware build does not require Ultralytics.

## TFLite export and Vela compilation

```bash
python scripts/export_yolo_tflite.py \
    --weights /path/to/best.pt \
    --data /path/to/dataset.yaml \
    --imgsz 320 --calibration-split train \
    --output-dir resources_downloaded/yolo/tflite

vela resources_downloaded/yolo/tflite/best_int8.tflite \
    --accelerator-config ethos-u85-1024 \
    --config scripts/vela/default_vela.ini \
    --system-config Ethos_U85_SYS_DRAM_Mid_1024 \
    --memory-mode Dedicated_Sram --arena-cache-size 393216 \
    --output-dir resources_downloaded/yolo/tflm_Dedicated_Sram
```

The exporter produces floating-point and INT8 TFLite files. Check the filenames and
tensor types reported by your installed Ultralytics version before invoking Vela.
The INT8 calibration split defaults to `train`; the original dataset YAML is preserved.
Inspect the input layout, tensor types and output layout before firmware deployment.
The application accepts NCHW or NHWC images and uses separate postprocessors for a
single decoded detection output and three/four-output DFL heads.

## ExecuTorch PTE export

```bash
python scripts/export_yolo_pte.py \
    --model /path/to/best.pt \
    --output resources_downloaded/yolo/pte_Dedicated_Sram/best.pte \
    --calibration-dir /path/to/calibration/images \
    --validation-dir /path/to/validation/images \
    --image-size 320 --preprocessing mlek-crop \
    --target ethos-u85-1024 \
    --vela-config scripts/vela/default_vela.ini \
    --system-config Ethos_U85_SYS_DRAM_Mid_1024 \
    --memory-mode Dedicated_Sram --arena-cache-size 393216
```

The PTE input is contiguous float32 RGB NCHW, normalized by 255. The single output
is contiguous float32 `[1, 4 + C, N]`, containing normalized XYWH boxes and class
probabilities. The exported graph includes box normalization and sigmoid. The script
rejects incompatible serialized tensor types, shapes and dimension orders.

Calibration sampling is seeded. Independent validation images measure the change
between FP32 and host PT2E predictions; this check does not exercise the compiled NPU
or replace a detection accuracy evaluation. `--max-score-drop` controls rejection at
the strongest FP32 anchor. `--max-planned-memory` optionally limits ExecuTorch planned
nonconstant memory; NPU scratch and the complete firmware memory footprint are separate.

A JSON sidecar records the configuration, validation and planned buffers. Intermediate
`float.pt2` and `quantized.pt2` graphs are saved beside the output by default;
`--no-save-intermediate` disables them. `--diagnose-only` stops before PTE compilation.
Open these intermediate models in Netron to inspect operators and tensor metadata.

For Shared SRAM, change `--memory-mode` to `Shared_Sram` and choose a separate output
directory. Cache size, target and system configuration are explicit compilation
parameters and must match the firmware's NPU and memory settings.

## Firmware configuration

Set `USE_CASE_BUILD=yolov8_detection`, the framework and your deployment platform.
For MPS4 SSE-320, `FPGA_PLATFORM_SSE_320=ON` selects FPGA registers and memory maps;
`OFF` selects FVP. Override these options for your exported model:

| CMake option | Value |
| --- | --- |
| `ML_FRAMEWORK` | `tflm` or `executorch` |
| `ETHOS_U_NPU_CONFIG_ID` | Matching NPU configuration, such as `Z1024` |
| `ETHOS_U_NPU_MEMORY_MODE` | Matching `Shared_Sram` or `Dedicated_Sram` |
| `ETHOS_U_NPU_CACHE_SIZE` | Matching Dedicated cache allocation in bytes |
| `yolov8_detection_MODEL_PATH` | Exported `.tflite` or `.pte` path |
| `yolov8_detection_IMAGE_SIZE` | Exported square input size |
| `yolov8_detection_NUM_CLASSES` | Number of classes in the checkpoint |
| `yolov8_detection_LABELS_YAML_FILE` | Dataset YAML or one-label-per-line text file |
| `yolov8_detection_FILE_PATH` | Input image or sample directory |
| `yolov8_detection_ACTIVATION_BUF_SZ` | TFLM arena or ExecuTorch method pool size |

The default samples and labels are the checked-in 320-pixel, 10-class gesture example.
Default model paths are under `resources_downloaded/gesture_detection/`, in
`tflm_<memory-mode>/best_int8_vela.tflite` or `pte_<memory-mode>/best.pte`. Models are
supplied separately; custom exports should set the path and matching metadata explicitly.

`yolov8_detection_ET_METHOD_MEMORY_IN_DDR=ON` places the ExecuTorch method pool in DDR
independently of NPU scratch. Shared deployments can then reserve SRAM for the NPU
temporary pool. Configure `ML_FWK_TMP_MEM_BASE` and `ML_FWK_TMP_MEM_SIZE` for the target
memory map, and keep the display frame buffer outside that pool. Dedicated mode already
places method memory in DDR through the normal activation buffer attributes.

## Contract checks

With the export dependencies installed:

```bash
PYTHONPATH=scripts python -m unittest discover \
    -s scripts/tests -p test_yolo_pte_contract.py -v
```
