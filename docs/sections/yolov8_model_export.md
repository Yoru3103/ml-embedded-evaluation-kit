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

For full-class gesture models, use the same unified exporter and explicit firmware
configuration described in [full gesture deployment](gesture_all_deployment.md).
Calibration and validation directories, class order and input layout are explicit;
no separate Python deployment entry point is required.

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

For the grayscale QR-code OBB checkpoint, use the unified
`export_model.py --model .../best-bifpn.pt --to pte --image-size 256 --io-layout nhwc`
entry point. It captures the four raw OBB outputs directly from PyTorch and
retains integer I/O for the `qrcode` use case. See the
[QR-code use-case guide](../../source/app/use_case/qrcode/README.md#exporting-the-bifpn-obb-checkpoint)
for calibration, output paths and the folding/parity checks. This direct path
does not require exporting an intermediate ONNX graph.

The following standalone exporter is for ordinary Detect checkpoints:

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

## QR-code ONNX export

### Unified export entry point

Use `scripts/export_model.py` to select the conversion route from the input extension
and `--to pte|tflite|both`. The entry point calls the existing exporter APIs; it does
not implement a second quantizer or NPU compiler.

| Source and target | Route |
| --- | --- |
| ONNX to PTE | ONNX2Torch, static PyTorch adaptation, shared PT2E quantization, ExecuTorch/Ethos-U |
| ONNX to TFLite (default) | ONNX2TF, full INT8 TFLite, Vela |
| ONNX to TFLite (`--tflite-backend litert`) | ONNX2Torch, LiteRT Torch, AI Edge Quantizer, Vela |
| OBB PT to PTE (`--to pte`) | Native grayscale raw heads, shared PT2E quantization, ExecuTorch/Ethos-U |
| Detect PT to PTE | Existing YOLO PT exporter, PT2E quantization, ExecuTorch/Ethos-U |
| PT to TFLite | Existing Ultralytics LiteRT exporter, INT8 quantization, Vela |

ONNX-to-PTE accepts the supported static four-output QR-code/OBB contract described
below. With `--to pte`, supported Ultralytics OBB checkpoints use the native grayscale
QR-code route with 256-pixel inputs, 15 classes and 8 DFL bins. Detect checkpoints
use the existing single-output YOLO detection contract. The unified entry point
selects the source adapter; it does
not make unsupported model architectures compatible automatically. ONNX2Torch produces
an in-memory PyTorch model, so an intermediate `.pt` checkpoint is not required.

The unified and four-output ONNX exporters default to `--io-layout nhwc`. ONNX2Torch
retains NCHW computation semantics; a shared `ChannelLastIO` adapter exposes NHWC
images and feature maps to both ExecuTorch and LiteRT. Calibration tensors use the
same NHWC interface. Validation maps feature maps back to source NCHW for comparison,
while angle `[1,1,anchors]` and decoded PT outputs `[1,4+C,anchors]` remain unchanged.
The four-output ONNX reports and PT PTE reports record actual serialized shapes.
Use `--io-layout nchw` with the
PyTorch export routes to retain an older firmware interface. ONNX2TF always exports
NHWC through its own layout conversion. The standalone PT scripts retain NCHW as
their default and also accept `--io-layout nhwc`.

Firmware must use the exported model's layout: RGB NHWC data is interleaved, and
raw detection heads are indexed as `[batch,height,width,channel]`. Existing firmware
and old NCHW model artifacts require their original contract. Selecting NHWC does
not prove that all internal transposes disappear; inspect Vela's per-layer report
and measure NPU cycles on the target.

```bash
# ONNX to PTE, using ONNX2Torch and the existing shared quantization APIs.
resources_downloaded/env/bin/python scripts/export_model.py \
    --model resources_downloaded/yolo_QRcode/best.onnx --to pte \
    --calibration-dir resources/object_detection/samples_QRcode \
    --output-dir /tmp/qrcode-pte

# ONNX directly to full INT8 TFLite, then Vela, using ONNX2TF.
resources_downloaded/env/bin/python scripts/export_model.py \
    --model resources_downloaded/yolo_QRcode/best.onnx --to tflite \
    --calibration-dir resources/object_detection/samples_QRcode \
    --output-dir /tmp/qrcode-onnx2tf

# Retain the alternative ONNX2Torch/LiteRT route explicitly.
resources_downloaded/env/bin/python scripts/export_model.py \
    --model resources_downloaded/yolo_QRcode/best.onnx --to tflite \
    --tflite-backend litert --output-dir /tmp/qrcode-litert

# PT to PTE and Vela-compiled TFLite. TFLite calibration requires a dataset YAML.
resources_downloaded/env/bin/python scripts/export_model.py \
    --model resources_downloaded/gesture_detection/best.pt --to both \
    --calibration-dir resources_downloaded/gesture_detection/calibration \
    --data /path/to/dataset.yaml --image-size 320 --output-dir /tmp/gesture-export
```

`--tflite-backend auto` chooses ONNX2TF for ONNX and LiteRT for PT. ONNX2TF cannot
accept a PT checkpoint directly. Without `--output-dir`, the unified entry point
writes to `<source-directory>/<source-stem>_export/`. `--memory-mode all` compiles
both Dedicated and Shared SRAM variants. Conversion failures mark the export report
as failed; no other converter is substituted automatically.

#### Shared QR-code PTE quantization and compilation

Implementation and artifact navigation: [model export directory guide](../../scripts/model_export/README.md).
The root export scripts are compatibility CLI launchers; shared PT2E calibration lives in
`model_export/common/quantization.py`, and QR-code compilation in `model_export/qrcode/backends.py`.

Native OBB PT and adapted ONNX models use `model_export.qrcode.backends.quantize_pte()`
to select input channels, adapt NCHW/NHWC interfaces and call the shared PT2E
quantizer. Direct exports and `--pt2-only` use this same preparation path.

- Direct export quantizes once, then calls `compile_pte_variants()` to compile
  the requested memory modes.
- `--pt2-only` saves the calibrated graph as `quantized.pt2`, together with
  `references.npz` and `pt2_report.json`, without compiling a PTE. These cache
  options apply to the native OBB PT route.
- `--quantized-pt2 /path/to/quantized.pt2` validates and loads that cache, then
  calls the same `compile_pte_variants()` without recalibrating.

Each memory variant compiles an independent copy of the calibrated graph through
`compile_quantized_pte()`, so compiler transformations cannot affect later variants.
With `--memory-mode all`, calibration runs once and both SRAM modes use the same
quantization result. Intermediate graphs from direct exports are saved under the
first memory mode (`Dedicated_Sram` when exporting both modes).

The PTE matrix scripts prepare the common PT2 cache once and reuse it for all
cache capacities. INT8 and FP32 output configurations can share this cache:
`--output-type` controls the compiled output boundary, not calibration.

ONNX2TF is an optional dependency, loaded only when its route is selected. Its
default engine here is `--onnx2tf-backend tf_converter`, which reads ONNX directly
through TensorFlow and requires `onnx2tf[tensorflow]`. The alternative
`--onnx2tf-backend flatbuffer_direct` avoids the TensorFlow conversion engine.
Choose the engine explicitly when testing compatibility; ONNX2TF 2.6.9's direct
engine failed to allocate tensors for the supplied RGB model's SE multiplication.
Both engines bypass ONNX2Torch and share the same TFLite validation and Vela APIs.
Install converter dependencies in a dedicated environment because their pinned
NumPy/LiteRT versions conflict with the shared ML environment:

```bash
resources_downloaded/env/bin/python -m venv resources_downloaded/env-onnx2tf
resources_downloaded/env-onnx2tf/bin/python -m pip install \
    -r scripts/py/requirements-onnx2tf.txt
```

The unified entry point automatically calls the ONNX2TF API with this interpreter
when it exists. Continue running `export_model.py` with `resources_downloaded/env/bin/python`;
validation, ExecuTorch and Vela use the shared environment. The TFLite export report
records the converter's actual Python executable and ONNX2TF version. If the dedicated
environment is absent, the exporter uses ONNX2TF installed in the current interpreter.

The existing PT-to-PTE API preserves floating-point I/O. PT-to-LiteRT also preserves
floating-point I/O around its INT8 graph, so Vela can retain CPU
boundary operators. The unified report records actual CPU/NPU operator counts for
that path. PT-to-TFLite calibration uses the complete `--data` dataset split selected
by `--calibration-split`, with Ultralytics' own preprocessing. `--calibration-dir`,
`--calibration-limit`, `--seed` and `--preprocessing` control ONNX exports and PT-to-PTE;
they do not change the PT-to-TFLite dataset loader. This policy is recorded in the
unified report. The four-output ONNX paths retain their stricter all-NPU, INT8-interface
checks. Host numerical validation does not execute the compiled NPU program.

### Four-output ONNX export

`export_qrcode_onnx.py` handles the fixed four-output model at
`resources_downloaded/yolo_QRcode/best.onnx` directly; no `.pt` checkpoint is needed.
Install the extra conversion dependencies into the existing project environment:

```bash
resources_downloaded/env/bin/python -m pip install \
    onnx==1.19.1 onnxruntime==1.31.0 onnx2torch==1.5.15 \
    litert-torch==0.9.4 ai-edge-litert==2.2.0 ai-edge-quantizer==0.9.0

resources_downloaded/env/bin/python scripts/export_qrcode_onnx.py --memory-mode all \
    --tflite-backend litert \
    --reference-tflite \
    /home/xx/alif_ml-embedded-evaluation-kit/20260827_Model/best_int8_new.tflite
```

The default calibration directory is `resources/qrcode/smaples/` (the existing
directory spelling). Validation defaults to that same selected calibration directory;
use `--validation-dir` to select an independent dataset.
The same `/255` preparation is used by both frameworks. Input channels are read from
the source graph: RGB stays RGB, while one-channel input uses Pillow luminance after
resizing and cropping. `--preprocessing mlek-crop` uses the same centered crop as
generated firmware samples; choose `letterbox` only when the deployment uses that
policy too. `--calibration-dir`, `--validation-dir`, limits, seed, output directory,
NPU target, Vela configuration, cache and memory budgets are configurable.
`--format pte` or `--format tflite` exports only the selected framework.
The default memory mode is `Dedicated_Sram`; `--memory-mode all` also exports Shared SRAM.
`--reference-tflite` is optional. The example reference uses the same ONNX source as
the QR-code model. Other files named `best_int8.tflite` or `yolov8n_int8.tflite` in that
external directory have a different detection interface and are not QR-code references.

The supplied ONNX input is float32 `[1,3,256,256]`. Its outputs, in source graph order,
are `[1,79,32,32]`, `[1,79,16,16]`, `[1,79,8,8]` and `[1,1,1344]`. The first three heads
contain 64 raw DFL values (16 bins per distance) and 15 class logits; the last output
contains **angle logits**, one per anchor. Confidence is `sigmoid(class_logits)`.
Decode the angle as `(sigmoid(angle_logits) - 0.25) * pi`; it does not scale confidence.
The graph contract reader checks terminal box/class/angle convolution weights rather
than guessing class counts from the total head channels. The grayscale reference at
`best_bifpn_gray_qop_int8_boardcal_20260819.onnx` has `[1,1,256,256]` input and 47-channel
heads: 32 DFL values (8 bins per distance) plus the same 15 class logits. Its QOperator
graph can be inspected and run with ONNX Runtime; that does not establish support for
converting QOperator operations through onnx2torch. ONNX2TF 2.6.9 also has unresolved
compatibility problems for this grayscale QOperator model: its TensorFlow backend
misinterprets intermediate layouts and quantization semantics, and the direct backend
generates incompatible intermediate shapes. Successful conversion of `best.onnx`
does not establish support for this already quantized model. Keep numerical checks
enabled; neither a generated file nor a completed converter process proves equivalence.
The ONNX2TF exporter rejects QLinear graphs before conversion to preserve existing
boardcal quantization parameters. It does not reconstruct floating weights or recalibrate
these models. Native INT8 experiments also failed the existing numerical acceptance
limits, so no validated grayscale Dedicated SRAM 384 KiB artifact is available yet.

The model reader restores missing `com.microsoft` domains for QLinearAdd/Mul/Concat
on an in-memory copy. Reference inference uses this same normalized graph. ONNX2TF
receives a staged copy and its in-place simplifier/name-generation passes are disabled,
so conversion cannot rewrite the original ONNX file. Detailed converter output is
retained at `<output-dir>/tflite/onnx2tf.log`, including when conversion fails.
Four-output deployment models use **INT8 input and all four INT8 outputs**, preserving
output order with NHWC images and feature maps by default. Previously exported RGB
models used the original NCHW interface. Quantize input with
`clip(round(rgb / 255 / scale) + zero_point, -128, 127)` and dequantize each output with
`(value - zero_point) * scale`. Each output has its own parameters; read them from the
report. The reference TFLite has NHWC heads in a different output order, which the
validation adapter maps back to the ONNX interface.
DFL decoding rotates the distance-derived center offset, retaining XYWH and angle for
class-aware probabilistic IoU NMS. `DetectionResult` stores only center X/Y, width,
height and angle, alongside score and class ID. Angle zero also represents ordinary
rectangles. The application rotates four corners and draws the four edges on the LCD,
clipping each edge to the displayed image and the screen.
DFL decoding, sigmoid and NMS remain application postprocessing. The zero-CPU-operator
requirement applies to the compiled neural-network graph. Do not use the single-output
`.pt` export contract for this model.

The adapter specializes fixed Resize, Slice and Reshape parameters, uses Linear for
constant MatMul weights and folds SE Expand into shape-checked multiplication.
ReLU is expressed as an equivalent clamp so the Arm PT2E quantizer inserts quantization
boundaries around convolutions even when concatenation branches share output parameters.
This avoids four Conv/ReLU pairs falling back to CPU when the shared zero point differs
from the quantized minimum.
Unsupported dynamic shapes or different detection interfaces fail explicitly.
ONNX Runtime reference predictions are compared against FP32 TFLite and, when the
ONNX2Torch route is selected, the captured PyTorch graph on every validation image.
A difference exceeding `rtol=1e-4`,
`atol=1e-4` aborts conversion. Host PT2E and INT8 LiteRT outputs are dequantized and
compared separately. Validation decodes oriented DFL boxes and computes
`sigmoid(class_logits)`, matching the raw-head postprocessor. It records the strongest
source and quantized class IDs; box IoU measures the display envelopes.
The default acceptance checks require the maximum absolute score difference to be at
most 0.15 and the box IoU at the strongest ONNX anchor to be at least 0.9. Configure
these limits with `--max-score-error` and `--min-box-iou`. Reports retain per-output
raw errors and per-image score/box metrics, including those for the optional reference.
These numerical checks do not measure labeled detection accuracy.
Reports created before the angle correction used incorrect confidence semantics;
rerun validation to obtain current score metrics. See
[QR-code processing and verification](qrcode_processing_optimization.md#验证与复现) for the code locations and
real-model C++ verification command.

Outputs are under `resources_downloaded/yolo_QRcode/`:

| Artifact | Path |
| --- | --- |
| FP32 and INT8 LiteRT | `tflite/best_float32.tflite`, `tflite/best_int8.tflite` |
| Vela-compiled TFLite | `tflm_<memory-mode>/best_int8_vela.tflite` |
| Ethos-U ExecuTorch | `pte_<memory-mode>/best.pte` |
| Intermediate PyTorch graphs | `pte_<memory-mode>/best_graphs/{float,quantized}.pt2` |
| Export configuration and numerical checks | `export_report.json` |
| PTE layouts, CPU operators and memory | `pte_<memory-mode>/best.json` |
| Optional compiled reference | `reference_<memory-mode>/best_int8_new_vela.tflite` |
| Vela logs and performance estimates | `tflm_<memory-mode>/vela.log`, `*.csv` |

The report includes the source SHA-256, package versions, calibration paths, original
output names, compiler arguments and stage status. PTE serialized shapes and dimension
orders must match the selected deployment layout, with INT8 tensors and only Ethos-U
delegates. The original ONNX contract stays NCHW in the report. The script checks
the serialized forward execution plan and rejects **any CPU operator**. PTE integer
I/O parameters are stored as configuration methods and in the JSON sidecar.
Vela-compiled TFLite must contain only `ethos-u` custom operators and INT8 I/O. The check
walks the actual operators in every subgraph, not unused entries in the opcode table.
A failed run
sets report status to `failed`; files left from earlier runs must not be assumed current.
`--no-save-intermediate` suppresses the `.pt2` archives.

The six supplied images allow calibration and numerical smoke checks. By default they
are also reused for validation, which is recorded in the report. Use a larger,
representative calibration set and independent labeled validation data before drawing
accuracy conclusions. Host checks do not execute the compiled NPU program.

ONNX has no class-name metadata. The exporter creates `labels.txt` with 15 `class_N`
placeholders when it does not already exist. Replace them with the actual training
labels for meaningful detection names, preserving class order.

This script stops after model conversion, Vela compilation and host validation.
The ExecuTorch tensor wrapper recognizes ordinary INT8/UINT8 tensors and the model
loader reads each tensor's scale and zero point from PTE configuration methods
(`input0_scale`, `input0_zp`, `output0_scale`, etc.; method-prefixed names also work).
Missing or invalid quantization parameters cause initialization to fail explicitly.
Floating interfaces continue to use the existing preprocessing.

Build and run the baseline firmware after model validation:

```bash
bash scripts/qrcode/build_qrcode_sse320.sh all all
bash scripts/qrcode/run_qrcode_fvp320.sh all
```

The builder generates eight separate directories for FVP/FPGA, PTE/TFLite and both
SRAM modes. Settings match the compiled U85 Z1024 / TA Mid models. The default input
is `samples_QRcode/1.png`, which produces a detection at the default score threshold.
`QRCODE_SAMPLE_PATH`, `QRCODE_LABELS_FILE`, `QRCODE_MODEL_ROOT` and `BUILD_JOBS` override
the input, labels, model directory and compiler parallelism. Shared PTE scratch uses
the 4 MiB SRAM, while its method pool and display frame buffer are placed in DDR.
The FVP runner uses the vendor runtime in a subshell and writes each project's
`fvp.log`. Set `FVP_PATH` and `FVP_TIMEOUT` when needed.

Dedicated SRAM fast-cache comparison includes **64, 128, 256, 384 and 768 KiB**:

```bash
PYTHONPATH=scripts resources_downloaded/env/bin/python -m model_export.tools.compile_fastcache
bash scripts/qrcode/build_qrcode_fastcache_sse320.sh all all all
bash scripts/qrcode/run_qrcode_fastcache_fvp320.sh all all
# One case:
bash scripts/qrcode/build_qrcode_fastcache_sse320.sh fpga pte 256
```

The compiler reuses the validated INT8 TFLite and calibrated PT2E graphs, keeping
calibration fixed while varying Vela's cache budget. It retains the original model
outputs and rejects CPU operators. Models are saved under
`resources_downloaded/yolo_QRcode/fastcache_dedicated/<tflm|pte>/<size>k/`, with
`compilation.json` containing status, hashes, cache size and NPU-only checks.
`--cache-kib 256` and `--format tflm` or `pte` select individual compiler cases.
The validated PT2E archive requires the initial export's default `--save-intermediate`.

Each firmware cache size matches its corresponding Vela model and is recorded in
CMake configuration. All 20 Dedicated variants have independent
`build-qrcode-fastcache-<fvp|fpga>-<tflm|pte>-<size>k-z1024-ta-mid/` projects. Builders
verify compilation status and model/configuration hashes before starting each case.
`QRCODE_FASTCACHE_ROOT` selects an alternate matrix model directory for building.
FPGA images are in `bin/sectors/yolov8_detection/{boot,bram,ddr}.bin`.
Per-project `configure.log`, `build.log` and `fvp.log` retain build and runtime results.
Use NPU PMU counts for FVP comparisons; measure hardware latency on the FPGA.

FVP measurements on 2026-10-09 used U85 Z1024, TA Mid, Dedicated SRAM,
Vela Performance/COP1 and the same `samples_QRcode/1.png` input:

| Cache (KiB) | TFLite NPU cycles | PTE NPU cycles |
|-------------|-------------------|----------------|
| 64 | 1,515,086 | 1,759,056 |
| 128 | 1,410,086 | 1,665,056 |
| 256 | 1,460,086 | 1,550,056 |
| 384 | 969,086 | 1,107,056 |
| 768 | 969,086 | 1,107,056 |

All ten compiled models contain zero CPU operators and one Ethos-U delegate;
all ten FVP runs completed successfully. Each TFLite case produced class 3,
score 0.305362 and box `(78, 24, 106, 109)`; each PTE case produced class 3,
score 0.311188 and box `(79, 25, 105, 107)`. Box values are `(x, y, width, height)`
in model-input coordinates. For this model, increasing the cache from 384 to
768 KiB did not reduce NPU cycles; the TFLite 256 KiB result also shows that
latency need not decrease at every cache increment.

These are single-image FVP measurements, not FPGA timings or an independent
accuracy evaluation. `CPU TOTAL` includes the elapsed inference interval while
the CPU waits for the NPU; its presence does not indicate CPU model operators.
Local `resources_downloaded/yolo_QRcode/fastcache_performance.csv` and
`firmware_test_report.json` retain the measured cycles, model hashes, build
configuration checks and firmware paths.

## Contract checks

With the export dependencies installed:

```bash
PYTHONPATH=scripts python -m unittest discover \
    -s scripts/tests -p test_yolo_pte_contract.py -v

PYTHONPATH=scripts python -m unittest discover \
    -s scripts/tests -p test_qrcode_onnx_export.py -v
```

Windows 构建和已验证的灰度模型上板流程见 [二维码复现指南](qrcode_reproduction.md)、[Windows TFLM](windows_armds.md) 和 [Windows ExecuTorch](windows_executorch.md)。
