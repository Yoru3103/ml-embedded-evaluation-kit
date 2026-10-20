# Grayscale QR-code use case

This use case accepts grayscale QR-code models with dimensions divisible by 32.
The board input tensor must be NHWC `[1, height, width, 1]`. The generated sample
array holds `width × height` bytes from each grayscale BMP after the configured
crop or letterbox resize; it never expands them to RGB. FP32 input
receives each byte divided by 255. INT8 or UINT8 input uses the model tensor's
scale and zero point. For scale `1/255` and zero point `-128`, INT8 preparation
is exactly `pixel - 128`.

The four outputs must contain detection heads with 47 channels at
`(height/8, width/8)`, `(height/16, width/16)` and `(height/32, width/32)`,
plus one raw angle logit per feature-map cell. The shared YOLOv8 decoder
treats 32 head channels as four 8-bin DFL distributions and the remaining 15
as class logits. Confidence comes from the best class logit. The angle head
only determines box orientation. The LCD renders the original grayscale
sample, class labels, confidence and rotated green boxes.

Select the use case with `-DUSE_CASE_BUILD=qrcode`. Supply the compiled model
with `-Dqrcode_MODEL_PATH=/absolute/path/to/model.tflite` or a PTE path when
building with ExecuTorch. The model must meet the tensor contract above; a
model from the RGB use case will fail the runtime check. The default locations
are `resources_downloaded/qrcode/tflm_<memory_mode>/best_int8_vela.tflite`
and `resources_downloaded/qrcode/pte_<memory_mode>/best.pte`.
The default resource setup does not install this custom model. An `all` build
skips this use case when its default model is absent and no model override is
provided.

`qrcode_FILE_PATH` selects a BMP or sample directory and defaults to
`resources/qrcode/smaples`. `qrcode_LABELS_FILE` defaults to the checkpoint's
13 known names followed by placeholders for class IDs 13 and 14.
Set `qrcode_IMAGE_WIDTH` and `qrcode_IMAGE_HEIGHT` to match the exported model;
`qrcode_IMAGE_RESIZE_MODE=letterbox` preserves the entire source image with
gray-114 padding. The handler checks that the generated sample dimensions match
the model input tensor and all four output shapes before inference.
The `InferenceAndPostProcessing` timer starts after input preparation and
ends after the four outputs have been decoded and NMS has finished.
The log also reports `PreProcessing`, `Inference`, `Decode`, and `NMS` separately.
See [processing optimizations](../../../../docs/sections/qrcode_processing_optimization.md)
for lookup-table formulas, timing boundaries, memory costs, and validation results.

## Exporting the BiFPN OBB checkpoint

For ExecuTorch, export directly from the original PT checkpoint. The unified
entry point recognizes this OBB head and uses native PyTorch capture and PT2E
quantization; it does not export an ONNX file or invoke the ONNX-to-PyTorch
adapter:

```bash
resources_downloaded/env/bin/python scripts/export_model.py \
    --model resources_downloaded/yolo_QRcode/best-bifpn.pt \
    --to pte --image-size 256 --io-layout nhwc \
    --calibration-dir resources_downloaded/qrcode/calibration_merged/images/calibration \
    --calibration-limit 0 \
    --validation-dir resources_downloaded/qrcode/calibration_merged/images/validation \
    --validation-limit 543 --threads 4 --preprocessing mlek-crop \
    --target ethos-u85-1024 \
    --system-config Ethos_U85_SYS_DRAM_Mid_1024 \
    --memory-mode Dedicated_Sram --arena-cache-size 393216 \
    --output-dir resources_downloaded/qrcode/direct_pt
```

The result is `resources_downloaded/qrcode/direct_pt/pte_Dedicated_Sram/best.pte`.
Supply this path as `qrcode_MODEL_PATH` and select `ML_FRAMEWORK=executorch`.
The output directory also contains the checkpoint's labels, a manifest and the
optional native `float.pt2`/`quantized.pt2` intermediate graphs. Use a separate
output directory for ONNX-based experiments.

The PT adapter still folds the first RGB convolution into a grayscale one,
retains the three 47-channel raw heads and raw angle output, and exposes NHWC
integer I/O. Before calibration it compares the folded model against the
original weights on repeated grayscale RGB input and checks native
`torch.export` capture parity. The regular score-error and box-IoU gates remain
enabled; an export failing those checks does not publish a new PTE. Direct PT
export removes an intermediate graph conversion; it does not establish detection
accuracy or remove the source model's existing classification errors.

Before Ethos-U partitioning, `scripts/model_export/qrcode/pte_passes.py` makes the calibrated
Conv/ReLU quantization boundaries explicit when the output zero point differs
from the integer minimum. This handles the C3Ghost branches whose output scales
are shared with signed concatenation inputs. It reuses their calibrated scales
and zero points; it does not change calibration or validation thresholds.
The default INT8-output PTE must contain only Ethos-U delegates and no CPU
operators. `--output-type float32` retains INT8 input and includes four terminal
output dequantizers in both PTE and TFLite. With `CPU_PROFILE_ENABLED=ON`,
`OutputDequantization` reports the sum of those four actual kernel intervals;
it is included in `Inference`. See [output dequantization and timing](../../../../docs/sections/qrcode_output_dequantization.md).
See the [quantization boundary fix](../../../../docs/sections/qrcode_processing_optimization.md#executorch-convrelu-量化边界修复)
for the graph transformation and measured checks.

The generic PT exporter handles a decoded Detect output. Use the QR-code PT
adapter below when an ONNX source is needed for TFLite conversion. The
`best-bifpn.pt` checkpoint's actual first convolution is RGB and its OBB
head contains 15 classes and 8 DFL bins:

```bash
resources_downloaded/env/bin/python scripts/export_qrcode_pt.py \
    --weights resources_downloaded/yolo_QRcode/best-bifpn.pt \
    --output resources_downloaded/qrcode/best_gray.onnx \
    --validation-dir resources/qrcode/smaples
```

The adapter sums the first convolution's RGB weights to obtain a single-channel
convolution, preserving its bias. It exposes the three raw DFL/class maps and
raw angle logits, checks them against the original model on repeated grayscale
RGB inputs, and validates the saved ONNX with ONNX Runtime. It writes an FP32
source ONNX, a JSON validation report and `best_gray.labels.txt`. Class IDs 13
and 14 retain placeholder names because this checkpoint only defines 13 names
despite having a 15-class head.

Pass that floating source to the existing four-output quantization pipeline:

```bash
resources_downloaded/env/bin/python scripts/export_model.py \
    --model resources_downloaded/qrcode/best_gray.onnx \
    --to both --tflite-backend litert --io-layout nhwc \
    --calibration-dir resources/qrcode/smaples \
    --validation-dir resources/qrcode/smaples \
    --preprocessing mlek-crop --memory-mode Dedicated_Sram \
    --output-dir resources_downloaded/qrcode
```

Use `--to pte` or `--to tflite` to export one framework. The source ONNX uses
NCHW `[1,1,256,256]` as required by the source adapter; `--io-layout nhwc`
exposes deployment input `[1,256,256,1]` and NHWC detection maps. The angle
output remains `[1,1,1344]`. This command performs new calibration, conversion
and NPU compilation; it does not preserve the older boardcal model's scales.
The firmware reads quantization parameters from the exported model.

Calibration runs the final four-output graph on grayscale `/255` FP32 inputs.
DFL decoding, class sigmoid, angle mapping and rotated NMS stay in the firmware.
Host validation decodes both floating and quantized results to compare class
scores and axis-aligned envelopes of oriented boxes. Labeled rotated IoU/mAP
evaluation is separate from calibration. The example reuses the seven checked-in
BMPs; select a separate representative calibration set and validation directory
for a deployment accuracy evaluation. The default `resources/qrcode/labels.txt`
contains the checkpoint's known class names and placeholders for IDs 13 and 14.
Override `qrcode_LABELS_FILE` when using a different checkpoint's labels.

## Generating synthetic calibration images

Generate the checkpoint's 13 named classes with the Zint CLI:

```bash
PYTHONPATH=scripts resources_downloaded/env/bin/python -m model_export.tools.generate_samples \
    --zint /home/xx/zint/build/frontend/zint \
    --per-class 30 --validation-per-class 20 --background-fraction 0.05 \
    --output-dir resources_downloaded/qrcode/zint_samples_batch1
```

With a system installation on `PATH`, omit `--zint`.
The generator refuses to overwrite an existing output directory.

Defaults produce 30 calibration and 20 validation images per class, plus
`round(positive_count * 0.05)` background images: 390 positive and 20 background
calibration BMPs, and 260 positive and 13 background validation BMPs. Every image
is eight-bit grayscale at 256 x 256. Both splits use different encoded content
and random seeds. Classes 13 and 14 have no generated examples.

| Class ID | Checkpoint name | Zint symbology |
| --- | --- | --- |
| 0 | BARCODE | CODE128, CODE39, EAN13 |
| 1 | DM | DATAMATRIX |
| 2 | HANXIN | HANXIN |
| 3 | QR | QRCODE |
| 4 | PDF | PDF417 |
| 5 | AZTEC | AZTEC |
| 6 | CODEONE | CODEONE |
| 7 | DOT | DOTCODE |
| 8 | GM | GRIDMATRIX |
| 9 | MAXI | MAXICODE, mode 4 |
| 10 | MPDF | MICROPDF417 |
| 11 | ULTRA | ULTRA, rendered in color before grayscale conversion |
| 12 | UPN | UPNQR |

Each class contains 70% scenes resembling the reference BMPs, 20% clearer
scenes and 10% mildly difficult scenes. The 256 x 256 source images guide the
gray illumination, soft-shadow and smaller-symbol ranges; their pixels are
not copied into the generated images. Source frames may have 114-valued
letterbox fill, matching the original BMPs. Each split varies symbol size,
rotation, mild perspective distortion, neutral background, brightness, blur
and sensor noise. Dense and narrow codes have minimum size limits.
Quiet zones are included;
annotations track the projected ink bounds, excluding the quiet zones.
`labels/calibration` and `labels/validation` contain matching YOLO quadrilateral
labels, with empty files for background images. The dataset also includes
`dataset.yaml`, `labels.txt`, and six preview contact sheets named
`preview_<calibration|validation>_<matched|clear|difficult>.png`. Green outlines
appear only on previews, never on calibration or validation BMPs.
`manifest.json` records payloads, geometry, lighting and shadow parameters,
the Zint version and binary checksum, and original reference image checksums.

Pass `images/calibration` to `--calibration-dir` and `images/validation` to
`--validation-dir`; do not pass the dataset root, which contains both splits
and the preview. Use `--calibration-limit 0` to include the entire calibration
split. The generator's labels describe the encoded symbology; they do not
independently confirm the supplied checkpoint's training class mapping.
Synthetic validation does not establish accuracy on real camera images.

## Using ONNX2TF

Select the TensorFlow converter explicitly for the floating four-output source:

```bash
resources_downloaded/env/bin/python scripts/export_model.py \
    --model resources_downloaded/qrcode/best_gray.onnx \
    --to tflite --tflite-backend onnx2tf --onnx2tf-backend tf_converter \
    --io-layout nhwc --calibration-dir resources/qrcode/smaples \
    --output-dir resources_downloaded/qrcode
```

The wrapper infers intermediate NCHW tensor shapes on its private ONNX copy
before conversion. These shapes let ONNX2TF convert feature-map concatenation
from channel axis 1 to NHWC channel axis 3. Changing the source input declaration
to NHWC or keeping its shape with `-kt` does not provide those intermediate
shapes. The source file remains unchanged.

The converter uses `resources_downloaded/env-onnx2tf/bin/python` when installed.
Numerical validation checks the floating model and the fully integer model
against ONNX Runtime before invoking Vela. Generated converter files are kept
under `tflite/onnx2tf_outputs/`, including when validation rejects them; their
existence alone does not indicate successful export. Check `export_report.json`
for completion. The validated artifacts retain the `best_float32.tflite` and
`best_int8.tflite` names. A quantization validation error after shape conversion
must be addressed with representative calibration data or an explicitly chosen
accuracy budget.

Floating parity uses `rtol=1e-4`, with `atol=2e-4` for detection maps and
`atol=5e-4` for raw angle logits. The latter accounts for measured CPU kernel
accumulation differences without changing the INT8 score-error or box-IoU
limits. Parity failures include the image, output index and tensor shape.

After fixing a validation issue, recheck saved converter models without
repeating calibration or invoking Vela:

```bash
PYTHONPATH=scripts resources_downloaded/env/bin/python -m model_export.tools.validate_tflite \
    --export-dir resources_downloaded/qrcode/merged_onnx2tf
```

This command verifies that the source ONNX is unchanged, uses the original
validation image list, and writes `tflite_validation.json`. It preserves the
original `export_report.json` and records a passed floating stage even if the
integer stage fails. A passed recheck is a host numerical check, not a
completed NPU export or a labeled OBB accuracy measurement.
