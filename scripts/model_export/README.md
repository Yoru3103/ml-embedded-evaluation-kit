# Model export tooling

For a copyable trainer-facing QR `.pt → quantized.pt2` bundle, see
[`trainer_pt2/`](trainer_pt2/README.md). It packages the native OBB adapter,
calibration, validation, cache report and pinned Linux/WSL requirements without
the PTE compiler command.

Run commands from the repository root. The supported unified entry point stays at
`scripts/export_model.py`; the implementation is grouped by responsibility below.

```text
scripts/
├── export_model.py          # Unified source/target routing
├── export_{qrcode,yolo}_*.py # Small compatibility CLI launchers
├── model_export/
│   ├── common/             # CLI, paths, images, layout, PT2E, reports
│   ├── qrcode/             # PT/ONNX adapters, PT2 cache, PTE/TFLite backends
│   ├── yolo/               # Detect adapter and float-I/O export
│   └── tools/              # Sample generation, validation, cache experiments
├── qrcode/                 # QR-code build/run shell scripts
└── tests/                  # Export and serialization regression tests
```

## One QR-code PTE pipeline

```text
Native OBB PT ─→ qrcode.pt.prepare_pte_source() ─┐
                                              ├→ qrcode.backends.quantize_pte()
Static ONNX ───→ qrcode.onnx model adaptation ──┘       │
                                                      ▼
                                     common.quantization.quantize()
                                                      │
                                           quantized GraphModule
                                                      │
                      optional save/load: quantized.pt2│
                                                      ▼
                                    qrcode.backends.compile_pte_variants()
                                                      │ deepcopy per variant
                                    qrcode.backends.compile_quantized_pte()
                                                      │
                                    qrcode.backends.lower_integer_pte()
                                                      │
                      export → ReLU boundary fix → Edge / partition / TOSA / Vela
                                                      │
                                  integer I/O → execution plan / memory planning
                                                      │
                                    contract / CPU fallback / serialized checks
                                                      ▼
                                                   best.pte
```

Direct export and cached PT2 export use the same quantizer and compiler functions.
`--memory-mode all` calibrates once and compiles independent copies for Dedicated
and Shared SRAM. A cached export loads the PT2 once and uses that same variant
compiler. INT8 output permits no CPU network kernels; FP32 output permits only
the four terminal dequantization calls while keeping the input INT8.

## Find the implementation

| Responsibility | File / functions |
| --- | --- |
| Source and target selection | `../export_model.py`, `common/cli.py` |
| Native grayscale OBB adaptation | `qrcode/pt.py`: `RawObb`, `prepare_pte_source` |
| ONNX source adaptation | `qrcode/onnx.py`, `qrcode/contract.py`, `qrcode/onnx_utils.py` |
| Layout and representative images | `common/layout.py`, `common/images.py` |
| Shared calibration and Vela compile specs | `common/quantization.py` |
| PT2 provenance, save/load and references | `qrcode/pt2_cache.py` |
| Shared QR PTE preparation and compilation | `qrcode/backends.py` |
| Conv/ReLU quantization boundary repair | `qrcode/pte_passes.py` |
| Integer/float output execution checks | `qrcode/output_io.py` |
| TFLite conversion and Vela invocation | `qrcode/onnx2tf.py`, `qrcode/tflite_utils.py` |
| Export manifests and per-model reports | `common/reporting.py` |

## Artifact locations

Choose the root with `--output-dir`. Existing model, calibration, report and build
directories are retained; reorganizing Python files does not move model data or
invalidate embedded provenance paths.

Existing locations in this workspace:

| Location | Contents |
| --- | --- |
| `resources_downloaded/yolo_QRcode/` | Original `best-bifpn.pt`, ONNX sources and earlier exported models |
| `resources_downloaded/qrcode/` | Grayscale sources, calibration datasets, validation/probe outputs and reports |
| `build-qrcode-pte-matrix/common/` | Common PT2 cache prepared by the INT8 PTE matrix |
| `build-qrcode-pte-matrix/models/<variant>/` | Exported per-capacity/per-mode PTEs and reports |
| `build-qrcode-pte-matrix/{fvp,fpga}/<variant>/` | Firmware builds for those models |
| `build-qrcode-pte-dequant-matrix/` | Corresponding FP32-output matrix, with separate outputs |
| `resources_downloaded/gesture_detection/` | Historical Detect models and calibration, kept separate from QR-code work |

The `probe`, `review` and historical model directories are evidence from previous
runs. A filename or directory name does not establish that an artifact is the
latest accepted model: use the manifest, per-model report and recorded hashes.

```text
<model-export-root>/
├── common/                         # --pt2-only --output-dir .../common
│   ├── quantized.pt2
│   ├── pt2_report.json             # Source/options/tool/image identity and hashes
│   └── references.npz              # Reference inputs and four raw outputs
└── variants/                       # --quantized-pt2 ... --output-dir .../variants
    ├── export_report.json
    ├── labels.txt
    ├── pte_Dedicated_Sram/
    │   ├── best.pte
    │   └── best.json               # Actual I/O, quantization, memory, validation
    └── pte_Shared_Sram/
        ├── best.pte
        └── best.json
```

The `common` and `variants` names above are organizational choices passed on the
CLI, not automatic extra nesting. Direct export writes its variant directories
directly under `--output-dir`; optional intermediate graphs are written beside
the corresponding output. PT2 reuse still checks the original PT and image files
for identity; changing the target, layout or calibration invalidates reuse.

Firmware build trees and FVP/board logs stay separate from export outputs. Use
`export_report.json` and each variant's `best.json` to locate the exact model and
settings. The existing QR matrix scripts under `scripts/qrcode/` retain their
configured output roots and maintain compilation records.

## Commands

The unified interface remains:

```sh
resources_downloaded/env/bin/python scripts/export_model.py --help
```

Additional tools use package execution, with `scripts` on the module search path:

```sh
PYTHONPATH=scripts resources_downloaded/env/bin/python -m model_export.tools.generate_samples --help
PYTHONPATH=scripts resources_downloaded/env/bin/python -m model_export.tools.validate_tflite --help
PYTHONPATH=scripts resources_downloaded/env/bin/python -m model_export.tools.compile_fastcache --help
bash scripts/qrcode/build_qrcode_pte_matrix.sh --help
```

The existing `resources_downloaded/qrcode/zint_samples_batch1/images/calibration`
can be shared across input sizes. To create a new shared source dataset instead:

```sh
PYTHONPATH=scripts resources_downloaded/env/bin/python -m model_export.tools.generate_samples \
    --zint /home/xx/zint/build/frontend/zint \
    --image-size 256 --seed 20261015 \
    --output-dir resources_downloaded/qrcode/size/calibration_shared
```

Use the same files from `images/calibration` for every model. The shared image
loader accepts a target `(width, height)` and applies the QR-code CMake
generator's bilinear, centered, gray-114 letterbox at calibration time. The
native QR-code PT to PTE route accepts rectangular inputs with `--image-width`
and `--image-height`. For example:

```sh
resources_downloaded/env/bin/python scripts/export_model.py \
    --model resources_downloaded/qrcode/size/trained_320x192.pt --to pte \
    --image-width 320 --image-height 192 --preprocessing letterbox \
    --calibration-dir resources_downloaded/qrcode/zint_samples_batch1/images/calibration \
    --validation-dir resources_downloaded/qrcode/zint_samples_batch1/images/validation \
    --output-dir resources_downloaded/qrcode/size/exported_320x192
```

For a four-output INT8 LiteRT model from the same rectangular QR checkpoint:

```sh
resources_downloaded/env/bin/python scripts/export_model.py \
    --model resources_downloaded/qrcode/size/trained_320x192.pt \
    --to tflite --tflite-backend litert --io-layout nhwc \
    --image-width 320 --image-height 192 --preprocessing letterbox \
    --calibration-dir resources_downloaded/qrcode/zint_samples_batch1/images/calibration \
    --calibration-limit 0 \
    --validation-dir resources_downloaded/qrcode/zint_samples_batch1/images/validation \
    --output-dir resources_downloaded/qrcode/size/litert_320x192
```

The uncompiled INT8 source for the Vela matrix is
`litert_320x192/tflite/best_int8.tflite`. The command also compiles one Vela
variant for validation. Each input size needs its own quantization and
compilation, but all three can use this same source image set.

Run regression tests with:

```sh
PYTHONPATH=scripts resources_downloaded/env/bin/python -m unittest discover -s scripts/tests -v
```

These tests include real Ethos-U fixture compilation, calibration-count checks,
direct/cached PT2 equivalence, integer and FP32 output boundaries, and serialized
I/O contracts. They do not substitute for accuracy or timing tests on the device.
