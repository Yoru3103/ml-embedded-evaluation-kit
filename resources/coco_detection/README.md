# COCO detector imported from the alif project

This profile ports the alif `best_int8` default from
`scripts/build_yolov8_fvp320.sh`, rather than the optional 640-pixel YOLOv8n model.
It reuses MLEK's `yolov8_detection` application, camera HAL, class-aware NMS and LCD
box drawing. The existing image-classification and gesture firmware are retained
in their original build directory.

The imported resources are the original `coco128.yaml` label mapping and
`20260827_Model/000000058350.jpg` reference image. The YAML retains its original
Ultralytics attribution. The image is the corresponding COCO reference sample.
Only the YAML's 80 class names are used by firmware generation.

The original model is copied to
`resources_downloaded/coco_detection/source/best_int8.tflite` and has SHA256
`f1f1f92cd1fa6f6e33c040c17006037456cf7844936d7b772890954f7f4b1f26`.
Model binaries follow the project's downloaded-resource convention and are not
tracked in Git. On a fresh checkout, provide `COCO_MODEL_SOURCE` or `ALIF_ROOT` to
the build script to import the same model.

| Setting | Value |
| --- | --- |
| Input | float32 RGB NCHW `[1,3,256,256]`, pixels divided by 255 |
| Output | float32 `[1,84,1344]`: normalized XYWH plus 80 class probabilities |
| Confidence / NMS thresholds | 0.1 / 0.45, matching the alif build script |
| Maximum detections | 20 |
| LCD downscale | 2, preserving image aspect ratio |
| TFLM arena | 2 MiB, matching the alif default |
| Current NPU profile | U85-1024, Dedicated_Sram, 384 KiB fast cache |

The original alif default uses U85-256 with Shared_Sram. This port recompiles the
same source graph with the current MLEK Vela configuration for U85-1024 and
Dedicated_Sram. Vela logs and hashes are saved in `build-coco-vsi`.
The compiled graph has a 1397760-byte scratch tensor. TFLM reuses storage across
tensor lifetimes: startup measured 1398352 arena bytes, so the original 2 MiB
reservation is sufficient. This arena metric does not include the separate
384 KiB NPU fast cache, frame buffer, heap or stack.

The profile explicitly sets `yolov8_detection_BOX_COORDINATES=normalized`, as
alif's decoder does. The reference image produces some coordinates above 2.0;
automatic magnitude-based inference would incorrectly interpret this output as
pixels. Other YOLO profiles retain the existing `auto` default.

Build from the MLEK root:

```bash
bash scripts/vsi/build_coco_fvp320.sh
```

Start the existing Windows camera preview server, then run:

```bash
VSI_VIDEO_HOST=<Windows-host-IP> bash scripts/vsi/run_coco_fvp320.sh
```

For WSL mirrored networking, use `VSI_VIDEO_HOST=127.0.0.1`. If unset, the run
script uses the default IPv4 gateway, as appropriate for ordinary WSL NAT.
The FVP board panel and LCD are enabled. Press KEY0 to capture one 256x256 RGB
frame, run detection and show boxes, class names and scores. The display remains
unchanged until the next fresh KEY0 press; holding the button does not repeat.
The Windows lower preview shows the same transmitted frame before annotations.

Validation on 2026-11-06 used the same RGB bytes for the original model's LiteRT
CPU execution and the compiled model on Corstone-320 FVP. The highest detection
was class 53 (`pizza`) in both, with matching XYWH geometry. Confidence was
0.554081 on CPU and 0.529271 on FVP; these values are not numerically identical.
FVP returned four detections: pizza, two people and a couch. This is a reference
image check, not an accuracy evaluation over the COCO dataset.

Iris-injected HAL button states verified no capture before KEY0, one request per
press, no repeats while held, KEY1 ignored, and a second request after a new KEY0
press. The LCD framebuffer remained byte-for-byte unchanged while waiting.
Detailed local reports are `build-coco-vsi/cpu_reference.json`,
`build-coco-vsi/coco-button-fvp-check.json` and the adjacent FVP log. Automated
validation did not exercise the Windows camera or physically click the GUI.
