# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Parse shared source, quantization, compiler and artifact options."""

import argparse
from pathlib import Path
import sys

from model_export.common.paths import ROOT

def select_tflite_backend(model: Path, backend: str) -> str:
    """Select a converter that accepts the source format.

    :param model:       Source model path.
    :param backend:     Automatic or explicit converter selection.
    :returns:           Selected TFLite converter name.
    :raises ValueError: Unsupported source or converter combination.
    """
    suffix = model.suffix.lower()
    if suffix not in (".onnx", ".pt"):
        raise ValueError(f"Unsupported model format: {suffix}")
    if backend == "auto":
        return "onnx2tf" if suffix == ".onnx" else "litert"
    if backend not in ("onnx2tf", "litert") or (suffix == ".pt" and backend == "onnx2tf"):
        raise ValueError(f"Cannot convert {suffix} with {backend}")
    return backend


def parse_args(argv: list[str] | None = None,
               source_formats: tuple[str, ...] = (".onnx",)) -> argparse.Namespace:
    """Parse source, calibration and matching NPU compiler options.

    :param argv:            Optional command-line arguments.
    :param source_formats:  Accepted source extensions.
    :returns:               Validated command-line options.
    """
    unified = source_formats != (".onnx",)
    parser = argparse.ArgumentParser(
        description="Export ONNX or Ultralytics PT to PTE/TFLite and compile with Vela."
        if unified else __doc__)
    parser.add_argument("--model", type=Path,
                        required=unified,
                        default=ROOT / "resources_downloaded/yolo_QRcode/best.onnx")
    parser.add_argument("--calibration-dir", type=Path,
                        default=ROOT / "resources/qrcode/smaples")
    parser.add_argument("--validation-dir", type=Path,
                        help="Defaults to --calibration-dir (resources/qrcode/smaples)")
    parser.add_argument("--reference-tflite", type=Path,
                        help="Optional external INT8 model for numerical and Vela comparison")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "resources_downloaded/yolo_QRcode")
    parser.add_argument("--format", "--to", choices=("both", "pte", "tflite"), default="both")
    parser.add_argument("--tflite-backend", choices=("auto", "onnx2tf", "litert"),
                        default="auto", help="auto: ONNX uses onnx2tf; PT uses LiteRT")
    parser.add_argument("--onnx2tf-backend", choices=("tf_converter", "flatbuffer_direct"),
                        default="tf_converter", help="ONNX2TF conversion engine")
    parser.add_argument("--io-layout", choices=("nhwc", "nchw"), default="nhwc",
                        help="Deployment interface; nchw retains legacy PyTorch exports")
    parser.add_argument("--output-type", choices=("int8", "float32"), default="int8",
                        help="Raw QR heads: int8 or model-side dequantized float32; "
                             "input stays int8")
    parser.add_argument("--data", type=Path, help="Dataset YAML for PT-to-TFLite calibration")
    parser.add_argument("--image-size", type=int, default=320, help="PT input size")
    parser.add_argument("--calibration-split", choices=("train", "val"), default="train")
    parser.add_argument("--max-score-drop", type=float, default=0.20, help="PT score-drop limit")
    parser.add_argument("--calibration-limit", type=int, default=128,
                        help="ONNX/PT-PTE images: 0 uses all; PT-TFLite uses --data")
    parser.add_argument("--validation-limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0, help="ONNX/PT-PTE image selection seed")
    parser.add_argument("--preprocessing", choices=("mlek-crop", "letterbox"),
                        default="mlek-crop", help="ONNX/PT-PTE image preparation")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-score-error", type=float, default=0.15,
                        help="Abort if any decoded score differs from ONNX by more than this")
    parser.add_argument("--min-box-iou", type=float, default=0.9,
                        help="Minimum box IoU at each image's strongest ONNX anchor")
    parser.add_argument("--per-channel", action=argparse.BooleanOptionalAction, default=True,
                        help="PTE/ONNX2TF per-channel weights; LiteRT uses its static INT8 recipe")
    parser.add_argument("--save-intermediate", action=argparse.BooleanOptionalAction, default=True)
    pt2 = parser.add_mutually_exclusive_group()
    pt2.add_argument("--pt2-only", action="store_true",
                     help="Native QR PT: calibrate/cache PT2 without compiling PTE")
    pt2.add_argument("--quantized-pt2", type=Path,
                     help="Native QR PT: compile a cached quantized.pt2 without recalibration")
    parser.add_argument("--target", default="ethos-u85-1024")
    parser.add_argument("--vela-config", type=Path, default=ROOT / "scripts/vela/default_vela.ini")
    parser.add_argument("--system-config", default="Ethos_U85_SYS_DRAM_Mid_1024")
    parser.add_argument("--memory-mode", choices=("Dedicated_Sram", "Shared_Sram", "all"),
                        default="Dedicated_Sram")
    parser.add_argument("--arena-cache-size", type=int, default=393216)
    parser.add_argument("--extra-flag", action="append", default=[])
    parser.add_argument("--memory-alignment", type=int, default=16)
    parser.add_argument("--max-planned-memory", type=int)
    args = parser.parse_args(argv)
    if args.validation_dir is None:
        args.validation_dir = args.calibration_dir
    validate_source_options(args, parser, source_formats, argv)
    validate_common_options(args, parser)
    for name in ("model", "output_dir", "calibration_dir", "vela_config", "validation_dir",
                 "reference_tflite", "data"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    return args


def validate_common_options(args: argparse.Namespace, parser: argparse.ArgumentParser):
    """Check shared resource paths, numerical limits and memory budgets.

    :param args:    Parsed command-line options.
    :param parser:  Parser reporting invalid settings.
    """
    if (args.pt2_only or args.quantized_pt2 is not None) and (
            args.model.suffix.lower() != ".pt" or args.format != "pte"):
        parser.error("PT2 options require a PT source and --to pte")
    if not args.vela_config.is_file() or not args.calibration_dir.is_dir():
        parser.error("Vela configuration and calibration directory must exist")
    if args.validation_dir is not None and not args.validation_dir.is_dir():
        parser.error("--validation-dir must name an existing directory")
    if args.reference_tflite is not None and not args.reference_tflite.is_file():
        parser.error("--reference-tflite must name an existing file")
    if not 0 <= args.max_score_error <= 1 or not 0 <= args.min_box_iou <= 1:
        parser.error("Score error and box IoU limits must be in [0, 1]")
    if args.calibration_limit < 0 or args.validation_limit <= 0 or args.threads <= 0:
        parser.error("Image limits and threads are invalid")
    if args.arena_cache_size <= 0 or (args.max_planned_memory is not None
                                     and args.max_planned_memory <= 0):
        parser.error("Memory budgets must be positive")
    if args.memory_alignment < 16 or args.memory_alignment & (args.memory_alignment - 1):
        parser.error("--memory-alignment must be a power of two >= 16")


def validate_source_options(args: argparse.Namespace, parser: argparse.ArgumentParser,
                            source_formats: tuple[str, ...], argv: list[str] | None):
    """Check source-specific settings and choose the default conversion route.

    :param args:            Parsed model and conversion options.
    :param parser:          Parser reporting invalid settings.
    :param source_formats:  Accepted model extensions.
    :param argv:            Explicit arguments or the process command line.
    """
    if source_formats != (".onnx",) and (argv is None or "--output-dir" not in argv):
        # Keep the dedicated QR-code CLI default; the unified CLI defaults beside its input.
        arguments = sys.argv[1:] if argv is None else argv
        if not any(item == "--output-dir" or item.startswith("--output-dir=")
                   for item in arguments):
            args.output_dir = args.model.parent / f"{args.model.stem}_export"
    if not args.model.is_file() or args.model.suffix.lower() not in source_formats:
        parser.error(f"--model must name an existing {source_formats} file")
    try:
        args.tflite_backend = select_tflite_backend(args.model, args.tflite_backend)
    except ValueError as error:
        parser.error(str(error))
    if (args.io_layout == "nchw" and args.format != "pte"
            and args.tflite_backend == "onnx2tf"):
        parser.error("ONNX2TF exports NHWC; use --io-layout nhwc or --tflite-backend litert")
    if args.model.suffix.lower() == ".pt":
        if args.image_size <= 0 or args.image_size % 32 or not 0 <= args.max_score_drop <= 1:
            parser.error("PT image size must be a positive multiple of 32 and score drop in [0,1]")
        if args.format != "pte" and (args.data is None or not args.data.is_file()):
            parser.error("PT-to-TFLite requires --data naming an existing dataset YAML")
