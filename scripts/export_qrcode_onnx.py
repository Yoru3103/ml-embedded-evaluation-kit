#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Export the four-output QR-code ONNX model to Ethos-U PTE and INT8 TFLite."""

import argparse
import hashlib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--validation-dir", type=Path)
    parser.add_argument("--reference-tflite", type=Path,
                        help="Optional external INT8 model for numerical and Vela comparison")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "resources_downloaded/yolo_QRcode")
    parser.add_argument("--format", "--to", choices=("both", "pte", "tflite"), default="both")
    parser.add_argument("--tflite-backend", choices=("auto", "onnx2tf", "litert"),
                        default="auto", help="auto: ONNX uses onnx2tf; PT uses LiteRT")
    parser.add_argument("--onnx2tf-backend", choices=("tf_converter", "flatbuffer_direct"),
                        default="tf_converter", help="ONNX2TF conversion engine")
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
    if args.model.suffix.lower() == ".pt":
        if args.image_size <= 0 or args.image_size % 32 or not 0 <= args.max_score_drop <= 1:
            parser.error("PT image size must be a positive multiple of 32 and score drop in [0,1]")
        if args.format != "pte" and (args.data is None or not args.data.is_file()):
            parser.error("PT-to-TFLite requires --data naming an existing dataset YAML")


def make_samples(paths: list[Path], metadata: dict,
                 args: argparse.Namespace) -> list[dict]:
    """Cache reference outputs from the source ONNX model on validation images.

    :param paths:       Selected validation image paths.
    :param metadata:    Checked input/output interface.
    :param args:        Source model and preprocessing options.
    :returns:           Inputs and all ONNX reference outputs in graph order.
    """
    ort = import_module("onnxruntime")
    load_image = import_module("yolo_pte_utils").load_image
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(args.model), sess_options=options,
                                  providers=["CPUExecutionProvider"])
    samples = []
    for path in paths:
        inputs = load_image(path, metadata["image_size"], args.preprocessing,
                            metadata["input_channels"]).numpy()
        outputs = session.run([item["name"] for item in metadata["outputs"]],
                              {metadata["input"]["name"]: inputs})
        samples.append({"image": str(path), "input": inputs, "reference": outputs,
                        "num_classes": metadata["num_classes"]})
    return samples


def write_manifest(path: Path, report: dict):
    """Persist stage status so failed exports cannot look like completed runs.

    :param path:    JSON report destination.
    :param report:  Configuration and completed stages.
    """
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def compare_reference(args: argparse.Namespace, samples: list[dict], metadata: dict,
                      modes: list[str]) -> dict:
    """Validate and compile the supplied reference using identical input preparation.

    :param args:        Reference path and export options.
    :param samples:     ONNX predictions on selected validation images.
    :param metadata:    Original tensor interface.
    :param modes:       Requested Vela memory configurations.
    :returns:           Numerical and NPU-only reference-model checks.
    """
    tflite = import_module("qrcode_tflite_utils")
    validate_outputs = import_module("qrcode_onnx_utils").validate_outputs
    predict, contract = tflite.lite_runner(args.reference_tflite, metadata, True)
    result = {"path": str(args.reference_tflite),
              "sha256": hashlib.sha256(args.reference_tflite.read_bytes()).hexdigest(),
              "tensor_contract": contract,
              "source_operators": tflite.inspect_tflite(args.reference_tflite),
              "validation": validate_outputs(predict, samples, False,
                                             args.max_score_error, args.min_box_iou),
              "compiled": {}}
    for mode in modes:
        variant = argparse.Namespace(**vars(args))
        variant.memory_mode = mode
        result["compiled"][mode] = tflite.compile_tflite(
            variant, args.reference_tflite, "reference")
    return result


def run_exports(args: argparse.Namespace, report: dict):
    """Check ONNX parity and export requested frameworks and memory variants.

    :param args:    Validated command-line options.
    :param report:  Mutable report populated after each completed stage.
    """
    images = import_module("yolo_pte_utils")
    graph, metadata = import_module("qrcode_contract").describe_model(args.model)
    args.image_size = metadata["image_size"]
    args.input_channels = metadata["input_channels"]
    report["contract"] = metadata
    labels = args.output_dir / "labels.txt"
    if not labels.exists():
        labels.write_text("".join(f"class_{index}\n" for index in range(metadata["num_classes"])),
                          encoding="utf-8")
    report["labels_file"] = str(labels)
    paths = images.find_images(args.calibration_dir, args.calibration_limit, args.seed)
    checks = images.find_images(args.validation_dir or args.calibration_dir,
                         args.validation_limit, args.seed + 1)
    report["calibration_images"] = [str(path) for path in paths]
    report["validation_images"] = [str(path) for path in checks]
    report["independent_validation"] = not bool(set(paths).intersection(checks))
    samples = make_samples(checks, metadata, args)
    model = None
    backends = None
    if args.format in ("both", "pte") or args.tflite_backend == "litert":
        backends = import_module("qrcode_export_backends")
        model, report["float_validation"] = backends.prepare_model(graph, samples)
    modes = (["Dedicated_Sram", "Shared_Sram"] if args.memory_mode == "all"
             else [args.memory_mode])
    if args.reference_tflite is not None:
        report["reference_tflite"] = compare_reference(args, samples, metadata, modes)
    if args.format in ("both", "tflite"):
        calibration = [images.load_image(path, args.image_size, args.preprocessing,
                                   args.input_channels).numpy()
                       for path in paths]
        if args.tflite_backend == "onnx2tf":
            report["tflite"] = import_module("qrcode_onnx2tf").export_tflite(
                args.model, calibration, samples, metadata, args)
            report.setdefault("versions", {})["onnx2tf"] = (
                report["tflite"]["converter_environment"]["onnx2tf"])
        else:
            report["tflite"] = backends.export_tflite(model, calibration, samples, metadata, args)
        report["tflm"] = {}
        for mode in modes:
            variant = argparse.Namespace(**vars(args))
            variant.memory_mode = mode
            report["tflm"][mode] = import_module("qrcode_tflite_utils").compile_tflite(variant)
    if args.format in ("both", "pte"):
        report["pte"] = {}
        for mode in modes:
            variant = argparse.Namespace(**vars(args))
            variant.memory_mode = mode
            variant.output = args.output_dir / f"pte_{mode}/best.pte"
            report["pte"][mode] = backends.export_pte(model, paths, samples, metadata, variant)


def run_export(args: argparse.Namespace) -> dict:
    """Convert a checked ONNX source and persist validation and compiler metadata.

    :param args:    Validated source, converter and NPU options.
    :returns:       Completed export report.
    """
    os.environ["PATH"] = os.pathsep.join(
        (str(Path(sys.executable).absolute().parent), os.environ.get("PATH", "")))
    import_module("torch").set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "started",
              "source_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
              "configuration": {key: str(value) if isinstance(value, Path) else value
                                for key, value in vars(args).items()},
              "versions": {},
              "validation_scope": "Host numerical checks; no labeled accuracy or NPU runtime test",
              "labels": "Generated class_N names are placeholders; "
                        "supply real labels for application use"}
    packages = ["torch", "onnx", "onnxruntime", "ethos-u-vela"]
    if args.format in ("both", "pte") or args.tflite_backend == "litert":
        packages += ["onnx2torch", "executorch"]
    if args.format in ("both", "tflite"):
        packages += (["onnx2tf"] if args.tflite_backend == "onnx2tf"
                     else ["litert-torch", "ai-edge-quantizer"])
    for name in packages:
        try:
            report["versions"][name] = version(name)
        except PackageNotFoundError:
            report["versions"][name] = "not installed"
    manifest = args.output_dir / "export_report.json"
    write_manifest(manifest, report)
    try:
        run_exports(args, report)
        report["status"] = "completed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        write_manifest(manifest, report)
    print(f"Export completed: {manifest}", flush=True)
    return report


def main():
    """Export the QR-code ONNX model using the selected backends."""
    run_export(parse_args())


if __name__ == "__main__":
    main()
