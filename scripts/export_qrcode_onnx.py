#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Export the four-output QR-code ONNX model to Ethos-U PTE and INT8 TFLite."""

import argparse
import hashlib
from importlib import import_module
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import onnxruntime as ort
from qrcode_onnx_utils import adapt_model, describe_model, validate_outputs
from qrcode_tflite_utils import compile_tflite, inspect_tflite, lite_runner

ROOT = Path(__file__).resolve().parents[1]


def normalize_options(args: argparse.Namespace):
    """Resolve framework aliases and KiB cache sizes for the existing backends.

    :param args:    Parsed command-line options, updated in place.
    """
    if args.format == "tflm":
        args.format = "tflite"
    if args.cache_kib is not None:
        args.arena_cache_size = args.cache_kib * 1024


def parse_args() -> argparse.Namespace:
    """Parse source, calibration and matching NPU compiler options.

    :returns:   Validated command-line options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=ROOT / "resources_downloaded/yolo_QRcode/best.onnx")
    parser.add_argument("--calibration-dir", type=Path,
                        default=ROOT / "resources/object_detection/samples_QRcode")
    parser.add_argument("--validation-dir", type=Path)
    parser.add_argument("--reference-tflite", type=Path,
                        help="Optional external INT8 model for numerical and Vela comparison")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "resources_downloaded/yolo_QRcode")
    parser.add_argument("--format", choices=("both", "pte", "tflite", "tflm"), default="both")
    parser.add_argument("--tflite-backend", choices=("litert-torch", "onnx2tf"),
                        default="litert-torch",
                        help="TFLite converter; onnx2tf exports NHWC without PyTorch")
    parser.add_argument("--calibration-limit", type=int, default=128, help="0 uses all images")
    parser.add_argument("--validation-limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preprocessing", choices=("mlek-crop", "letterbox"), default="mlek-crop")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-score-error", type=float, default=0.15,
                        help="Abort if any decoded score differs from ONNX by more than this")
    parser.add_argument("--min-box-iou", type=float, default=0.9,
                        help="Minimum box IoU at each image's strongest ONNX anchor")
    parser.add_argument("--per-channel", action=argparse.BooleanOptionalAction, default=True,
                        help="PTE/onnx2tf weight quantization; LiteRT uses its static INT8 recipe")
    parser.add_argument("--save-intermediate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target", default="ethos-u85-1024")
    parser.add_argument("--vela-config", type=Path, default=ROOT / "scripts/vela/default_vela.ini")
    parser.add_argument("--system-config", default="Ethos_U85_SYS_DRAM_Mid_1024")
    parser.add_argument("--memory-mode", choices=("Dedicated_Sram", "Shared_Sram", "all"),
                        default="Dedicated_Sram")
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--arena-cache-size", type=int, default=393216)
    cache.add_argument("--cache-kib", type=int, help="NPU cache capacity in KiB")
    parser.add_argument("--extra-flag", action="append", default=[])
    parser.add_argument("--memory-alignment", type=int, default=16)
    parser.add_argument("--max-planned-memory", type=int)
    args = parser.parse_args()
    normalize_options(args)
    if not args.model.is_file() or args.model.suffix != ".onnx":
        parser.error("--model must name an existing .onnx file")
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
    for name in ("model", "output_dir", "calibration_dir", "vela_config", "validation_dir",
                 "reference_tflite"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    return args


def image_array(path: Path, image_size: int, args: argparse.Namespace) -> np.ndarray:
    """Prepare NCHW RGB data using the selected backend's deployment policy.

    :param path:        Source RGB image.
    :param image_size:  Square deployment resolution.
    :param args:        Converter and preprocessing options.
    :returns:           Contiguous float32 NCHW data normalized by 255.
    """
    if args.tflite_backend == "onnx2tf":
        return import_module("qrcode_onnx2tf_backend").load_image_array(
            path, image_size, args.preprocessing)
    return import_module("yolo_pte_utils").load_image(path, image_size, args.preprocessing).numpy()


def make_samples(paths: list[Path], metadata: dict,
                 args: argparse.Namespace) -> list[dict]:
    """Cache reference outputs from the source ONNX model on validation images.

    :param paths:       Selected validation image paths.
    :param metadata:    Checked input/output interface.
    :param args:        Source model and preprocessing options.
    :returns:           Inputs and all ONNX reference outputs in graph order.
    """
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(args.model), sess_options=options,
                                  providers=["CPUExecutionProvider"])
    samples = []
    for path in paths:
        inputs = image_array(path, metadata["image_size"], args)
        outputs = session.run([item["name"] for item in metadata["outputs"]],
                              {metadata["input"]["name"]: inputs})
        samples.append({"image": str(path), "input": inputs, "reference": outputs})
    return samples


def write_manifest(path: Path, report: dict):
    """Persist stage status so failed exports cannot look like completed runs.

    :param path:    JSON report destination.
    :param report:  Configuration and completed stages.
    """
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def validate_torch_model(graph: Any, samples: list[dict], report: dict) -> Any:
    """Preserve the original PyTorch adaptation and floating-point parity check.

    :param graph:   Checked ONNX graph for the existing export route.
    :param samples: ONNX input/output reference samples.
    :param report:  Export report receiving the parity results.
    :returns:       Adapted PyTorch model for PTE or LiteRT Torch export.
    """
    torch_api = import_module("torch")
    backend = import_module("qrcode_export_backends")
    example = torch_api.from_numpy(samples[0]["input"]).to(memory_format=torch_api.channels_last)
    model = adapt_model(graph, example).to(memory_format=torch_api.channels_last)
    exported = torch_api.export.export(model, (example,), strict=True)
    report["float_validation"] = validate_outputs(
        lambda inputs: backend.torch_predict(exported.module(check_guards=False), inputs),
        samples, True)
    return model


def compare_reference(args: argparse.Namespace, samples: list[dict], metadata: dict,
                      modes: list[str]) -> dict:
    """Validate and compile the supplied reference using identical input preparation.

    :param args:        Reference path and export options.
    :param samples:     ONNX predictions on selected validation images.
    :param metadata:    Original tensor interface.
    :param modes:       Requested Vela memory configurations.
    :returns:           Numerical and NPU-only reference-model checks.
    """
    predict, contract = lite_runner(args.reference_tflite, metadata, True)
    result = {"path": str(args.reference_tflite),
              "sha256": hashlib.sha256(args.reference_tflite.read_bytes()).hexdigest(),
              "tensor_contract": contract,
              "source_operators": inspect_tflite(args.reference_tflite),
              "validation": validate_outputs(predict, samples, False,
                                             args.max_score_error, args.min_box_iou),
              "compiled": {}}
    for mode in modes:
        variant = argparse.Namespace(**vars(args))
        variant.memory_mode = mode
        result["compiled"][mode] = compile_tflite(variant, args.reference_tflite, "reference")
    return result


def run_exports(args: argparse.Namespace, report: dict):
    """Check ONNX parity and export requested frameworks and memory variants.

    :param args:    Validated command-line options.
    :param report:  Mutable report populated after each completed stage.
    """
    graph, metadata = describe_model(args.model)
    args.image_size = metadata["image_size"]
    report["contract"] = metadata
    labels = args.output_dir / "labels.txt"
    if not labels.exists():
        labels.write_text("".join(f"class_{index}\n" for index in range(metadata["num_classes"])),
                          encoding="utf-8")
    report["labels_file"] = str(labels)
    image_helpers = import_module("qrcode_onnx2tf_backend" if args.tflite_backend == "onnx2tf"
                                  else "yolo_pte_utils")
    paths = image_helpers.find_images(args.calibration_dir, args.calibration_limit, args.seed)
    checks = image_helpers.find_images(args.validation_dir or args.calibration_dir,
                                       args.validation_limit, args.seed + 1)
    report["calibration_images"] = [str(path) for path in paths]
    report["validation_images"] = [str(path) for path in checks]
    report["independent_validation"] = not bool(set(paths).intersection(checks))
    samples = make_samples(checks, metadata, args)
    model = None
    if args.format in ("both", "pte") or args.tflite_backend == "litert-torch":
        model = validate_torch_model(graph, samples, report)
    modes = (["Dedicated_Sram", "Shared_Sram"] if args.memory_mode == "all"
             else [args.memory_mode])
    if args.reference_tflite is not None:
        report["reference_tflite"] = compare_reference(args, samples, metadata, modes)
    if args.format in ("both", "tflite"):
        calibration = [image_array(path, args.image_size, args) for path in paths]
        if args.tflite_backend == "onnx2tf":
            report["tflite"] = import_module("qrcode_onnx2tf_backend").export_tflite(
                graph, calibration, samples, metadata, args)
        else:
            report["tflite"] = import_module("qrcode_export_backends").export_tflite(
                model, calibration, samples, metadata, args)
        report["tflm"] = {}
        for mode in modes:
            variant = argparse.Namespace(**vars(args))
            variant.memory_mode = mode
            report["tflm"][mode] = compile_tflite(variant)
    if args.format in ("both", "pte"):
        backend = import_module("qrcode_export_backends")
        report["pte"] = {}
        for mode in modes:
            variant = argparse.Namespace(**vars(args))
            variant.memory_mode = mode
            variant.output = args.output_dir / f"pte_{mode}/best.pte"
            report["pte"][mode] = backend.export_pte(model, paths, samples, metadata, variant)


def main():
    """Run conversion and write reproducible configuration and validation metadata."""
    args = parse_args()
    os.environ["PATH"] = os.pathsep.join(
        (str(Path(sys.executable).absolute().parent), os.environ.get("PATH", "")))
    packages = ["onnx", "onnxruntime", "ethos-u-vela", "pillow"]
    if args.format in ("both", "pte") or args.tflite_backend == "litert-torch":
        import_module("torch").set_num_threads(args.threads)
        packages.extend(["torch", "onnx2torch", "executorch"])
    if args.format in ("both", "tflite"):
        packages.extend(["onnx2tf", "tensorflow", "tf-keras"] if args.tflite_backend == "onnx2tf"
                        else ["litert-torch", "ai-edge-quantizer"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "started",
              "source_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
              "configuration": {key: str(value) if isinstance(value, Path) else value
                                for key, value in vars(args).items()},
              "versions": {name: version(name) for name in packages},
              "validation_scope": "Host numerical checks; no labeled accuracy or NPU runtime test",
              "labels": "Generated class_N names are placeholders; "
                        "supply real labels for application use"}
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


if __name__ == "__main__":
    main()
