#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Export the four-output QR-code ONNX model to Ethos-U PTE and INT8 TFLite."""

import argparse
import hashlib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import sys


from model_export.common.cli import parse_args
from model_export.common.reporting import write_manifest










def make_samples(paths: list[Path], metadata: dict,
                 args: argparse.Namespace) -> list[dict]:
    """Cache reference outputs from the source ONNX model on validation images.

    :param paths:       Selected validation image paths.
    :param metadata:    Checked input/output interface.
    :param args:        Source model and preprocessing options.
    :returns:           Inputs and all ONNX reference outputs in graph order.
    """
    ort = import_module("onnxruntime")
    load_image = import_module("model_export.common.images").load_image
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    graph = import_module("model_export.qrcode.contract").load_model(args.model)
    session = ort.InferenceSession(graph.SerializeToString(), sess_options=options,
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






def compare_reference(args: argparse.Namespace, samples: list[dict], metadata: dict,
                      modes: list[str]) -> dict:
    """Validate and compile the supplied reference using identical input preparation.

    :param args:        Reference path and export options.
    :param samples:     ONNX predictions on selected validation images.
    :param metadata:    Original tensor interface.
    :param modes:       Requested Vela memory configurations.
    :returns:           Numerical and NPU-only reference-model checks.
    """
    tflite = import_module("model_export.qrcode.tflite_utils")
    validate_outputs = import_module("model_export.qrcode.onnx_utils").validate_outputs
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
    images = import_module("model_export.common.images")
    graph, metadata = import_module("model_export.qrcode.contract").describe_model(args.model)
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
        backends = import_module("model_export.qrcode.backends")
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
            report["tflite"] = import_module("model_export.qrcode.onnx2tf").export_tflite(
                args.model, calibration, samples, metadata, args)
            report.setdefault("versions", {})["onnx2tf"] = (
                report["tflite"]["converter_environment"]["onnx2tf"])
        else:
            report["tflite"] = backends.export_tflite(model, calibration, samples, metadata, args)
        report["tflm"] = {}
        for mode in modes:
            variant = argparse.Namespace(**vars(args))
            variant.memory_mode = mode
            report["tflm"][mode] = import_module(
                "model_export.qrcode.tflite_utils").compile_tflite(variant)
    if args.format in ("both", "pte"):
        report["pte"] = {}
        backends.compile_pte_variants(
            backends.quantize_pte(model, paths, metadata, args),
            samples, metadata, args, report["pte"])


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
