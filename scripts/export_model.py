#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Export ONNX or Ultralytics PT models through the existing backend APIs."""

import argparse
from importlib import import_module
import json

from export_qrcode_onnx import export_manifest, parse_args, select_tflite_backend


def run_export(args: argparse.Namespace) -> dict:
    """Dispatch source models to existing conversion APIs and compile TFLite with Vela.

    :param args:    Source, target, calibration and compiler options.
    :returns:       Export report with generated artifacts.
    """
    if args.format in ("both", "tflite"):
        args.tflite_backend = select_tflite_backend(args.model, args.tflite_backend)
    if args.model.suffix.lower() == ".onnx":
        return import_module("export_qrcode_onnx").run_export(args)
    if args.model.suffix.lower() != ".pt":
        raise ValueError(f"Unsupported model format: {args.model.suffix}")
    if args.reference_tflite is not None:
        raise ValueError("--reference-tflite is supported only for ONNX input")
    if args.format == "pte":
        result = import_module("export_qrcode_pt").try_export_pte(args)
        if result is not None:
            return result
    if getattr(args, "output_type", "int8") != "int8":
        raise ValueError("--output-type float32 requires the raw QR-code ONNX exporter "
                         "or the native OBB PT-to-PTE exporter")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "export_report.json"
    report = {"status": "started", "source": str(args.model), "format": args.format,
              "tflite_backend": args.tflite_backend, "io_layout": args.io_layout.upper(),
              "pte": {}, "tflm": {}}
    modes = (["Dedicated_Sram", "Shared_Sram"] if args.memory_mode == "all"
             else [args.memory_mode])
    with export_manifest(manifest, report):
        if args.format in ("both", "pte"):
            exporter = import_module("export_yolo_pte")
            for mode in modes:
                variant = argparse.Namespace(**vars(args))
                variant.memory_mode = mode
                variant.output = args.output_dir / f"pte_{mode}" / f"{args.model.stem}.pte"
                variant.diagnose_only = False
                report["pte"][mode] = exporter.run_export(variant)
        if args.format in ("both", "tflite"):
            module = import_module("export_yolo_tflite")
            exporter = (module.export_nhwc_models if args.io_layout == "nhwc"
                        else module.export_models)
            float_path, int8_path = exporter(
                args.model, args.data, args.image_size, args.calibration_split,
                args.output_dir / "tflite")
            report["tflite"] = {"float32": str(float_path), "int8": str(int8_path),
                                "calibration": {"data": str(args.data),
                                                "split": args.calibration_split,
                                                "preprocessing": "Ultralytics dataset loader",
                                                "selection": "entire selected dataset split"}}
            for mode in modes:
                variant = argparse.Namespace(**vars(args))
                variant.memory_mode = mode
                # Ultralytics retains float I/O, so Vela may leave boundary operators on CPU.
                report["tflm"][mode] = import_module("qrcode_tflite_utils").compile_tflite(
                    variant, int8_path, require_npu=False)
    print(json.dumps({"report": str(manifest), "status": report["status"]}), flush=True)
    return report


def main():
    """Select the source adapter and target exporter from one command-line interface."""
    run_export(parse_args(source_formats=(".onnx", ".pt")))


if __name__ == "__main__":
    main()
