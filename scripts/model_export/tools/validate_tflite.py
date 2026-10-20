#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Recheck saved QR-code TFLite models without recalibration or Vela compilation."""

import argparse
import hashlib
from importlib import import_module
import json
from pathlib import Path

from model_export.qrcode.onnx import make_samples
from model_export.common.reporting import write_manifest
from model_export.qrcode.contract import describe_model
from model_export.qrcode.onnx_utils import validate_outputs
from model_export.qrcode.tflite_utils import check_tflite_layout, lite_runner


def saved_export(directory: Path) -> tuple[argparse.Namespace, list[Path]]:
    """Load the source configuration and exact validation image list from the failed run.

    :param directory:   Existing export directory containing export_report.json.
    :returns:           Original options and original validation image paths.
    :raises ValueError: The source model changed or validation images were not recorded.
    """
    original = json.loads((directory / "export_report.json").read_text(encoding="utf-8"))
    args = argparse.Namespace(**original["configuration"])
    args.model = Path(args.model)
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != original["source_sha256"]:
        raise ValueError("Source ONNX changed since conversion; export again before validating")
    paths = [Path(name) for name in original.get("validation_images", [])]
    if not paths:
        raise ValueError("The original export did not record validation images")
    return args, paths


def model_paths(directory: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    """Select retained converter artifacts, including models rejected by numerical checks.

    :param directory:   Existing export directory.
    :param args:        Original source filename and converter backend.
    :returns:           Floating and integer TFLite paths.
    """
    if args.tflite_backend == "onnx2tf":
        folder = directory / "tflite/onnx2tf_outputs"
        paths = (folder / f"{args.model.stem}_float32.tflite",
                 folder / f"{args.model.stem}_full_integer_quant.tflite")
    else:
        paths = (directory / "tflite/best_float32.tflite", directory / "tflite/best_int8.tflite")
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Saved converter model is missing: {path}")
    return paths


def recheck(directory: Path) -> dict:
    """Validate both saved models and retain successful stages when a later check fails.

    :param directory:   Existing export directory.
    :returns:           Host validation report, without modifying the original export report.
    """
    report = {"status": "started", "validation_only": True,
              "validation_scope": "ONNX versus saved host TFLite; no labeled mAP or NPU test"}
    try:
        args, paths = saved_export(directory)
        _, metadata = describe_model(args.model)
        import_module("torch").set_num_threads(args.threads)
        samples = make_samples(paths, metadata, args)
        report.update(source=str(args.model), validation_images=[str(path) for path in paths])
        for label, path, integer in zip(("float32", "int8"), model_paths(directory, args),
                                        (False, True)):
            report["stage"] = label
            predict, contract = lite_runner(path, metadata, integer)
            check_tflite_layout(contract, metadata, args.io_layout)
            report[label] = {"path": str(path), "tensor_contract": contract,
                             "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "validation": validate_outputs(predict, samples, not integer,
                                                            args.max_score_error, args.min_box_iou)}
            print(f"{label}: {len(samples)} validation images passed", flush=True)
        report["status"] = "validated"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_manifest(directory / "tflite_validation.json", report)
    return report


def main():
    """Revalidate saved conversion outputs using the original run's exact options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    args = parser.parse_args()
    if not (args.export_dir / "export_report.json").is_file():
        parser.error("--export-dir must contain export_report.json from an existing export")
    recheck(args.export_dir.resolve())


if __name__ == "__main__":
    main()
