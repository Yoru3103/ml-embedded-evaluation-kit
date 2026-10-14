# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Direct ONNX2TF conversion with the existing QR-code validation contract."""

import argparse
from importlib import import_module
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from qrcode_tflite_utils import validate_tflite_models

CONVERTER_PYTHON = (Path(__file__).resolve().parents[1]
                    / "resources_downloaded/env-onnx2tf/bin/python")


def convert_onnx(options: dict) -> dict:
    """Call the ONNX2TF API in its dedicated environment when available.

    :param options:       JSON-compatible ONNX2TF API arguments.
    :returns:             Converter version and Python executable used.
    :raises RuntimeError: The isolated converter process failed.
    """
    if CONVERTER_PYTHON.is_file():
        script = (
            "import json, sys\n"
            "from pathlib import Path\n"
            "import onnx2tf\n"
            "options = json.load(sys.stdin)\n"
            "onnx2tf.convert(**options)\n"
            "Path(options['output_folder_path'], 'converter.json').write_text("
            "json.dumps({'python': sys.executable, 'onnx2tf': onnx2tf.__version__}))\n"
        )
        try:
            subprocess.run([str(CONVERTER_PYTHON), "-c", script], input=json.dumps(options),
                           text=True, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"ONNX2TF conversion exited with status {error.returncode}") from error
        return json.loads((Path(options["output_folder_path"]) / "converter.json").read_text())
    converter = import_module("onnx2tf")
    converter.convert(**options)
    return {"python": sys.executable, "onnx2tf": getattr(converter, "__version__", "unknown")}


def export_tflite(model: Path, calibration: list[np.ndarray], samples: list[dict],
                  metadata: dict, args: argparse.Namespace) -> dict:
    """Convert ONNX directly and validate floating and integer outputs against ONNX Runtime.

    :param model:       Original ONNX source, without PyTorch adaptation.
    :param calibration: Preprocessed NCHW float32 calibration tensors.
    :param samples:     Independent ONNX predictions and matching inputs.
    :param metadata:    Original input and ordered output interface.
    :param args:        Output directory and numerical acceptance limits.
    :returns:           Validated floating and fully integer TFLite artifacts.
    """
    directory = args.output_dir / "tflite"
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="onnx2tf-", dir=directory) as work:
        work = Path(work)
        calibration_path = work / "calibration.npy"
        # ONNX2TF expects data in its converted NHWC layout; normalization is already applied.
        np.save(calibration_path, np.concatenate(calibration).transpose(0, 2, 3, 1))
        try:
            environment = convert_onnx({
                "input_onnx_file_path": str(model), "output_folder_path": str(work),
                "output_integer_quantized_tflite": True, "tflite_backend": args.onnx2tf_backend,
                "custom_input_op_name_np_data_path": [
                    [metadata["input"]["name"], str(calibration_path), 0.0, 1.0]],
                "input_quant_dtype": "int8", "output_quant_dtype": "int8",
                "quant_type": "per-channel" if args.per_channel else "per-tensor",
                "non_verbose": True})
        except SystemExit as error:
            raise RuntimeError(f"ONNX2TF conversion exited with status {error.code}") from error
        report = validate_tflite_models(work / f"{model.stem}_float32.tflite",
                                        work / f"{model.stem}_full_integer_quant.tflite",
                                        samples, metadata, args)
        report.update(backend="onnx2tf", onnx2tf_backend=args.onnx2tf_backend,
                      converter_environment=environment)
        for label in ("float32", "int8"):
            destination = directory / f"best_{label}.tflite"
            shutil.copy2(report[label]["path"], destination)
            report[label]["path"] = str(destination)
        return report
