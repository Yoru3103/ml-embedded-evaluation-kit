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
import onnx

from qrcode_contract import load_model
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
        log_path = Path(options["output_folder_path"]) / "converter.log"
        try:
            with log_path.open("w", encoding="utf-8") as stream:
                subprocess.run([str(CONVERTER_PYTHON), "-c", script], input=json.dumps(options),
                               text=True, check=True, stdout=stream, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as error:
            details = "\n".join(log_path.read_text(encoding="utf-8", errors="replace")
                                .splitlines()[-25:])
            raise RuntimeError(
                f"ONNX2TF conversion exited with status {error.returncode}\n{details}") from error
        return json.loads((Path(options["output_folder_path"]) / "converter.json").read_text())
    converter = import_module("onnx2tf")
    converter.convert(**options)
    return {"python": sys.executable, "onnx2tf": getattr(converter, "__version__", "unknown")}


def save_converter_outputs(work: Path, directory: Path, stem: str) -> tuple[Path, Path]:
    """Retain raw converter artifacts even when numerical validation rejects them.

    :param work:        Temporary converter workspace.
    :param directory:   Export destination containing logs and validated models.
    :param stem:        Source ONNX filename without its extension.
    :returns:           Unvalidated floating and integer TFLite paths.
    """
    destination = directory / "onnx2tf_outputs"
    destination.mkdir(parents=True, exist_ok=True)
    paths = tuple(destination / f"{stem}_{suffix}.tflite"
                  for suffix in ("float32", "full_integer_quant"))
    for path in paths:
        shutil.copy2(work / path.name, path)
    return paths


def export_tflite(model: Path, calibration: list[np.ndarray], samples: list[dict],
                  metadata: dict, args: argparse.Namespace) -> dict:
    """Convert ONNX directly and validate floating and integer outputs against ONNX Runtime.

    :param model:       Original ONNX source, without PyTorch adaptation.
    :param calibration: Preprocessed NCHW float32 calibration tensors.
    :param samples:     Independent ONNX predictions and matching inputs.
    :param metadata:    Original input and ordered output interface.
    :param args:        Output directory and numerical acceptance limits.
    :returns:           Validated floating and fully integer TFLite artifacts.
    :raises ValueError: Native QOperator conversion cannot preserve boardcal parameters.
    """
    graph = load_model(model)
    if any(node.op_type.startswith("QLinear") for node in graph.graph.node):
        raise ValueError(
            "ONNX2TF QOperator conversion with preserved boardcal quantization parameters "
            "is unsupported by this exporter: native INT8 mapping has not passed layout "
            "and numerical validation. No recalibration or Vela compilation was performed.")
    # PyTorch exports may omit intermediate value_info. ONNX2TF needs those
    # NCHW shapes to map feature-map Concat axes to NHWC without guessing.
    graph = onnx.shape_inference.infer_shapes(graph, strict_mode=True)
    directory = args.output_dir / "tflite"
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="onnx2tf-", dir=directory) as work:
        work = Path(work)
        # ONNX2TF preprocessors can rewrite their input in place; always stage a private copy.
        staged = work / model.name
        onnx.save(graph, staged)
        calibration_path = work / "calibration.npy"
        # ONNX2TF expects data in its converted NHWC layout; normalization is already applied.
        np.save(calibration_path, np.concatenate(calibration).transpose(0, 2, 3, 1))
        try:
            environment = convert_onnx({
                "input_onnx_file_path": str(staged), "output_folder_path": str(work),
                "output_integer_quantized_tflite": True, "tflite_backend": args.onnx2tf_backend,
                "custom_input_op_name_np_data_path": [
                    [metadata["input"]["name"], str(calibration_path), 0.0, 1.0]],
                "input_quant_dtype": "int8", "output_quant_dtype": "int8",
                "quant_type": "per-channel" if args.per_channel else "per-tensor",
                "not_use_onnxsim": True, "not_use_opname_auto_generate": True,
                "non_verbose": True})
        except SystemExit as error:
            raise RuntimeError(f"ONNX2TF conversion exited with status {error.code}") from error
        finally:
            if (work / "converter.log").is_file():
                shutil.copy2(work / "converter.log", directory / "onnx2tf.log")
        report = validate_tflite_models(
            *save_converter_outputs(work, directory, model.stem), samples, metadata, args)
        report.update(backend="onnx2tf", onnx2tf_backend=args.onnx2tf_backend,
                      converter_environment=environment)
        if (directory / "onnx2tf.log").is_file():
            report["converter_log"] = str(directory / "onnx2tf.log")
        publish_converter_outputs(directory, model.stem, report)
        return report


def publish_converter_outputs(directory: Path, stem: str, report: dict):
    """Publish validated aliases while preserving the integer model for Vela.

    :param directory:   Export destination.
    :param stem:        Source model stem used by ONNX2TF.
    :param report:      Validated floating and quantized artifacts.
    """
    shutil.copy2(directory / f"onnx2tf_outputs/{stem}_full_integer_quant.tflite",
                 directory / "best_int8.tflite")
    for label in ("float32", "int8"):
        source = Path(report[label]["path"])
        suffix = "_float_outputs" if source.stem.endswith("_float_outputs") else ""
        destination = directory / f"best_{label}{suffix}.tflite"
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        report[label]["path"] = str(destination)
