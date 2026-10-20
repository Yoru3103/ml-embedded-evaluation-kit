# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Native ONNX-to-NHWC TFLite export without PyTorch or ExecuTorch dependencies."""

import argparse
from importlib import import_module
from pathlib import Path
import shutil

import numpy as np
import onnx

from model_export.common import images
from model_export.qrcode.onnx_utils import validate_outputs
from model_export.qrcode.tflite_utils import inspect_tflite, lite_runner

find_images = images.find_images
load_image_array = images.load_image_array



def check_nhwc_contract(contract: dict, metadata: dict):
    """Reject retained NCHW boundaries or altered auxiliary-output semantics.

    :param contract:    Actual interpreter input/output metadata.
    :param metadata:    Original ordered ONNX tensor shapes.
    """
    if contract["input_layout"] != "NHWC":
        raise ValueError("onnx2tf must expose an NHWC input")
    for actual, source in zip(contract["outputs"], metadata["outputs"]):
        shape = source["shape"]
        expected = [shape[index] for index in (0, 2, 3, 1)] if len(shape) == 4 else shape
        if actual["shape"] != expected:
            raise ValueError(f"Expected NHWC heads and unchanged auxiliary shape: {actual}")


def convert_tflite(graph: onnx.ModelProto, calibration: list[np.ndarray],
                   metadata: dict, args: argparse.Namespace) -> tuple[Path, Path]:
    """Convert a private ONNX copy with firmware-matched calibration data.

    :param graph:       Original ONNX graph, without PTE-specific rewrites.
    :param calibration: Float32 NCHW RGB samples, already normalized by 255.
    :param metadata:    Original ordered input/output tensor contract.
    :param args:        Converter and output options.
    :returns:           Generated FP32 and fully integer TFLite model paths.
    """
    directory = args.output_dir / "tflite"
    conversion = directory / "onnx2tf"
    conversion.mkdir(parents=True, exist_ok=True)
    source = conversion / "best.onnx"
    # Work on a copy: converter preprocessing must never modify the user's ONNX.
    onnx.save(graph, str(source))
    calibration_path = conversion / "calibration_nhwc.npy"
    values = np.concatenate(calibration, axis=0).transpose(0, 2, 3, 1)
    np.save(calibration_path, np.ascontiguousarray(values))
    import_module("onnx2tf").convert(
        input_onnx_file_path=str(source), output_folder_path=str(conversion),
        tflite_backend="tf_converter", output_integer_quantized_tflite=True,
        quant_type="per-channel" if args.per_channel else "per-tensor",
        input_quant_dtype="int8", output_quant_dtype="int8",
        custom_input_op_name_np_data_path=[
            [metadata["input"]["name"], str(calibration_path), 0.0, 1.0]],
        not_use_onnxsim=True, non_verbose=True)
    float_path = directory / "best_float32.tflite"
    int8_path = directory / "best_int8.tflite"
    shutil.copyfile(conversion / "best_float32.tflite", float_path)
    shutil.copyfile(conversion / "best_full_integer_quant.tflite", int8_path)
    return float_path, int8_path


def export_tflite(graph: onnx.ModelProto, calibration: list[np.ndarray],
                  samples: list[dict], metadata: dict, args: argparse.Namespace) -> dict:
    """Export and validate NHWC models before allowing NPU compilation.

    :param graph:       Original ONNX graph, without PTE-specific rewrites.
    :param calibration: Float32 NCHW RGB samples, already normalized by 255.
    :param samples:     ONNX reference predictions for host validation.
    :param metadata:    Original ordered input/output tensor contract.
    :param args:        Converter, validation and output options.
    :returns:           Validated FP32/INT8 paths, interfaces and operator counts.
    """
    float_path, int8_path = convert_tflite(graph, calibration, metadata, args)
    result = {"backend": "onnx2tf", "converter_backend": "tf_converter",
              "calibration_layout": "NHWC", "calibration_count": len(calibration),
              "preprocessing": args.preprocessing,
              "validation_scope": "ONNX versus host LiteRT; compiled NPU execution untested"}
    for name, path, integer in [("float32", float_path, False), ("int8", int8_path, True)]:
        predict, contract = lite_runner(path, metadata, integer)
        check_nhwc_contract(contract, metadata)
        result[name] = {"path": str(path), "tensor_contract": contract,
                        "operators": inspect_tflite(path),
                        "validation": validate_outputs(predict, samples, not integer,
                                                       args.max_score_error, args.min_box_iou)}
    return result
