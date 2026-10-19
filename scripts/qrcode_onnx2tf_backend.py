# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Native ONNX-to-NHWC TFLite export without PyTorch or ExecuTorch dependencies."""

import argparse
from importlib import import_module
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image, ImageOps
import onnx

from qrcode_onnx_utils import validate_outputs
from qrcode_tflite_utils import inspect_tflite, lite_runner


def find_images(directory: Path, limit: int, seed: int) -> list[Path]:
    """Select reproducible calibration images without loading a training framework.

    :param directory:   Directory containing RGB samples.
    :param limit:       Maximum images, or zero to use all images.
    :param seed:        Deterministic shuffle seed.
    :returns:           Selected image paths.
    """
    paths = sorted(path for path in directory.rglob("*")
                   if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not paths:
        raise ValueError(f"No calibration images found in {directory}")
    random.Random(seed).shuffle(paths)
    return paths[:limit] if limit else paths


def load_image_array(path: Path, image_size: int, preprocessing: str,
                     channels: int = 3) -> np.ndarray:
    """Prepare RGB data, matching the firmware generator's centered crop exactly.

    :param path:            Source RGB image.
    :param image_size:      Square deployment resolution.
    :param preprocessing:  Centered aspect crop or padded letterbox.
    :param channels:        Model input channels, one or three.
    :returns:               Contiguous float32 NCHW data divided by 255.
    """
    with Image.open(path) as source:
        image = source.convert("RGB")
    if preprocessing == "mlek-crop":
        ratio = image_size / min(image.size)
        image = image.resize((int(image.width * ratio), int(image.height * ratio)),
                             Image.Resampling.BILINEAR)
        left = (image.width - image_size) / 2
        top = (image.height - image_size) / 2
        image = image.crop((left, top, left + image_size, top + image_size))
    elif preprocessing == "letterbox":
        ratio = image_size / max(image.size)
        image = image.resize((round(image.width * ratio), round(image.height * ratio)),
                             Image.Resampling.BILINEAR)
        width, height = image_size - image.width, image_size - image.height
        image = ImageOps.expand(image, (width // 2, height // 2,
                                      width - width // 2, height - height // 2),
                                fill=(114, 114, 114))
    else:
        raise ValueError(f"Unknown preprocessing: {preprocessing}")
    if channels == 1:
        values = np.asarray(image.convert("L"), dtype=np.float32)[None, None] / 255.0
    elif channels == 3:
        values = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    else:
        raise ValueError("Model input must have one or three channels")
    return np.ascontiguousarray(values)


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
