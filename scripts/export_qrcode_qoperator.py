# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Compile the supplied grayscale QOperator ONNX for dedicated-cache FPGA testing."""

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

from qrcode_obb_utils import decode_top1, rotated_iou
from qrcode_qoperator_tflite import QOperatorTflite
from qrcode_tflite_utils import compile_tflite, inspect_tflite, lite_runner

ROOT = Path(__file__).resolve().parents[1]


def tensor_contract(model: onnx.ModelProto) -> dict:
    """Validate the exact grayscale raw-head deployment interface.

    :param model:   Original quantized ONNX.
    :returns:       Source metadata accepted by the existing TFLite tools.
    """
    tensors = []
    for value in list(model.graph.input) + list(model.graph.output):
        tensor = value.type.tensor_type
        if tensor.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError("The supplied QOperator model must have float32 boundaries")
        tensors.append({"name": value.name,
                        "shape": [dim.dim_value for dim in tensor.shape.dim]})
    expected = [[1, 1, 256, 256], [1, 47, 32, 32], [1, 47, 16, 16],
                [1, 47, 8, 8], [1, 1, 1344]]
    if [value["shape"] for value in tensors] != expected:
        raise ValueError("Expected single-channel input and DFL8 + 15-class raw heads")
    return {"input": tensors[0], "outputs": tensors[1:], "image_size": 256,
            "input_channels": 1, "num_classes": 15, "dfl_bins": 8,
            "oriented_boxes": True, "angle_mapping": "(sigmoid(raw)-0.25)*pi"}


def write_fixture(path: Path, gray: np.ndarray, outputs: list[np.ndarray],
                   contract: dict, decoded: dict):
    """Save actual outputs and expected OBB fields for the C++ host check.

    :param path:        Fixture path.
    :param gray:        Original uint8 grayscale pixels.
    :param outputs:     Ordered NCHW dequantized TFLite outputs.
    :param contract:    Interpreter quantization metadata.
    :param decoded:     Python reference decoding of those same outputs.
    """
    with path.open("wb") as stream:
        stream.write(gray.astype(np.uint8).tobytes())
        for value, metadata in zip(outputs, contract["outputs"]):
            scale, zero = metadata["scale"], metadata["zero_point"]
            if metadata["dtype"] == "float32":
                codes = value.astype(np.float32)
            else:
                codes = np.rint(value.astype(np.float64) / scale + zero).astype(np.int8)
            physical = codes.transpose(0, 2, 3, 1) if codes.ndim == 4 else codes
            shape = physical.shape
            stream.write(struct.pack("<I", len(shape)))
            stream.write(struct.pack("<" + "I" * len(shape), *shape))
            stream.write(struct.pack("<fiI", scale, zero, physical.size))
            stream.write(physical.tobytes())
        stream.write(struct.pack("<if", decoded["class_id"], decoded["score"]))
        stream.write(struct.pack("<5f", *decoded["obb_cxcywh_radians"]))
        stream.write(struct.pack("<4i", *decoded["display_xywh"]))


def parse_args() -> argparse.Namespace:
    """Parse the experimental FPGA export options.

    :returns:   Source, samples and dedicated-cache export options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "resources_downloaded/yolo_QRcode/"
                        "compare_result/best_bifpn_gray_qop_int8_boardcal_20260819.onnx")
    parser.add_argument("--samples", type=Path,
                        default=ROOT / "resources_downloaded/yolo_QRcode/compare_result")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "resources_downloaded/yolo_QRcode/gray_qoperator_384k")
    parser.add_argument("--cache-kib", type=int, default=384, choices=[384])
    parser.add_argument("--allow-runtime-rounding-differences", action="store_true",
                        help="Keep diagnostic numerical failures for an experimental board test")
    parser.add_argument("--preserve-boundary-qdq", action="store_true",
                        help="Retain float32 I/O and execute original boundary Q/DQ on CPU")
    return parser.parse_args()


def validate_image(path: Path, session: ort.InferenceSession, predict: object,
                   contract: dict, output_dir: Path) -> dict:
    """Compare one grayscale image and write its actual INT8 firmware fixture.

    :param path:        Original BMP.
    :param session:     Source ONNX Runtime session.
    :param predict:     Bound TFLite prediction callable.
    :param contract:    Interpreter tensor metadata.
    :param output_dir:  Export directory.
    :returns:           Detection and numerical differences.
    """
    with Image.open(path) as image:
        if image.mode != "L" or image.size != (256, 256):
            raise ValueError("Samples must be 256x256 grayscale BMPs")
        gray = np.asarray(image, dtype=np.uint8)
    inputs = gray.astype(np.float32)[None, None] / 255.0
    reference = session.run(None, {"images": inputs})
    observed = predict(inputs)
    decoded = decode_top1(observed, 256, 15, 8)
    expected = decode_top1(reference, 256, 15, 8)
    overlap = rotated_iou(decoded["obb_cxcywh_radians"], expected["obb_cxcywh_radians"])
    difference = abs(decoded["score"] - expected["score"])
    row = {"sample": path.parent.name, "source": expected, "tflite": decoded,
           "top1_score_difference": difference, "top1_rotated_iou": overlap,
           "same_class": decoded["class_id"] == expected["class_id"],
           "same_cell": decoded["global_cell"] == expected["global_cell"],
           "numerical_limits_passed": difference <= 0.15 and overlap >= 0.9,
           "raw_max_errors": [float(np.abs(first - second).max())
                              for first, second in zip(reference, observed)]}
    if not row["same_class"] or not decoded["detected"]:
        raise ValueError(f"Deployment sanity check failed: {row}")
    write_fixture(output_dir / f"host_fixtures/{path.parent.name}.bin", gray, observed,
                  contract, decoded)
    print(json.dumps(row), flush=True)
    return row


def validate_samples(args: argparse.Namespace, metadata: dict, raw_path: Path) -> dict:
    """Run all seven references and retain explicit numerical-warning status.

    :param args:        Export options.
    :param metadata:    Source tensor interface.
    :param raw_path:    Integer TFLite path.
    :returns:           Interface, operator and numerical validation records.
    """
    predict, contract = lite_runner(raw_path, metadata, not args.preserve_boundary_qdq)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    (args.output_dir / "host_fixtures").mkdir(exist_ok=True)
    results = [validate_image(path, session, predict, contract, args.output_dir)
               for path in sorted(args.samples.glob("*/input.bmp"))]
    if len(results) != 7:
        raise ValueError("Expected seven BMP inputs")
    passed = all(row["numerical_limits_passed"] for row in results)
    if not passed and not args.allow_runtime_rounding_differences:
        raise ValueError("Numerical differences recorded; explicit experimental-test flag required")
    return {"tensor_contract": contract, "source_operators": inspect_tflite(raw_path),
            "host_validation": results, "numerical_validation_passed": passed}


def perform_export(args: argparse.Namespace, model: onnx.ModelProto, report: dict):
    """Translate the existing parameters, validate and compile without recalibration.

    :param args:    Export options.
    :param model:   Read-only source graph.
    :param report:  Mutable stage report.
    """
    raw_path = args.output_dir / "tflite/best_int8.tflite"
    report["translation"] = QOperatorTflite(model, args.preserve_boundary_qdq).write(raw_path)
    report.update(validate_samples(args, report["contract"], raw_path))
    configuration = argparse.Namespace(
        output_dir=args.output_dir, target="ethos-u85-1024",
        vela_config=ROOT / "scripts/vela/default_vela.ini",
        system_config="Ethos_U85_SYS_DRAM_Mid_1024", memory_mode="Dedicated_Sram",
        arena_cache_size=393216, extra_flag=[])
    report["compiled"] = compile_tflite(configuration, raw_path,
                                        allow_float_boundaries=args.preserve_boundary_qdq)
    (args.output_dir / "labels.txt").write_text(
        "".join(f"class_{index}\n" for index in range(15)), encoding="utf-8")


def main():
    """Preserve quantization, record numerical differences and compile the NPU model."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original = args.model.read_bytes()
    model = onnx.load(str(args.model))
    onnx.checker.check_model(model)
    metadata = tensor_contract(model)
    report = {"status": "started", "source_sha256": hashlib.sha256(original).hexdigest(),
              "source": str(args.model), "contract": metadata,
              "configuration": {"target": "ethos-u85-1024",
                                "system_config": "Ethos_U85_SYS_DRAM_Mid_1024",
                                "memory_mode": "Dedicated_Sram", "arena_cache_size": 393216,
                                "preprocessing": "grayscale /255; RGB sample storage for LCD"},
              "validation_scope": "Host integer kernels and source ONNX; board not executed",
              "allow_runtime_rounding_differences": args.allow_runtime_rounding_differences}
    report["preserve_boundary_qdq"] = args.preserve_boundary_qdq
    manifest = args.output_dir / "export_report.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    try:
        perform_export(args, model, report)
        if args.model.read_bytes() != original:
            raise RuntimeError("Original ONNX changed")
        report["status"] = "completed"
    except Exception as error:
        report["status"], report["error"] = "failed", str(error)
        raise
    finally:
        manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Completed: {manifest}", flush=True)


if __name__ == "__main__":
    main()
