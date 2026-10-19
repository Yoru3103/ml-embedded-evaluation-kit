# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Compare real-model C++ decoding with Ultralytics OBB decoding and class sigmoid."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Callable

import numpy as np
import onnxruntime as ort
import torch
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.ops import xywhr2xyxyxyxy
from ultralytics.utils.tal import dist2rbox

from qrcode_contract import describe_model
from qrcode_export_backends import torch_predict
from qrcode_onnx_utils import detection_metrics, detection_values
from qrcode_tflite_utils import lite_runner
from yolo_pte_utils import load_image


def reference_decode(outputs: list[np.ndarray], metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    """Decode with Torch softmax and the installed upstream dist2rbox implementation.

    :param outputs:     Real-model raw outputs.
    :param metadata:    Graph-checked channel split and input resolution.
    :returns:           Class probabilities and pixel-coordinate XYWH/angle boxes.
    """
    heads = sorted((item for item in outputs if item.ndim == 4),
                   key=lambda item: item.shape[2], reverse=True)
    angles = torch.from_numpy(next(item for item in outputs if item.ndim == 3)).sigmoid()
    angles = (angles - 0.25) * torch.pi
    distances, scores, anchors, strides = [], [], [], []
    bins = metadata["dfl_bins"]
    for head in heads:
        raw = torch.from_numpy(head[0])
        probabilities = raw[:4 * bins].reshape(4, bins, -1).softmax(dim=1)
        distances.append((probabilities * torch.arange(bins)[None, :, None]).sum(dim=1))
        scores.append(raw[4 * bins:].reshape(metadata["num_classes"], -1).sigmoid())
        rows, columns = torch.meshgrid(torch.arange(head.shape[2]), torch.arange(head.shape[2]),
                                      indexing="ij")
        anchors.append(torch.stack((columns.flatten(), rows.flatten())) + 0.5)
        strides.append(torch.full((head.shape[2] ** 2,), metadata["image_size"] / head.shape[2]))
    boxes = dist2rbox(torch.cat(distances, dim=1)[None], angles,
                     torch.cat(anchors, dim=1)[None], dim=1)
    boxes *= torch.cat(strides)
    return torch.cat(scores, dim=1).numpy(), torch.cat((boxes, angles), dim=1)[0].T.numpy()


def run_decoder(outputs: list[np.ndarray], decoder: Path, metadata: dict) -> np.ndarray:
    """Send real float or dequantized model outputs through the actual C++ postprocessor.

    :param outputs:     Raw model outputs in any source order.
    :param decoder:     Built yolov8_processing_tests binary.
    :param metadata:    Graph-checked image size and head channels.
    :returns:           C++ class, score and unified XYWH/angle geometry rows.
    """
    heads = sorted((item for item in outputs if item.ndim == 4),
                   key=lambda item: item.shape[2], reverse=True)
    angle = next(item for item in outputs if item.ndim == 3)
    with tempfile.TemporaryDirectory(prefix="qrcode-decoder-") as directory:
        path = Path(directory) / "outputs.bin"
        path.write_bytes(b"".join(item.astype(np.float32).tobytes() for item in heads + [angle]))
        result = subprocess.run([str(decoder.resolve()), str(path), str(metadata["image_size"]),
                                 str(heads[0].shape[1]), str(metadata["num_classes"])],
                                check=True, capture_output=True, text=True)
    return np.array([[float(item) for item in line.split()]
                     for line in result.stdout.splitlines()], dtype=np.float64).reshape(-1, 7)


def check_detections(scores: np.ndarray, boxes: np.ndarray,
                     detections: np.ndarray) -> dict:
    """Check classes, confidence, OBB geometry and class-aware rotated NMS.

    :param scores:      Independently decoded class probabilities.
    :param boxes:       Upstream decoded oriented boxes.
    :param detections:  Actual C++ output rows.
    :returns:           Detection count, highest class/score and maximum decoding errors.
    """
    best = np.unravel_index(scores.argmax(), scores.shape)
    best_classes = np.flatnonzero(np.any(np.abs(scores - scores[best]) < 2e-7, axis=1))
    score_error, box_error = 0.0, 0.0
    if scores[best] >= 0.25:
        if detections.size == 0:
            raise AssertionError("C++ dropped every confident detection")
        np.testing.assert_allclose(detections[0, 1], scores[best], atol=2e-6, rtol=0)
        if int(detections[0, 0]) not in best_classes:
            raise AssertionError("C++ highest-confidence class differs from the model")
    for row in detections:
        errors = np.abs(boxes - row[2:])
        matching = np.flatnonzero(errors.max(axis=1) < 2e-3)
        matching = [anchor for anchor in matching
                    if scores[:, anchor].argmax() == int(row[0])
                    and abs(scores[int(row[0]), anchor] - row[1]) < 2e-6]
        if not matching:
            raise AssertionError(f"C++ detection has no matching model class/score/box: {row}")
        anchor = matching[0]
        score_error = max(score_error, float(abs(scores[int(row[0]), anchor] - row[1])))
        box_error = max(box_error, float(errors[anchor].max()))
    for index, row in enumerate(detections):
        others = detections[index + 1:]
        same_class = others[others[:, 0] == row[0]]
        if len(same_class):
            overlaps = batch_probiou(row[None, 2:].astype(np.float32),
                                     same_class[:, 2:].astype(np.float32))
            if float(overlaps.max()) > 0.4501:
                raise AssertionError("C++ retained overlapping detections of the same class")
    return {"detections": len(detections),
            "best_class": int(detections[0, 0]) if len(detections) else int(best[0]),
            "max_score_classes": best_classes.tolist(),
            "best_score": float(scores[best]), "max_cpp_score_error": score_error,
            "max_cpp_obb_error": box_error, "class_mismatches": 0}


def check_samples(predict: Callable, metadata: dict, paths: list[Path],
                  decoder: Path, reference: Callable | None = None) -> list[dict]:
    """Validate every sample against independent decoding, preserving optional quantization metrics.

    :param predict:     ONNX or integer LiteRT prediction callable.
    :param metadata:    Checked source model contract.
    :param paths:       Input images.
    :param decoder:     C++ postprocessing binary.
    :param reference:   Optional original ONNX predictor for quantization comparisons.
    :returns:           Per-image decoding and optional quantization evidence.
    """
    results = []
    for path in paths:
        inputs = load_image(path, metadata["image_size"], "mlek-crop",
                            metadata["input_channels"]).numpy()
        outputs = predict(inputs)
        scores, boxes = reference_decode(outputs, metadata)
        python_scores, envelopes = detection_values(outputs, metadata["image_size"],
                                                   metadata["num_classes"])
        np.testing.assert_allclose(python_scores, scores, atol=2e-7, rtol=0)
        expected = xywhr2xyxyxyxy(boxes)
        expected = np.concatenate((expected.min(axis=1), expected.max(axis=1)), axis=1)
        np.testing.assert_allclose(envelopes.T, expected, atol=2e-3, rtol=0)
        result = {"image": str(path), **check_detections(
            scores, boxes, run_decoder(outputs, decoder, metadata))}
        if reference is not None:
            result["quantization"] = detection_metrics(reference(inputs), outputs,
                                                       metadata["image_size"],
                                                       metadata["num_classes"])
        results.append(result)
        print(f"{path.name}: class={result['best_class']}, score={result['best_score']:.6f}, "
              f"detections={result['detections']}", flush=True)
    return results


def check_model(path: Path, args: argparse.Namespace, paths: list[Path]) -> tuple[dict, Callable]:
    """Check one ONNX model and return its predictor for the original deployment comparison.

    :param path:    Source ONNX model.
    :param args:    Decoder, report and optional LiteRT options.
    :param paths:   Validation samples.
    :returns:       ONNX evidence and prediction callable.
    """
    _, metadata = describe_model(path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), sess_options=options,
                                  providers=["CPUExecutionProvider"])
    def predict(inputs: np.ndarray) -> list[np.ndarray]:
        """Run all ONNX outputs in source graph order.

        :param inputs:  Prepared NCHW image.
        :returns:       Raw model outputs.
        """
        return session.run(None, {metadata["input"]["name"]: inputs})

    print(f"Model: {path}", flush=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "contract": metadata,
            "validation": check_samples(predict, metadata, paths, args.decoder)}, predict


def main():
    """Validate models and write a separate report without replacing deployment artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--samples", type=Path,
                        default=Path("resources/object_detection/samples_QRcode"))
    parser.add_argument("--tflite", type=Path,
                        help="Optional INT8 deployment source; uses the first ONNX contract")
    parser.add_argument("--pte-graph", type=Path,
                        help="Optional calibrated PT2E archive from the existing PTE deployment")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    paths = sorted(path for path in args.samples.iterdir()
                   if path.suffix.lower() in {".png", ".bmp", ".jpg", ".jpeg"})
    if not paths:
        raise ValueError("No validation images found")
    report = {"status": "running", "models": [],
              "scope": "Real-model class sigmoid, DFL/angle decoding and C++ rotated NMS; "
                       "class names and labeled detection accuracy are not evaluated"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    try:
        predictors = []
        for path in args.model:
            evidence, predict = check_model(path, args, paths)
            report["models"].append(evidence)
            predictors.append(predict)
        if args.tflite:
            metadata = report["models"][0]["contract"]
            predict, contract = lite_runner(args.tflite, metadata, True)
            report["tflite"] = {"path": str(args.tflite), "contract": contract,
                                "sha256": hashlib.sha256(args.tflite.read_bytes()).hexdigest(),
                                "validation": check_samples(predict, metadata, paths, args.decoder,
                                                             predictors[0])}
        if args.pte_graph:
            metadata = report["models"][0]["contract"]
            graph = torch.export.load(args.pte_graph).module(check_guards=False)
            report["pte_host"] = {
                "path": str(args.pte_graph),
                "sha256": hashlib.sha256(args.pte_graph.read_bytes()).hexdigest(),
                "validation": check_samples(lambda inputs: torch_predict(graph, inputs), metadata,
                                             paths, args.decoder, predictors[0])}
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
