#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Export grayscale QR-code OBB weights to raw ONNX or directly to Ethos-U PTE."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import sys

import numpy as np
from model_export.common.paths import ROOT
from model_export.common.reporting import export_manifest
from model_export.qrcode.contract import describe_model
from model_export.common.images import find_images, load_image

torch = import_module("torch")
OUTPUT_NAMES = ("p3_det", "p4_det", "p5_det", "angle")


class RawObb(torch.nn.Module):
    """Expose three DFL/class maps and raw angle logits without decoding or NMS."""

    def __init__(self, model: torch.nn.Module):
        """Keep the checkpoint's backbone and OBB branches.

        :param model:   Loaded Ultralytics OBB model.
        """
        super().__init__()
        self.model = model

    def fold_grayscale(self):
        """Sum RGB first-layer weights, retaining its bias and batch normalization.

        :raises ValueError: Unsupported first-layer input channels or grouping.
        """
        convolution = self.model.model[0].conv
        if convolution.in_channels not in (1, 3) or convolution.groups != 1:
            raise ValueError("Expected a one- or three-channel first convolution with groups=1")
        if convolution.in_channels == 3:
            convolution.weight = torch.nn.Parameter(
                convolution.weight.detach().sum(dim=1, keepdim=True))
            convolution.in_channels = 1

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run the original feature routing and expose the raw prediction branches.

        :param inputs:  NCHW float32 pixels normalized by 255.
        :returns:       P3/P4/P5 DFL and class logits, followed by raw angle logits.
        """
        saved = []
        current = inputs
        for layer in self.model.model[:-1]:
            if layer.f != -1:
                current = (saved[layer.f] if isinstance(layer.f, int)
                           else [current if index == -1 else saved[index] for index in layer.f])
            current = layer(current)
            saved.append(current)
        head = self.model.model[-1]
        features = [saved[index] for index in head.f]
        detections = tuple(torch.cat((head.cv2[index](feature), head.cv3[index](feature)), dim=1)
                           for index, feature in enumerate(features))
        angles = torch.cat([head.cv4[index](feature).reshape(1, 1, -1)
                            for index, feature in enumerate(features)], dim=2)
        return (*detections, angles)


def prepare_models(weights: Path,
                   checkpoint: object | None = None) -> tuple[RawObb, RawObb, int, dict]:
    """Create independent RGB-reference and single-channel raw-head models.

    :param weights:     Original OBB checkpoint.
    :param checkpoint:  Optional already loaded Ultralytics checkpoint.
    :returns:           Reference model, grayscale model, source channels and names.
    :raises ValueError: The checkpoint does not match the QR-code use case.
    """
    yolo = checkpoint if checkpoint is not None else import_module("ultralytics").YOLO(str(weights))
    model = yolo.model.cpu().float().eval()
    head = model.model[-1]
    if (yolo.task != "obb" or getattr(head, "end2end", False)
            or (head.nc, head.reg_max, head.ne, head.nl) != (15, 8, 1, 3)):
        raise ValueError("Expected a non-end-to-end OBB model with 15 classes and 8 DFL bins")
    first = model.model[0].conv
    channels = first.in_channels
    grayscale = RawObb(deepcopy(model)).eval()
    grayscale.fold_grayscale()
    return RawObb(model).eval(), grayscale, channels, model.names


def reference_samples(reference: RawObb, grayscale: RawObb, paths: list[Path], image_size: int,
                      preprocessing: str = "mlek-crop") -> list[dict]:
    """Verify grayscale weight folding against repeated grayscale RGB images.

    :param reference:       Original model with raw output heads.
    :param grayscale:       Model with a one-channel first convolution.
    :param paths:           Images used to check both models.
    :param image_size:      Fixed square input resolution.
    :param preprocessing:   Image resize policy shared with calibration.
    :returns:               Cached grayscale inputs and checked floating outputs.
    """
    samples = []
    channels = reference.model.model[0].conv.in_channels
    with torch.inference_mode():
        for path in paths:
            image = load_image(path, image_size, preprocessing, channels=1)
            expected = reference(image.repeat(1, channels, 1, 1))
            actual = grayscale(image)
            for first, second in zip(expected, actual):
                torch.testing.assert_close(first, second, rtol=1e-4, atol=5e-4)
            samples.append({"path": str(path), "input": image.numpy(),
                            "outputs": [value.numpy() for value in actual],
                            "folding_max_absolute_error": max(
                                float((first - second).abs().max())
                                for first, second in zip(expected, actual))})
    return samples


def prepare_pte_source(args: argparse.Namespace, checkpoint: object) -> tuple:
    """Check native grayscale OBB capture against the original RGB weights.

    :param args:        Input size, calibration and validation selection.
    :param checkpoint:  Loaded Ultralytics OBB checkpoint.
    :returns:           Native grayscale model, validation samples and source report.
    :raises ValueError: The model or output contract is incompatible with qrcode.
    """
    if args.image_size != 256:
        raise ValueError("Direct QR-code PT export requires --image-size 256")
    reference, grayscale, channels, names = prepare_models(args.model, checkpoint)
    paths = find_images(args.calibration_dir, args.calibration_limit, args.seed)
    checks = find_images(args.validation_dir, args.validation_limit, args.seed + 1)
    verified = reference_samples(reference, grayscale, checks,
                                 args.image_size, args.preprocessing)
    expected = [[1, 47, side, side] for side in (32, 16, 8)] + [[1, 1, 1344]]
    for sample in verified:
        if [list(value.shape) for value in sample["outputs"]] != expected:
            raise ValueError("Direct PT export requires three 47-channel heads and angle logits")
    metadata = {"input": {"name": "images", "shape": [1, 1, 256, 256]},
                "outputs": [{"name": name, "shape": shape}
                            for name, shape in zip(OUTPUT_NAMES, expected)],
                "image_size": 256, "input_channels": 1, "num_classes": 15,
                "dfl_bins": 8, "auxiliary_kind": "angle_logits"}
    samples = [{"image": sample["path"], "input": sample["input"],
                "reference": sample["outputs"], "num_classes": 15} for sample in verified]
    return grayscale, samples, {
        "contract": metadata, "calibration_images": [str(path) for path in paths],
        "validation_images": [str(path) for path in checks],
        "independent_validation": not bool(set(paths).intersection(checks)),
        "source_input_channels": channels,
        "labels": [names.get(index, f"class_{index}") for index in range(15)],
        "unnamed_class_ids": [index for index in range(15) if index not in names],
        "grayscale_folding": "sum RGB convolution weights; bias unchanged",
        "folding_validation": [{"image": sample["path"],
                                "max_absolute_error": sample["folding_max_absolute_error"]}
                               for sample in verified],
        "float_validation": import_module("model_export.qrcode.backends").validate_native_capture(
            grayscale, samples)}


def run_pte_export(args: argparse.Namespace, checkpoint: object) -> dict:
    """Export the checkpoint through native torch.export/PT2E without ONNX conversion.

    :param args:        Unified PTE export arguments.
    :param checkpoint:  Already loaded OBB checkpoint.
    :returns:           Source parity, quantization and serialized PTE report.
    """
    os.environ["PATH"] = os.pathsep.join(
        (str(Path(sys.executable).absolute().parent), os.environ.get("PATH", "")))
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "export_report.json"
    report = {"status": "started", "source": str(args.model),
              "source_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
              "source_route": "PT -> native grayscale raw OBB -> PT2E -> Ethos-U PTE",
              "configuration": {key: str(value) if isinstance(value, Path) else value
                                for key, value in vars(args).items()},
              "validation_scope": "PT versus host PT2E; compiled NPU execution untested"}
    with export_manifest(manifest, report):
        model, samples, source = prepare_pte_source(args, checkpoint)
        report.update(source)
        labels = args.output_dir / "labels.txt"
        labels.write_text("\n".join(source["labels"]) + "\n", encoding="utf-8")
        report["labels_file"] = str(labels)
        report["pte"] = {}
        backends = import_module("model_export.qrcode.backends")
        options = argparse.Namespace(**vars(args))
        options.reference_source = "Original PT raw OBB heads"
        graph = backends.quantize_pte(
            model, [Path(path) for path in source["calibration_images"]],
            source["contract"], options)
        backends.compile_pte_variants(graph, samples, source["contract"], options, report["pte"])
    print(json.dumps({"report": str(manifest), "status": report["status"]}), flush=True)
    return report


def try_export_pte(args: argparse.Namespace) -> dict | None:
    """Route OBB checkpoints directly, leaving Detect checkpoints to the generic exporter.

    :param args:    Unified PT-to-PTE arguments.
    :returns:       OBB export report, or no result for another checkpoint task.
    """
    if getattr(args, "pt2_only", False) or getattr(args, "quantized_pt2", None) is not None:
        return import_module("model_export.qrcode.pt2_cache").run_export(args)
    checkpoint = import_module("ultralytics").YOLO(str(args.model))
    if checkpoint.task != "obb":
        return None
    return run_pte_export(args, checkpoint)


def normalize_source(path: Path):
    """Fold tensor Constants and expose Resize ROI inputs for the static ONNX adapter.

    :param path:    Floating ONNX written by the PyTorch exporter.
    """
    onnx = import_module("onnx")
    graph = onnx.load(path)
    retained = []
    for index, node in enumerate(graph.graph.node):
        if (node.op_type == "Constant" and len(node.attribute) == 1
                and node.attribute[0].name == "value"
                and node.attribute[0].type == onnx.AttributeProto.TENSOR):
            value = onnx.numpy_helper.to_array(node.attribute[0].t)
            graph.graph.initializer.append(onnx.numpy_helper.from_array(value, node.output[0]))
            continue
        if node.op_type == "Resize" and len(node.input) >= 3 and not node.input[1]:
            name = f"qrcode_empty_resize_roi_{index}"
            graph.graph.initializer.append(onnx.numpy_helper.from_array(
                np.empty(0, dtype=np.float32), name))
            node.input[1] = name
        retained.append(node)
    graph.graph.ClearField("node")
    graph.graph.node.extend(retained)
    onnx.save(graph, path)


def validate_onnx(path: Path, samples: list[dict]) -> tuple[dict, list[dict]]:
    """Check the serialized four-output contract and compare all raw floating outputs.

    :param path:    Temporary exported ONNX file.
    :param samples: Inputs and outputs checked against the RGB checkpoint.
    :returns:       Tensor metadata and per-image numerical errors.
    """
    ort = import_module("onnxruntime")
    graph, metadata = describe_model(path)
    if (metadata["input_channels"], metadata["num_classes"], metadata["dfl_bins"]) != (1, 15, 8):
        raise ValueError("Exported grayscale tensor contract does not match the use case")
    options = ort.SessionOptions()
    options.intra_op_num_threads = torch.get_num_threads()
    session = ort.InferenceSession(graph.SerializeToString(), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    checks = []
    for sample in samples:
        observed = session.run(list(OUTPUT_NAMES), {"images": sample["input"]})
        errors = []
        for expected, actual in zip(sample["outputs"], observed):
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=5e-4)
            errors.append(float(np.max(np.abs(expected - actual))))
        checks.append({"image": sample["path"],
                       "folding_max_absolute_error": sample["folding_max_absolute_error"],
                       "onnx_max_absolute_errors": dict(zip(OUTPUT_NAMES, errors))})
    return metadata, checks


def export_model(weights: Path, output: Path, paths: list[Path], image_size: int) -> dict:
    """Export an FP32 ONNX and check its interface and CPU results before publishing it.

    :param weights:     Source OBB checkpoint.
    :param output:      Destination for the grayscale floating ONNX.
    :param paths:       Independent host validation images.
    :param image_size:  Input size supported by the fixed QR-code use case.
    :returns:           Source, interface, label and numerical validation report.
    """
    if image_size != 256:
        raise ValueError("The QR-code use case requires 256x256 inputs")
    reference, grayscale, channels, names = prepare_models(weights)
    samples = reference_samples(reference, grayscale, paths, image_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".onnx.tmp")
    try:
        torch.onnx.export(grayscale, (torch.from_numpy(samples[0]["input"]),), str(temporary),
                          input_names=["images"], output_names=list(OUTPUT_NAMES),
                          opset_version=13, dynamo=False, do_constant_folding=True)
        normalize_source(temporary)
        metadata, checks = validate_onnx(temporary, samples)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    label_path = output.with_suffix(".labels.txt")
    labels = [names.get(index, f"class_{index}") for index in range(metadata["num_classes"])]
    label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    report = {"source": str(weights),
              "source_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
              "output": str(output), "source_input_channels": channels,
              "input_normalization": "uint8 / 255", "source_onnx_layout": "NCHW",
              "grayscale_folding": "sum RGB convolution weights; bias unchanged",
              "quantized": False, "contract": metadata, "labels": str(label_path),
              "unnamed_class_ids": [index for index in range(len(labels)) if index not in names],
              "validation": checks}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    """Export the floating grayscale source consumed by export_model.py."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "resources_downloaded/qrcode/best_gray.onnx")
    parser.add_argument("--validation-dir", type=Path, default=ROOT / "resources/qrcode/smaples")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not args.weights.is_file() or args.threads <= 0:
        parser.error("Checkpoint must exist and threads must be positive")
    torch.set_num_threads(args.threads)
    paths = find_images(args.validation_dir, limit=0, seed=0)
    report = export_model(args.weights.resolve(), args.output.resolve(), paths, 256)
    print(json.dumps({key: report[key] for key in ("output", "contract", "labels",
                                                  "unnamed_class_ids")}, indent=2))


if __name__ == "__main__":
    main()
