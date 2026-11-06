#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Standalone QR OBB checkpoint calibration and PT2E export for model trainers."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from importlib import import_module
from importlib.metadata import version
import json
from pathlib import Path
import random
from typing import Iterator

import numpy as np
from PIL import Image

torch = import_module("torch")
YOLO = import_module("ultralytics").YOLO
EthosUCompileSpec = import_module("executorch.backends.arm.ethosu").EthosUCompileSpec
quantizer_api = import_module("executorch.backends.arm.quantizer")
EthosUQuantizer = quantizer_api.EthosUQuantizer
get_symmetric_quantization_config = quantizer_api.get_symmetric_quantization_config
pt2e_api = import_module("torchao.quantization.pt2e.quantize_pt2e")
convert_pt2e, prepare_pt2e = pt2e_api.convert_pt2e, pt2e_api.prepare_pt2e


class RawObb(torch.nn.Module):
    """Expose the original model's three raw detection maps and angle logits."""

    def __init__(self, model: torch.nn.Module):
        """Keep the checkpoint backbone and OBB branches.

        :param model:   Loaded Ultralytics OBB module.
        """
        super().__init__()
        self.model = model

    def fold_grayscale(self):
        """Replace the RGB first convolution with its equivalent grayscale weights."""
        convolution = self.model.model[0].conv
        if convolution.in_channels not in (1, 3) or convolution.groups != 1:
            raise ValueError("Expected one or three first-layer channels and groups=1")
        if convolution.in_channels == 3:
            convolution.weight = torch.nn.Parameter(
                convolution.weight.detach().sum(dim=1, keepdim=True))
            convolution.in_channels = 1

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return P3/P4/P5 DFL and class maps followed by raw angle logits.

        :param inputs:  NCHW float32 pixels normalized by 255.
        :returns:       Four ordered raw outputs.
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


class ChannelLastIO(torch.nn.Module):
    """Keep NCHW computation while exposing NHWC input and feature maps."""

    def __init__(self, model: torch.nn.Module):
        """Wrap an NCHW model.

        :param model:   Floating reference model.
        """
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Move four-dimensional boundaries to NHWC.

        :param inputs:  NHWC normalized pixels.
        :returns:       Ordered outputs with NHWC feature maps.
        """
        outputs = self.model(inputs.permute(0, 3, 1, 2))
        return tuple(self.output(value) for value in outputs)

    @staticmethod
    def output(value: torch.Tensor) -> torch.Tensor:
        """Move a four-dimensional feature map to NHWC.

        :param value:  NCHW map or unchanged auxiliary output.
        :returns:      Tensor in the deployment layout.
        """
        return value.permute(0, 2, 3, 1).contiguous() if value.ndim == 4 else value


def find_images(directory: Path, limit: int, seed: int) -> list[Path]:
    """Select a reproducible shuffled set of representative images.

    :param directory:  Image tree.
    :param limit:      Maximum number, or zero for all.
    :param seed:       Shuffle seed.
    :returns:          Selected image paths.
    """
    paths = sorted(path for path in directory.rglob("*")
                   if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not paths:
        raise ValueError(f"No images found in {directory}")
    random.Random(seed).shuffle(paths)
    return paths[:limit] if limit else paths


def load_image(path: Path, size: tuple[int, int], policy: str) -> torch.Tensor:
    """Apply the same grayscale calibration preparation as the repository exporter.

    :param path:    Source image.
    :param size:    Target width and height.
    :param policy:  ``mlek-crop`` or ``letterbox``.
    :returns:       Contiguous normalized NCHW float32 image.
    """
    width, height = size
    with Image.open(path) as source:
        image = source.convert("RGB" if policy == "mlek-crop" else "L")
    if policy == "mlek-crop":
        ratio = max(width / image.width, height / image.height)
        image = image.resize((int(image.width * ratio), int(image.height * ratio)),
                             Image.Resampling.BILINEAR)
        left, top = (image.width - width) / 2, (image.height - height) / 2
        image = image.crop((left, top, left + width, top + height)).convert("L")
    else:
        ratio = min(width / image.width, height / image.height)
        scaled_size = (min(width, max(1, round(image.width * ratio))),
                       min(height, max(1, round(image.height * ratio))))
        scaled = image.resize(scaled_size,
                              Image.Resampling.BILINEAR)
        padded = Image.new("L", size, 114)
        padded.paste(scaled, ((width - scaled.width) // 2, (height - scaled.height) // 2))
        image = padded
    values = np.asarray(image, dtype=np.float32)[None, None] / 255.0
    return torch.from_numpy(np.ascontiguousarray(values))


def model_input(image: torch.Tensor, layout: str) -> torch.Tensor:
    """Arrange the requested PT2 interface.

    :param image:   NCHW image.
    :param layout:  ``nhwc`` or ``nchw``.
    :returns:       Model input tensor.
    """
    return image.permute(0, 2, 3, 1).contiguous() if layout == "nhwc" else image


def predict(model: torch.nn.Module, image: np.ndarray, layout: str) -> list[np.ndarray]:
    """Restore model outputs to NCHW for source comparison.

    :param model:   Float or quantized graph.
    :param image:   Normalized NCHW source image.
    :param layout:  Graph boundary layout.
    :returns:       Four NCHW NumPy outputs.
    """
    with torch.no_grad():
        outputs = model(model_input(torch.from_numpy(image), layout))
        return [(value.permute(0, 3, 1, 2) if layout == "nhwc" and value.ndim == 4
                 else value).detach().numpy() for value in outputs]


def reference_samples(reference: RawObb, grayscale: RawObb, channels: int,
                      validation: list[Path], args: argparse.Namespace
                      ) -> tuple[list[dict], list[dict]]:
    """Check grayscale folding on the selected validation images.

    :param reference:   Original RGB or grayscale model.
    :param grayscale:   Folded one-channel model.
    :param channels:    Original input channels.
    :param validation:  Selected validation images.
    :param args:        Input size and preprocessing.
    :returns:           Validation samples and folding diagnostics.
    """
    samples, folding = [], []
    for path in validation:
        image = load_image(path, (args.image_width, args.image_height), args.preprocessing)
        with torch.inference_mode():
            expected = reference(image.repeat(1, channels, 1, 1))
            actual = grayscale(image)
            for index, (first, second) in enumerate(zip(expected, actual)):
                torch.testing.assert_close(first, second, rtol=1e-4,
                                           atol=1e-3 if index == 3 else 5e-4)
            samples.append({"image": str(path), "input": image.numpy(),
                            "reference": [value.numpy() for value in actual],
                            "num_classes": 15})
            folding.append({"image": str(path), "max_absolute_error": max(
                float((first - second).abs().max()) for first, second in zip(expected, actual))})
    return samples, folding


def source_contract(args: argparse.Namespace, samples: list[dict]) -> dict:
    """Check and describe the fixed four-output QR OBB interface.

    :param args:     Input dimensions.
    :param samples: Reference samples to check.
    :returns:        Source tensor contract.
    """
    sizes = [(args.image_height // stride, args.image_width // stride)
             for stride in (8, 16, 32)]
    shapes = [[1, 47, rows, columns] for rows, columns in sizes]
    shapes.append([1, 1, sum(rows * columns for rows, columns in sizes)])
    for sample in samples:
        if [list(value.shape) for value in sample["reference"]] != shapes:
            raise ValueError("Expected three 47-channel maps followed by angle logits")
    return {"input": {"name": "images", "shape": [1, 1, args.image_height, args.image_width]},
            "outputs": [{"name": name, "shape": shape} for name, shape in
                        zip(("p3_det", "p4_det", "p5_det", "angle"), shapes)],
            "image_size": (args.image_width if args.image_width == args.image_height
                           else [args.image_width, args.image_height]),
            "image_width": args.image_width, "image_height": args.image_height,
            "input_channels": 1, "num_classes": 15, "dfl_bins": 8,
            "auxiliary_kind": "angle_logits"}


def prepare_source(args: argparse.Namespace,
                   validation: list[Path]) -> tuple[RawObb, list[dict], dict]:
    """Load and verify the fixed QR OBB source contract and grayscale folding.

    :param args:        Export arguments.
    :param validation:  Validation image selection.
    :returns:           Grayscale model, reference samples and source metadata.
    """
    checkpoint = YOLO(str(args.model))
    source = checkpoint.model.cpu().float().eval()
    head = source.model[-1]
    if (checkpoint.task != "obb" or getattr(head, "end2end", False)
            or (head.nc, head.reg_max, head.ne, head.nl) != (15, 8, 1, 3)):
        raise ValueError("Expected non-end-to-end QR OBB with 15 classes and 8 DFL bins")
    channels = source.model[0].conv.in_channels
    reference = RawObb(source).eval()
    grayscale = RawObb(deepcopy(source)).eval()
    grayscale.fold_grayscale()
    samples, folding = reference_samples(reference, grayscale, channels, validation, args)
    names = source.names
    labels = [names.get(index, f"class_{index}") for index in range(15)]
    return grayscale, samples, {"contract": source_contract(args, samples),
                                "source_input_channels": channels,
                                "labels": labels,
                                "unnamed_class_ids": [index for index in range(15)
                                                      if index not in names],
                                "grayscale_folding": "sum RGB convolution weights; bias unchanged",
                                "folding_validation": folding}


def compare_outputs(reference: list[np.ndarray], actual: list[np.ndarray]) -> list[dict]:
    """Measure raw output errors and reject incompatible tensors.

    :param reference:  Source outputs.
    :param actual:     Graph outputs.
    :returns:          Per-output error statistics.
    """
    if len(reference) != len(actual):
        raise ValueError("Output count changed")
    results = []
    for expected, observed in zip(reference, actual):
        if (expected.shape != observed.shape or not np.isfinite(observed).all()
                or not np.isfinite(expected).all()):
            raise ValueError("Output shape changed or contains non-finite values")
        difference = np.abs(expected - observed)
        results.append({"shape": list(expected.shape),
                        "max_absolute_error": float(difference.max()),
                        "mean_absolute_error": float(difference.mean())})
    return results


def dfl_distances(head: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Decode one map's DFL logits into four pixel distances.

    :param head:  NCHW raw map.
    :param size:  Input width and height.
    :returns:     Left, top, right and bottom pixel distances.
    """
    logits = head[0, :32].reshape(4, 8, -1)
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    distances = (weights * np.arange(8)[None, :, None]).sum(axis=1)
    distances *= np.array([size[0] / head.shape[3], size[1] / head.shape[2],
                           size[0] / head.shape[3], size[1] / head.shape[2]])[:, None]
    return distances


def oriented_boxes(centers: np.ndarray, distances: np.ndarray,
                   angle: np.ndarray) -> np.ndarray:
    """Turn DFL distances and angles into axis-aligned OBB envelopes.

    :param centers:    Anchor center coordinates.
    :param distances:  Four decoded DFL distances.
    :param angle:      Angle radians per anchor.
    :returns:          XYXY envelopes.
    """
    shift = (distances[2:] - distances[:2]) * 0.5
    cosine, sine = np.cos(angle), np.sin(angle)
    centers += np.stack((shift[0] * cosine - shift[1] * sine,
                         shift[0] * sine + shift[1] * cosine))
    dimensions = distances[:2] + distances[2:]
    extents = np.stack((np.abs(cosine) * dimensions[0] + np.abs(sine) * dimensions[1],
                        np.abs(sine) * dimensions[0] + np.abs(cosine) * dimensions[1])) * 0.5
    return np.concatenate((centers - extents, centers + extents), axis=0)


def decode_head(head: np.ndarray, size: tuple[int, int], angle: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """Decode one DFL/class map into scores and OBB envelopes.

    :param head:   NCHW raw map.
    :param size:   Input width and height.
    :param angle:  Angle values for this head.
    :returns:      Scores and XYXY envelopes.
    """
    if head.shape[1] != 47:
        raise ValueError("Invalid DFL channel split")
    rows, columns = np.indices(head.shape[2:4])
    stride_x, stride_y = size[0] / head.shape[3], size[1] / head.shape[2]
    centers = np.stack(((columns.ravel() + 0.5) * stride_x,
                        (rows.ravel() + 0.5) * stride_y))
    boxes = oriented_boxes(centers, dfl_distances(head, size), angle)
    scores = 1 / (1 + np.exp(-np.clip(head[0, 32:].reshape(15, -1), -80, 80)))
    return scores, boxes


def detection_values(outputs: list[np.ndarray],
                     size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Decode scores and OBB envelopes for quantization acceptance.

    :param outputs:  Three raw maps and raw angle logits.
    :param size:     Input width and height.
    :returns:      Class scores and XYXY envelopes.
    """
    heads = sorted((value for value in outputs if value.ndim == 4),
                   key=lambda value: value.shape[2] * value.shape[3], reverse=True)
    angles = (1 / (1 + np.exp(-np.clip(next(value for value in outputs if value.ndim == 3)[0, 0],
                                      -80, 80))) - 0.25) * np.pi
    scores, boxes, offset = [], [], 0
    for head in heads:
        cells = head.shape[2] * head.shape[3]
        head_scores, head_boxes = decode_head(head, size, angles[offset:offset + cells])
        offset += cells
        scores.append(head_scores)
        boxes.append(head_boxes)
    if offset != angles.size:
        raise ValueError("Angle count does not match feature cells")
    return np.concatenate(scores, axis=1), np.concatenate(boxes, axis=1)


def detection_metrics(reference: list[np.ndarray], actual: list[np.ndarray],
                      size: tuple[int, int]) -> dict:
    """Compare class scores and the strongest source anchor's box.

    :param reference:  Source model outputs.
    :param actual:     Quantized graph outputs.
    :param size:       Input width and height.
    :returns:          Quantization acceptance metrics.
    """
    scores, boxes = detection_values(reference, size)
    observed, quantized_boxes = detection_values(actual, size)
    location = np.unravel_index(scores.argmax(), scores.shape)
    anchor = int(location[1])
    original, changed = boxes[:, anchor], quantized_boxes[:, anchor]
    intersection = np.maximum(0, np.minimum(original[2:], changed[2:])
                              - np.maximum(original[:2], changed[:2])).prod()
    union = sum((box[2:] - box[:2]).prod() for box in (original, changed)) - intersection
    return {"float_max_score": float(scores.max()),
            "quantized_max_score": float(observed.max()),
            "quantized_score_at_float_best": float(observed[location]),
            "max_absolute_score_error": float(np.abs(scores - observed).max()),
            "mean_absolute_score_error": float(np.abs(scores - observed).mean()),
            "box_iou_at_float_best": float(intersection / union) if union > 0 else 0.0,
            "float_best_class": int(location[0]), "float_best_anchor": anchor,
            "quantized_best_class": int(np.unravel_index(observed.argmax(), observed.shape)[0])}


def validate(model: torch.nn.Module, samples: list[dict], args: argparse.Namespace,
             strict: bool) -> list[dict]:
    """Check float capture parity or quantized detection tolerance.

    :param model:    Captured or quantized model.
    :param samples:  Original floating reference samples.
    :param args:     Layout and acceptance limits.
    :param strict:   Require FP32 parity when true.
    :returns:        Per-image diagnostics.
    """
    results = []
    for sample in samples:
        actual = predict(model, sample["input"], args.io_layout if not strict else "nchw")
        expected = sample["reference"]
        result = {"image": sample["image"], "outputs": compare_outputs(expected, actual)}
        if strict:
            for index, (first, second) in enumerate(zip(expected, actual)):
                np.testing.assert_allclose(second, first, rtol=1e-4,
                                           atol=5e-4 if index == 3 else 2e-4)
        else:
            metrics = detection_metrics(expected, actual,
                                        (args.image_width, args.image_height))
            result["detection"] = metrics
            if (metrics["max_absolute_score_error"] > args.max_score_error
                    or metrics["box_iou_at_float_best"] < args.min_box_iou):
                raise ValueError(f"Quantization validation failed for {sample['image']}: {metrics}")
        results.append(result)
    return results


def file_hash(path: Path) -> str:
    """Hash a checkpoint, image or export artifact.

    :param path:  File to hash.
    :returns:     SHA256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(args: argparse.Namespace, calibration: list[Path], validation: list[Path]) -> dict:
    """Record the options and input bytes that define the quantized PT2.

    :param args:         Export arguments.
    :param calibration:  Selected calibration images.
    :param validation:   Selected validation images.
    :returns:            Cache identity.
    """
    result = {name: getattr(args, name) for name in
              ("image_width", "image_height", "io_layout", "preprocessing", "per_channel",
               "seed", "target")}
    result["source_sha256"] = file_hash(args.model)
    result["versions"] = {name: version(name) for name in ("torch", "executorch", "torchao")}
    for name, paths in (("calibration", calibration), ("validation", validation)):
        result[name] = [{"path": str(path), "sha256": file_hash(path)} for path in paths]
    return result


@contextmanager
def manifest(path: Path, report: dict) -> Iterator[dict]:
    """Persist completed or failed export status.

    :param path:    Report path.
    :param report:  Mutable export report.
    :returns:       Context holding the report.
    """
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    try:
        yield report
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    else:
        report["status"] = "completed"
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def quantize(model: RawObb, calibration: list[Path], args: argparse.Namespace,
             example: torch.Tensor) -> torch.fx.GraphModule:
    """Capture, prepare, calibrate and convert with the Ethos-U PT2E quantizer.

    :param model:        Verified grayscale reference model.
    :param calibration:  Representative image paths.
    :param args:         Quantization options.
    :param example:      Model input example in the chosen layout.
    :returns:            Quantized graph.
    """
    if args.io_layout == "nhwc":
        model = ChannelLastIO(model).eval()
    exported = torch.export.export(model, (example,), strict=True)
    flags = list(args.extra_flag) + [f"--arena-cache-size={args.arena_cache_size}"]
    mode = "Dedicated_Sram" if args.memory_mode == "all" else args.memory_mode
    spec = EthosUCompileSpec(target=args.target, config_ini=str(args.vela_config),
                             system_config=args.system_config, memory_mode=mode,
                             extra_flags=flags)
    quantizer = EthosUQuantizer(spec).set_global(
        get_symmetric_quantization_config(is_per_channel=args.per_channel))
    prepared = prepare_pt2e(exported.module(check_guards=False), quantizer)
    with torch.no_grad():
        for index, path in enumerate(calibration, 1):
            prepared(model_input(load_image(path, (args.image_width, args.image_height),
                                            args.preprocessing), args.io_layout))
            if index % 16 == 0 or index == len(calibration):
                print(f"Calibrated {index}/{len(calibration)}", flush=True)
    return convert_pt2e(prepared)


def reuse_cache(args: argparse.Namespace, expected_identity: dict) -> dict:
    """Verify and reuse a previously exported PT2 cache.

    :param args:               Export settings and output path.
    :param expected_identity:  Current source and input identity.
    :returns:                  Verified cached report.
    """
    path = args.output_dir / "quantized.pt2"
    cached = json.loads((args.output_dir / "pt2_report.json").read_text(encoding="utf-8"))
    references = args.output_dir / "references.npz"
    if (cached.get("status") != "completed" or cached.get("identity") != expected_identity
            or cached.get("pt2_sha256") != file_hash(path)
            or cached.get("references_sha256") != file_hash(references)):
        raise ValueError("PT2 cache mismatch or incomplete cache; choose a new output directory")
    samples = []
    with np.load(references, allow_pickle=False) as arrays:
        for index, image in enumerate(cached["validation_images"]):
            samples.append({"image": image, "input": arrays[f"input_{index}"],
                            "reference": [arrays[f"reference_{index}_{head}"]
                                          for head in range(4)]})
    validate(torch.export.load(path).module(check_guards=False), samples, args, False)
    print(f"Reusing calibrated PT2: {path}", flush=True)
    return cached


def save_artifacts(args: argparse.Namespace, graph: torch.fx.GraphModule,
                   example: torch.Tensor, samples: list[dict], labels: list[str]) -> dict:
    """Write the PT2, floating reference archive and labels.

    :param args:      Export destination.
    :param graph:     Validated quantized graph.
    :param example:   Input example in the graph layout.
    :param samples:   Floating validation samples.
    :param labels:    Source class names.
    :returns:         Artifact hashes for the cache report.
    """
    path = args.output_dir / "quantized.pt2"
    temporary = args.output_dir / "quantized.tmp.pt2"
    torch.export.save(torch.export.export(graph, (example,), strict=True), temporary)
    temporary.replace(path)
    arrays = {}
    for index, sample in enumerate(samples):
        arrays[f"input_{index}"] = sample["input"]
        arrays.update({f"reference_{index}_{head}": value
                       for head, value in enumerate(sample["reference"])})
    references = args.output_dir / "references.npz"
    np.savez_compressed(references, **arrays)
    (args.output_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    return {"pt2_sha256": file_hash(path), "references_sha256": file_hash(references)}


def run(args: argparse.Namespace) -> dict:
    """Export a reusable PT2 with the same calibration contract as the QR PTE path.

    :param args:  Validated command-line options.
    :returns:     Completed or reused report.
    """
    torch.set_num_threads(args.threads)
    calibration = find_images(args.calibration_dir, args.calibration_limit, args.seed)
    validation = find_images(args.validation_dir, args.validation_limit, args.seed + 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "pt2_report.json"
    current_identity = identity(args, calibration, validation)
    if report_path.exists():
        return reuse_cache(args, current_identity)
    report = {"status": "started", "identity": current_identity}
    with manifest(report_path, report):
        model, samples, source = prepare_source(args, validation)
        report.update(source)
        report["calibration_images"] = [str(value) for value in calibration]
        report["validation_images"] = [str(value) for value in validation]
        report["independent_validation"] = not bool(set(calibration).intersection(validation))
        captured = torch.export.export(model, (torch.from_numpy(samples[0]["input"]),),
                                       strict=True).module(check_guards=False)
        report["float_validation"] = validate(captured, samples, args, True)
        example = model_input(load_image(calibration[0], (args.image_width, args.image_height),
                                         args.preprocessing), args.io_layout)
        graph = quantize(model, calibration, args, example)
        report["quantization_validation"] = validate(graph, samples, args, False)
        report.update(save_artifacts(args, graph, example, samples, source["labels"]))
    print(f"Calibrated PT2 saved: {args.output_dir / 'quantized.pt2'}", flush=True)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the standalone trainer-facing PT-to-PT2 command.

    :param argv:  Optional command-line tokens.
    :returns:     Validated and resolved settings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path,
                        help="Defaults to --calibration-dir; use independent images when possible")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--image-width", type=int)
    parser.add_argument("--image-height", type=int)
    parser.add_argument("--io-layout", choices=("nhwc", "nchw"), default="nhwc")
    parser.add_argument("--preprocessing", choices=("mlek-crop", "letterbox"),
                        default="mlek-crop")
    parser.add_argument("--calibration-limit", type=int, default=128)
    parser.add_argument("--validation-limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-score-error", type=float, default=0.15)
    parser.add_argument("--min-box-iou", type=float, default=0.9)
    parser.add_argument("--per-channel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target", default="ethos-u85-1024")
    parser.add_argument("--vela-config", type=Path,
                        default=Path(__file__).with_name("default_vela.ini"))
    parser.add_argument("--system-config", default="Ethos_U85_SYS_DRAM_Mid_1024")
    parser.add_argument("--memory-mode", choices=("Dedicated_Sram", "Shared_Sram", "all"),
                        default="Dedicated_Sram")
    parser.add_argument("--arena-cache-size", type=int, default=393216)
    parser.add_argument("--extra-flag", action="append", default=[])
    args = parser.parse_args(argv)
    args.image_width = args.image_width or args.image_size
    args.image_height = args.image_height or args.image_size
    args.validation_dir = args.validation_dir or args.calibration_dir
    for name in ("model", "calibration_dir", "validation_dir", "output_dir", "vela_config"):
        setattr(args, name, getattr(args, name).resolve())
    if not args.model.is_file() or args.model.suffix.lower() != ".pt":
        parser.error("--model must name an existing .pt file")
    if not args.calibration_dir.is_dir() or not args.validation_dir.is_dir():
        parser.error("Calibration and validation directories must exist")
    if not args.vela_config.is_file():
        parser.error("--vela-config must name an existing Vela configuration")
    if any(size <= 0 or size % 32 for size in (args.image_width, args.image_height)):
        parser.error("Image width and height must be positive multiples of 32")
    invalid_counts = (args.calibration_limit < 0 or args.validation_limit <= 0
                      or args.threads <= 0 or args.arena_cache_size <= 0)
    invalid_thresholds = (not 0 <= args.max_score_error <= 1
                          or not 0 <= args.min_box_iou <= 1)
    if invalid_counts or invalid_thresholds:
        parser.error("Image limits, thread count or acceptance thresholds are invalid")
    return args


def main():
    """Run standalone calibration and emit a portable PT2 artifact."""
    run(parse_args())


if __name__ == "__main__":
    main()
