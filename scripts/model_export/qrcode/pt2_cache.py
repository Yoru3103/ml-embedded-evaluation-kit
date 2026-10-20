# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Cache calibrated native QR-code PT2 graphs independently of Vela memory budgets."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from model_export.common.reporting import export_manifest
from model_export.qrcode.pt import prepare_pte_source
from model_export.common.layout import model_input
from model_export.qrcode.backends import compile_pte_variants, quantize_pte, torch_predict
from model_export.qrcode.onnx_utils import validate_outputs
from model_export.common.images import find_images


def file_hash(path: Path) -> str:
    """Hash one source or cached artifact.

    :param path:    File to hash.
    :returns:       SHA256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_identity(args: argparse.Namespace) -> dict:
    """Identify calibration semantics while excluding Vela scheduling options.

    :param args:    Source and quantization options.
    :returns:       Configuration, tool versions and selected image hashes.
    """
    names = ("image_size", "io_layout", "preprocessing", "per_channel", "seed", "target")
    result = {name: getattr(args, name) for name in names}
    result["source_sha256"] = file_hash(args.model)
    result["versions"] = {name: version(name) for name in ("torch", "executorch", "torchao")}
    for name, limit, seed in (("calibration", args.calibration_limit, args.seed),
                              ("validation", args.validation_limit, args.seed + 1)):
        paths = find_images(getattr(args, f"{name}_dir"), limit, seed)
        result[name] = [{"path": str(path.resolve()), "sha256": file_hash(path)} for path in paths]
    return result


def load_cache(path: Path, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    """Check cached provenance and restore independent floating reference outputs.

    :param path:    Quantized PT2 file beside its manifest and reference archive.
    :param args:    Requested source and quantization options.
    :returns:       Checked manifest and validation samples.
    :raises ValueError: Cache is incomplete, changed or incompatible.
    """
    report = json.loads((path.parent / "pt2_report.json").read_text())
    references = path.parent / "references.npz"
    if (report["status"] != "completed" or report["identity"] != cache_identity(args)
            or report["pt2_sha256"] != file_hash(path)
            or report["references_sha256"] != file_hash(references)):
        raise ValueError("PT2 cache mismatch or incomplete cache; use a new common directory")
    samples = []
    with np.load(references, allow_pickle=False) as arrays:
        for index, image in enumerate(report["validation_images"]):
            samples.append({"image": image, "num_classes": 15,
                            "input": arrays[f"input_{index}"],
                            "reference": [arrays[f"reference_{index}_{head}"]
                                          for head in range(4)]})
    return report, samples


def prepare_cache(args: argparse.Namespace) -> dict:
    """Calibrate once, validate, and publish the reusable PT2 and reference archive.

    :param args:    Native OBB export options and common output directory.
    :returns:       Completed PT2 cache manifest.
    """
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "quantized.pt2"
    if (directory / "pt2_report.json").exists():
        report, samples = load_cache(path, args)
        graph = torch.export.load(path).module(check_guards=False)
        validate_outputs(lambda inputs: torch_predict(graph, inputs, args.io_layout), samples,
                         False, args.max_score_error, args.min_box_iou)
        print(f"Reusing calibrated PT2: {path}", flush=True)
        return report
    report = {"status": "started", "identity": cache_identity(args)}
    with export_manifest(directory / "pt2_report.json", report):
        model, samples, source = prepare_pte_source(args, None)
        report.update(source)
        options = argparse.Namespace(**vars(args))
        options.save_intermediate = False
        options.output = directory / "best.pte"
        graph = quantize_pte(model, [Path(item) for item in source["calibration_images"]],
                             source["contract"], options)
        report["quantization_validation"] = validate_outputs(
            lambda inputs: torch_predict(graph, inputs, args.io_layout), samples, False,
            args.max_score_error, args.min_box_iou)
        example = model_input(torch.from_numpy(samples[0]["input"]), args.io_layout)
        temporary = directory / "quantized.tmp.pt2"
        torch.export.save(torch.export.export(graph, (example,), strict=True), temporary)
        temporary.replace(path)
        arrays = {}
        for index, sample in enumerate(samples):
            arrays[f"input_{index}"] = sample["input"]
            arrays.update({f"reference_{index}_{head}": value
                           for head, value in enumerate(sample["reference"])})
        references = directory / "references.npz"
        np.savez_compressed(references, **arrays)
        report.update(pt2_sha256=file_hash(path), references_sha256=file_hash(references))
    print(f"Calibrated PT2 saved: {path}", flush=True)
    return report


def run_export(args: argparse.Namespace) -> dict:
    """Prepare only PT2 or compile it without loading or calibrating the PT model.

    :param args:    Unified native PT export options.
    :returns:       PT2 cache or per-memory-mode PTE report.
    """
    os.environ["PATH"] = os.pathsep.join(
        (str(Path(sys.executable).absolute().parent), os.environ["PATH"]))
    torch.set_num_threads(args.threads)
    if args.pt2_only:
        return prepare_cache(args)
    path = args.quantized_pt2.resolve()
    cached, samples = load_cache(path, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "started", "source": str(args.model),
              "source_sha256": cached["identity"]["source_sha256"],
              "source_route": "PT -> native grayscale raw OBB -> PT2E -> Ethos-U PTE",
              "contract": cached["contract"], "quantized_pt2": str(path), "pte": {},
              "configuration": {key: str(value) if isinstance(value, Path) else value
                                for key, value in vars(args).items()}}
    with export_manifest(args.output_dir / "export_report.json", report):
        (args.output_dir / "labels.txt").write_text("\n".join(cached["labels"]) + "\n")
        options = argparse.Namespace(**vars(args))
        options.reference_source = "Original PT raw OBB heads (cached)"
        graph = torch.export.load(path).module(check_guards=False)
        compile_pte_variants(graph, samples, cached["contract"], options, report["pte"])
    return report
