#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Recompile validated QR-code INT8 graphs for five Dedicated SRAM cache capacities."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from export_qrcode_onnx import make_samples, write_manifest
from qrcode_export_backends import compile_quantized_pte
from qrcode_contract import describe_model
from qrcode_tflite_utils import inspect_tflite

ROOT = Path(__file__).resolve().parents[1]
CACHE_SIZES = (64, 128, 256, 384, 768)


@dataclass
class CompilationContext:
    """Fixed graph, tensor interface and validation data shared by cache variants."""

    options: argparse.Namespace
    samples: list[dict]
    metadata: dict
    quantized: torch.fx.GraphModule | None


def parse_args() -> argparse.Namespace:
    """Select validated source graphs and Dedicated SRAM cache sizes in KiB.

    :returns:   Compilation options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path,
                        default=ROOT / "resources_downloaded/yolo_QRcode")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--format", choices=("both", "tflm", "pte"), default="both")
    parser.add_argument("--cache-kib", type=int, choices=CACHE_SIZES,
                        nargs="+", default=CACHE_SIZES)
    args = parser.parse_args()
    args.model_root = args.model_root.resolve()
    args.output_root = (args.output_root or args.model_root / "fastcache_dedicated").resolve()
    return args


def compile_tflm(source: Path, directory: Path, options: argparse.Namespace) -> dict:
    """Compile the same INT8 TFLite graph with a matching Vela fast-cache budget.

    :param source:      Validated integer-I/O TFLite graph.
    :param directory:   Per-cache destination.
    :param options:     Matched Vela settings.
    :returns:           File and actual NPU-only operator checks.
    """
    command = [sys.executable, "-m", "ethosu.vela", str(source),
               "--config", str(options.vela_config), "--accelerator-config", options.target,
               "--system-config", options.system_config, "--memory-mode", "Dedicated_Sram",
               "--arena-cache-size", str(options.arena_cache_size),
               "--optimise", "Performance", "--cop-format", "COP1", "--verbose-performance",
               "--output-dir", str(directory)]
    with (directory / "compile.log").open("w", encoding="utf-8") as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    model = directory / f"{source.stem}_vela.tflite"
    return {"path": str(model), "command": command, **inspect_tflite(model, require_npu=True)}


def compile_case(framework: str, cache_kib: int, args: argparse.Namespace,
                 context: CompilationContext):
    """Write an invalidated completion record before each NPU-only compilation.

    :param framework:   PTE or TFLM.
    :param cache_kib:   Dedicated SRAM cache capacity.
    :param args:        Source and matrix output directories.
    :param context:     Compiler settings, validated graph and host samples.
    """
    directory = args.output_root / framework / f"{cache_kib}k"
    directory.mkdir(parents=True, exist_ok=True)
    options = context.options
    options.arena_cache_size = cache_kib * 1024
    options.output = directory / "best.pte"
    options.extra_flag = ["--optimise=Performance", "--cop-format=COP1", "--verbose-performance"]
    source = (args.model_root / "tflite/best_int8.tflite" if framework == "tflm"
              else args.model_root / "pte_Dedicated_Sram/best_graphs/quantized.pt2")
    record = {"status": "started", "framework": framework, "cache_bytes": cache_kib * 1024,
              "target": options.target, "system_config": options.system_config,
              "memory_mode": "Dedicated_Sram", "image_size": context.metadata["image_size"],
              "num_classes": context.metadata["num_classes"], "source_path": str(source),
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "vela_config_sha256": hashlib.sha256(options.vela_config.read_bytes()).hexdigest()}
    completion = directory / "compilation.json"
    write_manifest(completion, record)
    print(f"Compile QR-code {framework} / {cache_kib} KiB", flush=True)
    try:
        result = (compile_tflm(source, directory, options) if framework == "tflm"
                  else compile_quantized_pte(context.quantized, context.samples,
                                             context.metadata, options))
        record.update(result)
        record["model_sha256"] = hashlib.sha256(Path(result["path"]).read_bytes()).hexdigest()
        record["status"] = "completed"
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        write_manifest(completion, record)


def main():
    """Keep calibration fixed while varying only the Dedicated SRAM compilation budget."""
    args = parse_args()
    report = json.loads((args.model_root / "export_report.json").read_text())
    if report["status"] != "completed":
        raise ValueError("Complete export_qrcode_onnx.py validation before cache compilation")
    options = argparse.Namespace(**report["configuration"])
    options.model, options.vela_config = Path(options.model), Path(options.vela_config)
    if hashlib.sha256(options.model.read_bytes()).hexdigest() != report["source_sha256"]:
        raise ValueError("ONNX source changed since the validated export")
    options.image_size = report["contract"]["image_size"]
    options.memory_mode = "Dedicated_Sram"
    os.environ["PATH"] = os.pathsep.join(
        (str(Path(sys.executable).absolute().parent), os.environ["PATH"]))
    torch.set_num_threads(options.threads)
    _, metadata = describe_model(options.model)
    samples = make_samples([Path(path) for path in report["validation_images"]], metadata, options)
    quantized = None
    if args.format in ("both", "pte"):
        quantized = torch.export.load(
            args.model_root / "pte_Dedicated_Sram/best_graphs/quantized.pt2"
        ).module(check_guards=False)
    context = CompilationContext(options, samples, metadata, quantized)
    for framework in (("tflm", "pte") if args.format == "both" else (args.format,)):
        for cache_kib in args.cache_kib:
            compile_case(framework, cache_kib, args, context)


if __name__ == "__main__":
    main()
