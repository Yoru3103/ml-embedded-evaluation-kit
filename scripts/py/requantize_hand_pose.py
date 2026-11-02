# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Recalibrate the hand pose model in the existing WSL PT2E environment."""
import argparse
import importlib
from pathlib import Path
import time

torch = importlib.import_module("torch")
train_eval = importlib.import_module("torchao.quantization.pt2e")
pt2e = importlib.import_module("torchao.quantization.pt2e.quantize_pt2e")
ethosu = importlib.import_module("executorch.backends.arm.ethosu")
quantizer_module = importlib.import_module("executorch.backends.arm.quantizer")
arm_quantizer = importlib.import_module("executorch.backends.arm.quantizer.arm_quantizer")
cortex_pass = importlib.import_module("executorch.backends.cortex_m.passes.cortex_m_pass_manager")
cortex_target = importlib.import_module("executorch.backends.cortex_m.target_config")
exir = importlib.import_module("executorch.exir")
adapters = importlib.import_module("pt2pte.adapters")
calibration = importlib.import_module("pt2pte.calibration")


def calibrate(prepared: torch.nn.Module, loaded: object, options: argparse.Namespace):
    """Feed the existing image manifest through activation observers.

    :param prepared:    Model with PT2E observers.
    :param loaded:      Original model adapter and preprocessing.
    :param options:     Calibration manifest and sample count.
    """
    paths = calibration.load_image_list(options.calibration_list, count=options.images)
    started = time.monotonic()
    with torch.inference_mode():
        for index, image in enumerate(paths, 1):
            prepared(loaded.preprocess(image))
            if index % 25 == 0 or index == len(paths):
                print(f"Calibration {index}/{len(paths)} elapsed={time.monotonic()-started:.1f}s",
                      flush=True)


def validate(converted: torch.nn.Module, loaded: object, directory: Path):
    """Print a small held-out check without writing a separate report.

    :param converted:   Calibrated model.
    :param loaded:      Original model adapter and preprocessing.
    :param directory:   Prepared held-out image directory.
    """
    for name in ("user-curled-input.png", "user-open-input.png"):
        tensor = loaded.preprocess(directory / name)
        with torch.inference_mode():
            reference = loaded.model(tensor)
            actual = converted(tensor)
        reference_index = reference[1][0, 0].argmax()
        actual_index = actual[1][0, 0].argmax()
        expected = reference[2][0, :, reference_index].reshape(21, 2)
        observed = actual[2][0, :, actual_index].reshape(21, 2)
        distance = (expected - observed).square().sum(dim=1).sqrt()
        print(f"Check {name}: mean_xy={distance.mean().item():.4f} "
              f"max_xy={distance.max().item():.4f} "
              f"score={actual[1][0, 0, actual_index].item():.6f}", flush=True)


def quantize(options: argparse.Namespace, spec: ethosu.EthosUCompileSpec) -> Path:
    """Calibrate INT8 body and A16W8 head using the existing training image manifest.

    :param options: Parsed paths and sample count.
    :param spec:    U85 Dedicated256 compilation settings.
    :returns:       Saved, newly calibrated PT2E program.
    """
    loaded = adapters.load_yolo_adapter(options.weights, input_size=224, pose_output="split")
    example = torch.zeros((1, 3, 224, 224)).to(memory_format=torch.channels_last)
    exported = torch.export.export(loaded.model, (example,), strict=True)
    quantizer = quantizer_module.EthosUQuantizer(spec)
    quantizer.set_global(arm_quantizer.get_symmetric_quantization_config(is_per_channel=True))
    quantizer.set_module_name("model.22", arm_quantizer.get_symmetric_a16w8_quantization_config())
    prepared = pt2e.prepare_pt2e(exported.module(check_guards=False), quantizer)
    calibrate(prepared, loaded, options)
    converted = pt2e.convert_pt2e(prepared)
    train_eval.allow_exported_model_train_eval(converted)
    program = torch.export.export(converted, (example,), strict=True)
    destination = options.output / "hand_pose_mixed_pt2e.pt2"
    torch.export.save(program, destination)
    print(f"PT2E saved: {destination}", flush=True)
    # Console checks only; no separate quantization report is produced.
    if options.validation_directory:
        validate(converted, loaded, options.validation_directory)
    return destination


def lower(path: Path, output: Path, spec: ethosu.EthosUCompileSpec):
    """Compile the saved quantized program to the existing four-output PTE contract.

    :param path:    Quantized PT2E program.
    :param output:  Output directory for PTE and compiler intermediates.
    :param spec:    U85 Dedicated256 compilation settings.
    """
    program = torch.export.load(path)
    edge = exir.to_edge_transform_and_lower(
        program, partitioner=[ethosu.EthosUPartitioner(spec)],
        compile_config=exir.EdgeCompileConfig(_check_ir_validity=False))
    # ExecuTorch stores the transformed methods in this program registry.
    getattr(edge, "_edge_programs")["forward"] = cortex_pass.CortexMPassManager(
        edge.exported_program(),
        target_config=cortex_target.CortexMTargetConfig(cpu=cortex_target.CortexM.M85)).transform()
    executable = edge.to_executorch(
        config=exir.ExecutorchBackendConfig(extract_delegate_segments=False))
    destination = output / "hand_pose_u85-1024_ded_256KiB_mixed.pte"
    with destination.open("wb") as handle:
        executable.write_to_file(handle)
    print(f"PTE saved: {destination} bytes={destination.stat().st_size}", flush=True)


def main():
    """Run quantization and lowering, or resume lowering from a saved PT2E file."""
    parser = argparse.ArgumentParser(description=__doc__)
    original = Path("/home/mokuroo/documents/python/yolo_hand")
    toolkit = Path("/home/mokuroo/documents/python/pt2pte_toolchain")
    parser.add_argument("--weights", type=Path, default=original / "models/best.pt")
    parser.add_argument("--calibration-list", type=Path,
                        default=original / "datasets/hand_pose/calibration/train_seed42_500.txt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-directory", type=Path)
    parser.add_argument("--images", type=int, default=500)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--stage", choices=("all", "quantize", "lower"), default="all")
    parser.add_argument("--vela-ini", type=Path, default=toolkit / "configs/vela.ini")
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(options.threads)
    spec = ethosu.EthosUCompileSpec(
        target="ethos-u85-1024", system_config="Ethos_U85_SYS_DRAM_1024",
        memory_mode="Dedicated_Sram", config_ini=str(options.vela_ini),
        extra_flags=["--optimise=Performance", "--arena-cache-size=262144"])
    intermediate = options.output / "compiler"
    intermediate.mkdir(exist_ok=True)
    spec.dump_intermediate_artifacts_to(str(intermediate))
    path = options.output / "hand_pose_mixed_pt2e.pt2"
    if options.stage != "lower":
        path = quantize(options, spec)
    if options.stage != "quantize":
        lower(path, options.output, spec)


if __name__ == "__main__":
    main()
