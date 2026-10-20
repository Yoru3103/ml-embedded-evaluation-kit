# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Shared QR-code PTE preparation, compilation and LiteRT conversion."""

import argparse
from copy import deepcopy
from pathlib import Path

from executorch.backends.arm.ethosu import EthosUPartitioner, EthosUCompileSpec
from executorch.backends.cortex_m.passes.replace_quant_nodes_pass import ReplaceQuantNodesPass
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir._serialize import _deserialize_pte_binary
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.exir.passes.quantize_io_pass import QuantizeInputs, QuantizeOutputs
import numpy as np
import torch

from model_export.common.quantization import make_compile_spec, quantize
from model_export.common.reporting import save_report
from model_export.qrcode.onnx_utils import adapt_model, check_pte_contract, validate_outputs
from model_export.qrcode.tflite_utils import export_litert_models, validate_tflite_models
from model_export.common.layout import ChannelLastIO, model_input
from model_export.qrcode.pte_passes import separate_quantized_relu
from model_export.qrcode.output_io import check_pte_execution


def torch_predict(model: torch.nn.Module, inputs: np.ndarray,
                  layout: str = "nchw") -> list[np.ndarray]:
    """Run a four-output PyTorch model without autograd state.

    :param model:   Floating-point or calibrated PT2E graph.
    :param inputs:  RGB NCHW data normalized by 255.
    :param layout:  Actual model interface layout.
    :returns:       Outputs in original ONNX order.
    """
    with torch.no_grad():
        outputs = model(model_input(torch.from_numpy(inputs), layout))
        return [(value.permute(0, 3, 1, 2) if layout == "nhwc" and value.ndim == 4
                 else value).detach().numpy() for value in outputs]


def prepare_model(graph: object, samples: list[dict]) -> tuple[torch.nn.Module, list[dict]]:
    """Adapt an ONNX graph and check the captured PyTorch outputs before quantization.

    :param graph:   Original static ONNX graph.
    :param samples: ONNX predictions and matching validation inputs.
    :returns:       Adapted model and floating-point parity checks.
    """
    model = adapt_model(graph, torch.from_numpy(samples[0]["input"]))
    return model, validate_native_capture(model, samples)


def validate_native_capture(model: torch.nn.Module, samples: list[dict]) -> list[dict]:
    """Check native torch.export capture against the supplied raw floating outputs.

    :param model:   PyTorch model exposing four raw OBB outputs.
    :param samples: Reference inputs and outputs from the original source.
    :returns:       Per-sample floating capture parity results.
    """
    exported = torch.export.export(model, (torch.from_numpy(samples[0]["input"]),), strict=True)
    return validate_outputs(
        lambda inputs: torch_predict(exported.module(check_guards=False), inputs), samples, True)


def lower_integer_pte(quantized: torch.fx.GraphModule, example: torch.Tensor,
                      spec: EthosUCompileSpec, alignment: int,
                      output_type: str = "int8") -> tuple[object, dict]:
    """Remove floating I/O boundaries and retain serialized quantization metadata.

    :param quantized:   Calibrated PT2E model.
    :param example:     Floating input for graph capture.
    :param spec:        Ethos-U compiler specification.
    :param alignment:   Runtime tensor memory alignment in bytes.
    :param output_type: INT8 heads or model-side dequantized FP32 heads.
    :returns:           Executable program and integer I/O quantization parameters.
    """
    exported = torch.export.export(quantized, (example,), strict=True)
    separate_quantized_relu(exported.graph_module)
    edge = to_edge_transform_and_lower(exported, partitioner=[EthosUPartitioner(spec)],
                                      compile_config=EdgeCompileConfig(_check_ir_validity=False))
    inputs, outputs = QuantizeInputs(edge, [0]), QuantizeOutputs(edge, list(range(4)))
    if output_type == "int8":
        edge = edge.transform([inputs, outputs])
    elif output_type == "float32":
        outputs.dequant_args = {index: node.args[1:] for index, node in enumerate(
            edge.exported_program().graph_module.graph.output_node().args[0])}
        edge = edge.transform([inputs, ReplaceQuantNodesPass()])
    else:
        raise ValueError(f"Unsupported PTE output type: {output_type}")
    qparams = {}
    for label, values in (("inputs", inputs.quant_args), ("outputs", outputs.dequant_args)):
        qparams[label] = [{"scale": float(scale), "zero_point": int(zero_point),
                           "quant_min": minimum, "quant_max": maximum, "dtype": str(dtype)}
                          for _, (scale, zero_point, minimum, maximum, dtype)
                          in sorted(values.items())]
    program = edge.to_executorch(config=ExecutorchBackendConfig(
        extract_delegate_segments=False,
        memory_planning_pass=MemoryPlanningPass(alignment=alignment)))
    return program, qparams


def write_pte(program: object, path: Path, metadata: dict, layout: str = "nchw",
              output_type: str = "int8"):
    """Atomically save a PTE after checking the serialized forward plan.

    :param program:     Lowered ExecuTorch program.
    :param path:        Destination file.
    :param metadata:    Source tensor layouts.
    :param layout:      Deployment input and feature-map layout.
    :param output_type: INT8 heads or model-side dequantized FP32 heads.
    """
    temporary = path.with_suffix(".pte.tmp")
    try:
        with temporary.open("wb") as stream:
            program.write_to_file(stream)
        serialized = _deserialize_pte_binary(temporary.read_bytes()).program
        plan = next(item for item in serialized.execution_plan if item.name == "forward")
        check_pte_contract(plan, metadata, layout, output_type)
        check_pte_execution(plan, output_type)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def quantize_pte(model: torch.nn.Module, paths: list[Path], metadata: dict,
                 args: argparse.Namespace) -> torch.fx.GraphModule:
    """Prepare the layout and calibrate through one path for direct and cached exports.

    :param model:       Floating native PT or adapted ONNX model with NCHW interfaces.
    :param paths:       Selected calibration images.
    :param metadata:    Original tensor contract.
    :param args:        Layout, quantization and intermediate-output options.
    :returns:           Calibrated PT2E graph ready for serialization or compilation.
    """
    options = argparse.Namespace(**vars(args))
    if options.memory_mode == "all":
        options.memory_mode = "Dedicated_Sram"
    options.input_channels = metadata["input_channels"]
    input_shape = metadata["input"]["shape"]
    options.image_size = (input_shape[3], input_shape[2])
    if not hasattr(options, "output"):
        options.output = options.output_dir / f"pte_{options.memory_mode}/best.pte"
    options.output.parent.mkdir(parents=True, exist_ok=True)
    if getattr(options, "io_layout", "nchw") == "nhwc":
        model = ChannelLastIO(model).eval()
    return quantize(model, paths, options, make_compile_spec(options))




def compile_pte_variants(quantized: torch.fx.GraphModule, samples: list[dict],
                         metadata: dict, args: argparse.Namespace, reports: dict) -> dict:
    """Compile independent memory variants from one calibrated or reloaded graph.

    :param quantized:   Calibrated PT2E graph, preserved across compiler mutations.
    :param samples:     Source reference predictions.
    :param metadata:    Original tensor contract.
    :param args:        Requested memory modes and compiler options.
    :param reports:     Mutable report retaining each completed variant on failure.
    :returns:           Per-memory-mode deployment reports.
    """
    modes = (["Dedicated_Sram", "Shared_Sram"] if args.memory_mode == "all"
             else [args.memory_mode])
    for mode in modes:
        options = argparse.Namespace(**vars(args))
        options.memory_mode = mode
        options.output = args.output_dir / f"pte_{mode}/best.pte"
        reports[mode] = compile_quantized_pte(deepcopy(quantized), samples, metadata, options)
    return reports


def compile_quantized_pte(quantized: torch.fx.GraphModule, samples: list[dict],
                          metadata: dict, args: argparse.Namespace) -> dict:
    """Compile an already calibrated graph, retaining NPU-only integer I/O checks.

    :param quantized:   Validated PT2E graph reused across fast-cache capacities.
    :param samples:     Source ONNX predictions for host validation.
    :param metadata:    Original tensor interface.
    :param args:        Compiler, destination and memory options.
    :returns:           Host validation and serialized deployment report.
    """
    spec = make_compile_spec(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    layout = getattr(args, "io_layout", "nchw")
    results = validate_outputs(lambda inputs: torch_predict(quantized, inputs, layout),
                               samples, False, args.max_score_error, args.min_box_iou)
    example = model_input(torch.from_numpy(samples[0]["input"]), layout)
    program, qparams = lower_integer_pte(quantized, example,
                                        spec, args.memory_alignment,
                                        getattr(args, "output_type", "int8"))
    plan = program.executorch_program.execution_plan[0]
    contract = check_pte_contract(plan, metadata, layout, getattr(args, "output_type", "int8"))
    execution = check_pte_execution(plan, getattr(args, "output_type", "int8"))
    buffers = list(plan.non_const_buffer_sizes)
    if args.max_planned_memory is not None and sum(buffers) > args.max_planned_memory:
        raise RuntimeError(f"PTE planned memory {sum(buffers)} exceeds the requested budget")
    write_pte(program, args.output, metadata, layout, getattr(args, "output_type", "int8"))
    report = {"path": str(args.output), "bytes": args.output.stat().st_size,
              "memory_mode": args.memory_mode, "serialized_tensor_layouts": contract,
              "io_layout": layout.upper(),
              "output_type": getattr(args, "output_type", "int8"),
              "planned_buffer_bytes": buffers, "planned_total_bytes": sum(buffers),
              "npu_delegate_count": len(plan.delegates),
              "cpu_operator_count": len(plan.operators),
              "cpu_operators": [item.name for item in plan.operators],
              "execution": execution,
              "io_quantization": qparams,
              "validation": results,
              "validation_scope": f"{getattr(args, 'reference_source', 'ONNX')} versus host "
                                  "PT2E; compiled NPU execution untested"}
    save_report(args.output, report)
    return report


def export_tflite(model: torch.nn.Module, calibration: list[np.ndarray],
                  samples: list[dict], metadata: dict, args: argparse.Namespace) -> dict:
    """Export FP32 and static INT8 LiteRT models using the same image preprocessing.

    :param model:       Native PT or adapted ONNX graph exposing raw OBB heads.
    :param calibration: Representative RGB NCHW samples.
    :param samples:     Cached ONNX predictions for validation.
    :param metadata:    Source tensor shapes and output order.
    :param args:        Export options.
    :returns:           FP32 and host INT8 validation results.
    """
    directory = args.output_dir / "tflite"
    layout = getattr(args, "io_layout", "nchw")
    float_path, int8_path = export_litert_models(model, calibration, directory, layout)
    report = validate_tflite_models(float_path, int8_path, samples, metadata, args)
    report["backend"] = "litert"
    return report
