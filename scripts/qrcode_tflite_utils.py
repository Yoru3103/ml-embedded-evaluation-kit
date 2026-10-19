# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Inspect actual TFLite operators and validate integer tensor interfaces."""

import argparse
from collections import Counter
from importlib import import_module
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

from ethosu.vela.tflite.BuiltinOperator import BuiltinOperator
from ethosu.vela.tflite.Model import Model
from ethosu.vela.tflite.TensorType import TensorType
import numpy as np

from qrcode_output_io import check_tflite_output_execution, dequantize_tflite_outputs

_OPERATOR_NAMES = {value: name for name, value in vars(BuiltinOperator).items()
                   if isinstance(value, int)}


def export_litert_models(model: object, calibration: list[np.ndarray], directory: Path,
                         layout: str = "nhwc", float_io: bool = False) -> tuple[Path, Path]:
    """Convert and calibrate a PyTorch model using the shared deployment interface.

    :param model:       Source module with NCHW computation semantics.
    :param calibration: Normalized NCHW representative images.
    :param directory:   Destination directory.
    :param layout:      Deployment image and feature-map layout.
    :param float_io:    Retain float boundaries for the legacy PT application.
    :returns:           Floating and integer TFLite paths.
    """
    layouts = import_module("model_layout")
    directory.mkdir(parents=True, exist_ok=True)
    float_path, int8_path = directory / "best_float32.tflite", directory / "best_int8.tflite"
    if layout == "nhwc":
        model = layouts.ChannelLastIO(model).eval()
    tensors = [layouts.model_input(import_module("torch").from_numpy(item), layout)
               for item in calibration]
    edge = import_module("litert_torch").convert(model, (tensors[0],))
    edge.export(str(float_path))
    recipe = import_module("ai_edge_quantizer.recipe")
    compiler = import_module("ai_edge_quantizer.quantizer").Quantizer(str(float_path))
    compiler.load_quantization_recipe(recipe.static_wi8_ai8())
    if float_io:
        operations = import_module("ai_edge_quantizer.qtyping").TFLOperationName
        for operation in (operations.INPUT, operations.OUTPUT):
            compiler.update_quantization_recipe(regex=".*", operation_name=operation,
                                                algorithm_key=recipe.AlgorithmName.NO_QUANTIZE)
    result = compiler.calibrate({"serving_default": [{"args_0": item.numpy()} for item in tensors]})
    compiler.quantize(calibration_result=result).export_model(str(int8_path), overwrite=True)
    return float_path, int8_path


def validate_compiled_boundaries(model: Any, counts: dict, allow_float_boundaries: bool):
    """Require NPU execution with only the explicitly requested CPU boundaries.

    :param model:                   Parsed FlatBuffer model.
    :param counts:                  Operator counts.
    :param allow_float_boundaries:  Require one Q and four DQ operators with float I/O.
    """
    cpu = {name: count for name, count in counts.items() if name != "ethos-u"}
    expected_cpu = {"QUANTIZE": 1, "DEQUANTIZE": 4} if allow_float_boundaries else {}
    if cpu != expected_cpu or not counts.get("ethos-u", 0):
        raise ValueError(f"Compiled TFLite violates the requested NPU boundary policy: {counts}")
    graph = model.Subgraphs(0)
    indices = [graph.Inputs(index) for index in range(graph.InputsLength())]
    indices += [graph.Outputs(index) for index in range(graph.OutputsLength())]
    expected_type = TensorType.FLOAT32 if allow_float_boundaries else TensorType.INT8
    if not indices or any(graph.Tensors(index).Type() != expected_type for index in indices):
        raise ValueError("Compiled TFLite does not match the requested boundary tensor type")


def inspect_tflite(path: Path, require_npu: bool = False, output_type: str = "int8",
                   allow_float_boundaries: bool = False) -> dict:
    """Count executed FlatBuffer operators, ignoring unused opcode-table entries.

    :param path:                    TFLite file to inspect.
    :param require_npu:             Validate the requested NPU boundary policy.
    :param output_type:             Permit terminal DEQUANTIZE for FP32 outputs.
    :param allow_float_boundaries:  Permit one Q and four DQ nodes with float I/O.
    :returns:                       Actual operator counts and partition summary.
    """
    if not Model.ModelBufferHasIdentifier(path.read_bytes(), 0):
        raise ValueError("Invalid TFLite FlatBuffer identifier")
    model = Model.GetRootAsModel(path.read_bytes(), 0)
    counts = Counter()
    for index in range(model.SubgraphsLength()):
        graph = model.Subgraphs(index)
        for operation in range(graph.OperatorsLength()):
            code = model.OperatorCodes(graph.Operators(operation).OpcodeIndex())
            name = (code.CustomCode().decode() if code.CustomCode()
                    else _OPERATOR_NAMES.get(max(code.BuiltinCode(), code.DeprecatedBuiltinCode()),
                                   str(code.BuiltinCode())))
            counts[name] += 1
    cpu = {name: count for name, count in counts.items() if name != "ethos-u"}
    npu = counts.get("ethos-u", 0)
    if require_npu and allow_float_boundaries:
        validate_compiled_boundaries(model, counts, True)
    if require_npu and not allow_float_boundaries and (not npu or (cpu and output_type == "int8")):
        raise ValueError(f"TFLite must contain only Ethos-U operators: {dict(counts)}")
    boundary = {}
    if require_npu and not allow_float_boundaries and output_type == "float32":
        if cpu != {"DEQUANTIZE": 4} or model.SubgraphsLength() != 1:
            raise ValueError("FP32 output mode requires exactly four terminal DEQUANTIZE nodes")
        boundary = check_tflite_output_execution(model)
    elif require_npu and not allow_float_boundaries:
        graph = model.Subgraphs(0)
        indices = [graph.Inputs(index) for index in range(graph.InputsLength())]
        indices += [graph.Outputs(index) for index in range(graph.OutputsLength())]
        if not indices or any(graph.Tensors(index).Type() != TensorType.INT8 for index in indices):
            raise ValueError("Compiled NPU-only TFLite requires INT8 inputs and outputs")
    return {"operators": dict(counts), "cpu_operators": cpu,
            "cpu_operator_count": sum(cpu.values()), "npu_delegate_count": npu, **boundary}


def tensor_metadata(details: dict) -> dict:
    """Read an interpreter tensor's layout and scalar quantization parameters.

    :param details: Tensor details from the LiteRT interpreter.
    :returns:       JSON-compatible tensor metadata.
    """
    scale, zero_point = details["quantization"]
    dtype = details["dtype"]
    if dtype not in (np.float32, np.int8) or (dtype == np.int8 and scale <= 0):
        raise ValueError("Expected float32 or per-tensor INT8 model input/output")
    return {"name": details["name"], "shape": details["shape"].tolist(),
            "dtype": np.dtype(dtype).name, "scale": float(scale), "zero_point": int(zero_point)}


def dequantize(value: np.ndarray, details: dict) -> np.ndarray:
    """Recover real values using each output's own quantization parameters.

    :param value:   Interpreter output tensor.
    :param details: Tensor details with scalar quantization parameters.
    :returns:       Float32 raw logits or DFL values.
    """
    if value.dtype == np.float32:
        return value
    scale, zero_point = details["quantization"]
    return (value.astype(np.float32) - zero_point) * scale


def lite_runner(path: Path, metadata: dict, require_int8: bool) -> tuple[Callable, dict]:
    """Bind NCHW or NHWC LiteRT tensors to the original ordered ONNX outputs.

    :param path:            Converted or externally supplied TFLite model.
    :param metadata:        Source ONNX input and ordered output shapes.
    :param require_int8:    Reject float interfaces when integer execution is required.
    :returns:               Prediction callable and checked tensor interface.
    """
    interpreter = create_interpreter(path)
    interpreter.allocate_tensors()
    inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 4:
        raise ValueError("Expected one LiteRT input and four raw outputs")
    tensors = [tensor_metadata(item) for item in inputs + outputs]
    if require_int8 and any(item["dtype"] != "int8" for item in tensors):
        raise ValueError("NPU-only model requires INT8 inputs and outputs")
    source_shape = metadata["input"]["shape"]
    nhwc = list(inputs[0]["shape"]) == [source_shape[index] for index in (0, 2, 3, 1)]
    if not nhwc and list(inputs[0]["shape"]) != source_shape:
        raise ValueError("LiteRT input shape does not match ONNX")
    bindings = []
    for item in metadata["outputs"]:
        shape = item["shape"]
        matches = [(details, False) for details in outputs if list(details["shape"]) == shape]
        if len(shape) == 4:
            matches += [(details, True) for details in outputs
                        if list(details["shape"]) == [shape[index] for index in (0, 2, 3, 1)]]
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely match LiteRT output to ONNX shape {shape}")
        bindings.append(matches[0])

    def predict(value: np.ndarray) -> list[np.ndarray]:
        """Quantize the input, invoke LiteRT, and restore NCHW floating outputs.

        :param value:   Float32 RGB NCHW /255 sample.
        :returns:       Outputs dequantized and ordered like ONNX.
        """
        if nhwc:
            value = value.transpose(0, 2, 3, 1)
        if inputs[0]["dtype"] == np.int8:
            scale, zero_point = inputs[0]["quantization"]
            value = np.clip(np.rint(value / scale) + zero_point, -128, 127).astype(np.int8)
        interpreter.set_tensor(inputs[0]["index"], np.ascontiguousarray(value))
        interpreter.invoke()
        result = []
        for details, transpose in bindings:
            output = dequantize(interpreter.get_tensor(details["index"]), details)
            result.append(output.transpose(0, 3, 1, 2) if transpose else output)
        return result

    return predict, {"input": tensors[0], "input_layout": "NHWC" if nhwc else "NCHW",
                     "outputs": [tensor_metadata(item) for item, _ in bindings]}


def compile_tflite(args: argparse.Namespace, source: Path | None = None,
                   label: str = "tflm", require_npu: bool = True,
                   allow_float_boundaries: bool = False) -> dict:
    """Run Vela with the same target and memory settings used by the PTE backend.

    :param args:                    Export and Vela options for this variant.
    :param source:                  Optional external reference TFLite model.
    :param label:                   Output-directory prefix for this compilation.
    :param require_npu:             Validate the requested NPU boundary policy.
    :param allow_float_boundaries:  Permit one Q and four DQ nodes with float I/O.
    :returns:                       Compiled file, checks and compiler command.
    """
    source = source or args.output_dir / "tflite/best_int8.tflite"
    directory = args.output_dir / f"{label}_{args.memory_mode}"
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "ethosu.vela",
               str(source),
               "--accelerator-config", args.target, "--config", str(args.vela_config),
               "--system-config", args.system_config, "--memory-mode", args.memory_mode,
               "--output-dir", str(directory), *args.extra_flag]
    if args.memory_mode == "Dedicated_Sram":
        command.extend(["--arena-cache-size", str(args.arena_cache_size)])
    log = directory / "vela.log"
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    path = directory / f"{source.stem}_vela.tflite"
    if not path.is_file():
        raise RuntimeError("Vela did not produce the expected TFLite model")
    if not allow_float_boundaries and getattr(args, "output_type", "int8") == "float32":
        # Preserve Vela's integer command stream and add runtime output boundaries afterwards.
        inspect_tflite(path, require_npu=True)
        path = dequantize_tflite_outputs(
            path, directory / f"{source.stem}_float_outputs_vela.tflite")
    operators = inspect_tflite(path, require_npu=require_npu,
                               output_type=getattr(args, "output_type", "int8"),
                               allow_float_boundaries=allow_float_boundaries)
    print(f"TFLite {args.memory_mode}: {operators}; log={log}", flush=True)
    return {"path": str(path), "bytes": path.stat().st_size, "command": command,
            "log": str(log), **operators,
            "summaries": [str(item) for item in directory.glob("*.csv")]}


def check_tflite_layout(contract: dict, metadata: dict, layout: str | None):
    """Reject converter interfaces that differ from the requested deployment layout.

    :param contract:    Actual source-ordered TFLite tensor interface.
    :param metadata:    Original NCHW tensor shapes.
    :param layout:      Required deployment layout, or no explicit requirement.
    :raises ValueError: Unexpected input or feature-map layout.
    """
    if not layout:
        return
    if contract["input_layout"] != layout.upper():
        raise ValueError(f"TFLite input layout must be {layout.upper()}")
    for source, actual in zip(metadata["outputs"], contract["outputs"]):
        shape = source["shape"]
        expected = ([shape[index] for index in (0, 2, 3, 1)]
                    if layout == "nhwc" and len(shape) == 4 else shape)
        if actual["shape"] != expected:
            raise ValueError(f"TFLite feature-map layout must be {layout.upper()}")


def validate_tflite_models(float_path: Path, int8_path: Path, samples: list[dict],
                           metadata: dict, args: argparse.Namespace) -> dict:
    """Apply the same tensor and numerical checks to either TFLite converter.

    :param float_path:  Floating-point TFLite model.
    :param int8_path:   Fully integer TFLite model.
    :param samples:     Original ONNX predictions.
    :param metadata:    Original input and ordered outputs.
    :param args:        Quantization acceptance limits.
    :returns:           Artifact paths, tensor interfaces and validation metrics.
    """
    validate_outputs = import_module("qrcode_onnx_utils").validate_outputs
    result = {"validation_scope": "ONNX versus host LiteRT; compiled NPU execution untested"}
    if getattr(args, "output_type", "int8") == "float32":
        int8_path = dequantize_tflite_outputs(
            int8_path, int8_path.with_name(f"{int8_path.stem}_float_outputs.tflite"))
    for label, path, integer in (("float32", float_path, False), ("int8", int8_path, True)):
        predict, contract = lite_runner(
            path, metadata, integer and getattr(args, "output_type", "int8") == "int8")
        if integer:
            required = getattr(args, "output_type", "int8")
            if (contract["input"]["dtype"] != "int8"
                    or any(item["dtype"] != required for item in contract["outputs"])):
                raise ValueError("Quantized model must retain INT8 input and requested output type")
        check_tflite_layout(contract, metadata, getattr(args, "io_layout", None))
        result[label] = {"path": str(path), "tensor_contract": contract,
                         "validation": validate_outputs(predict, samples, not integer,
                                                        args.max_score_error, args.min_box_iou)}
    return result


def create_interpreter(path: Path) -> Any:
    """Load an available LiteRT or TensorFlow interpreter only when needed.

    :param path:    TFLite model to execute on the host.
    :returns:       Interpreter with four host worker threads.
    """
    try:
        interpreter_type = import_module("ai_edge_litert.interpreter").Interpreter
    except ModuleNotFoundError:
        interpreter_type = import_module("tensorflow").lite.Interpreter
    return interpreter_type(model_path=str(path), num_threads=4)
