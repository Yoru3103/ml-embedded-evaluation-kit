# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Inspect actual TFLite operators and validate integer tensor interfaces."""

from collections import Counter
from pathlib import Path
from typing import Callable

from ai_edge_litert.interpreter import Interpreter
from ethosu.vela.tflite.BuiltinOperator import BuiltinOperator
from ethosu.vela.tflite.Model import Model
from ethosu.vela.tflite.TensorType import TensorType
import numpy as np


def inspect_tflite(path: Path, require_npu: bool = False) -> dict:
    """Count executed FlatBuffer operators, ignoring unused opcode-table entries.

    :param path:        TFLite file to inspect.
    :param require_npu: Reject every operator except the Ethos-U custom operator.
    :returns:           Actual operator counts and NPU/CPU partition summary.
    """
    data = path.read_bytes()
    if not Model.ModelBufferHasIdentifier(data, 0):
        raise ValueError("Invalid TFLite FlatBuffer identifier")
    model = Model.GetRootAsModel(data, 0)
    names = {value: name for name, value in vars(BuiltinOperator).items()
             if isinstance(value, int)}
    counts = Counter()
    for index in range(model.SubgraphsLength()):
        graph = model.Subgraphs(index)
        for operation in range(graph.OperatorsLength()):
            code = model.OperatorCodes(graph.Operators(operation).OpcodeIndex())
            builtin = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
            name = (code.CustomCode().decode() if builtin == BuiltinOperator.CUSTOM
                    else names.get(builtin, str(builtin)))
            counts[name] += 1
    cpu = {name: count for name, count in counts.items() if name != "ethos-u"}
    npu = counts.get("ethos-u", 0)
    if require_npu and (cpu or not npu):
        raise ValueError(f"TFLite must contain only Ethos-U operators: {dict(counts)}")
    if require_npu:
        graph = model.Subgraphs(0)
        indices = [graph.Inputs(index) for index in range(graph.InputsLength())]
        indices += [graph.Outputs(index) for index in range(graph.OutputsLength())]
        if not indices or any(graph.Tensors(index).Type() != TensorType.INT8 for index in indices):
            raise ValueError("Compiled NPU-only TFLite requires INT8 inputs and outputs")
    return {"operators": dict(counts), "cpu_operators": cpu,
            "cpu_operator_count": sum(cpu.values()), "npu_delegate_count": npu}


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

    :param value:       Interpreter output tensor.
    :param details:     Tensor details with scalar quantization parameters.
    :returns:           Float32 raw logits or DFL values.
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
    interpreter = Interpreter(model_path=str(path), num_threads=4)
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
