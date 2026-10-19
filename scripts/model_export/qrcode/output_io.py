# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Retain INT8 network execution while optionally dequantizing all output heads."""

from collections import Counter
import argparse
from copy import deepcopy
from importlib import import_module
from pathlib import Path

import flatbuffers
import numpy as np


class AlignedTfliteBuilder(flatbuffers.Builder):
    """Preserve the 16-byte alignment required by embedded Ethos-U buffer data."""

    def StartVector(self, elemSize: int, numElems: int, alignment: int) -> int:
        """Align byte vectors containing command streams and weights.

        :param elemSize:    Vector element size in bytes.
        :param numElems:    Number of vector elements.
        :param alignment:   Scalar alignment requested by the generated schema.
        :returns:           FlatBuffer vector offset.
        """
        return super().StartVector(elemSize, numElems, max(alignment, 16 if elemSize == 1 else 1))


def check_pte_execution(plan: object, output_type: str = "int8") -> dict:
    """Allow CPU kernels only when they dequantize the four final outputs.

    :param plan:        Serialized forward execution plan.
    :param output_type: INT8 heads or model-side FP32 outputs.
    :returns:           Counts distinguishing boundary kernels from network fallback.
    :raises ValueError: Unexpected CPU operation, backend or output connection.
    """
    schema = import_module("executorch.exir.schema")
    if not plan.delegates or any(item.id != "EthosUBackend" for item in plan.delegates):
        raise ValueError("PTE requires Ethos-U network delegation")
    calls = [instruction.instr_args for chain in plan.chains for instruction in chain.instructions]
    kernels = [call for call in calls if isinstance(call, schema.KernelCall)]
    if output_type == "int8":
        if plan.operators or kernels:
            raise ValueError(f"PTE network contains CPU operators: {plan.operators}")
    elif output_type == "float32":
        if (len(kernels) != 4 or len(plan.outputs) != 4 or not plan.operators
                or any(item.name != "cortex_m::dequantize_per_tensor" or item.overload != "out"
                       for item in plan.operators)):
            raise ValueError("FP32 heads permit only four terminal Cortex-M dequantize kernels")
        delegated = {index for call in calls if isinstance(call, schema.DelegateCall)
                     for index in call.args if index not in plan.inputs}
        if (Counter(call.args[-1] for call in kernels) != Counter(plan.outputs)
                or any(call.args[0] not in delegated for call in kernels)):
            raise ValueError("PTE dequantize kernels must connect delegates to final outputs")
    else:
        raise ValueError(f"Unsupported output type: {output_type}")
    return {"cpu_kernel_call_count": len(kernels),
            "boundary_dequantize_count": len(kernels), "network_cpu_operator_count": 0}


def dequantize_tflite_outputs(source: Path, destination: Path) -> Path:
    """Append four DEQUANTIZE operations without modifying INT8 network tensors.

    :param source:      Fully integer raw-head TFLite, before or after Vela.
    :param destination: Separate FP32-output artifact; source is preserved.
    :returns:           Model with INT8 input and four FP32 output tensors.
    :raises ValueError: Unsupported output interface or quantization parameters.
    """
    schema = import_module("ai_edge_litert.schema_py_generated")
    if source.resolve() == destination.resolve():
        raise ValueError("FP32 output artifact must preserve the integer source model")
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(source.read_bytes(), 0))
    if any(buffer.offset or buffer.size for buffer in model.buffers):
        raise ValueError("External TFLite buffers cannot be relocated by this output wrapper")
    if len(model.subgraphs) != 1:
        raise ValueError("Expected one QR-code TFLite subgraph")
    graph = model.subgraphs[0]
    indices = list(graph.outputs)
    if (len(graph.inputs) != 1 or len(indices) != 4
            or graph.tensors[graph.inputs[0]].type != schema.TensorType.INT8):
        raise ValueError("Expected one INT8 input and four INT8 output heads")
    code = schema.OperatorCodeT()
    code.builtinCode = code.deprecatedBuiltinCode = schema.BuiltinOperator.DEQUANTIZE
    code.version = 2
    model.operatorCodes = list(model.operatorCodes or [])
    model.operatorCodes.append(code)
    graph.operators = list(graph.operators or [])
    for index in indices:
        tensor = graph.tensors[index]
        quant = tensor.quantization
        if (tensor.type != schema.TensorType.INT8 or quant is None or quant.scale is None
                or quant.zeroPoint is None):
            raise ValueError("Output dequantization requires per-tensor INT8 quantization")
        if (len(quant.scale) != 1 or len(quant.zeroPoint) != 1
                or not np.isfinite(quant.scale[0]) or quant.scale[0] <= 0):
            raise ValueError("Output dequantization requires per-tensor INT8 quantization")
        output = deepcopy(tensor)
        output.name = tensor.name + b"/dequantized"
        output.type, output.quantization, output.buffer = schema.TensorType.FLOAT32, None, 0
        graph.tensors.append(output)
        operation = schema.OperatorT()
        operation.opcodeIndex = len(model.operatorCodes) - 1
        operation.inputs, operation.outputs = [index], [len(graph.tensors) - 1]
        graph.operators.append(operation)
    graph.outputs = list(range(len(graph.tensors) - 4, len(graph.tensors)))
    for signature in model.signatureDefs or []:
        for tensor in signature.outputs:
            tensor.tensorIndex = dict(zip(indices, graph.outputs)).get(
                tensor.tensorIndex, tensor.tensorIndex)
    extend_offline_allocation(model)
    builder = AlignedTfliteBuilder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(bytes(builder.Output()))
    return destination


def extend_offline_allocation(model: object):
    """Preserve Vela offsets and let TFLM allocate the four new output buffers.

    :param model:   Mutable single-subgraph TFLite with four appended tensors.
    :raises ValueError: Unsupported or inconsistent offline allocation metadata.
    """
    for metadata in model.metadata or []:
        if metadata.name != b"OfflineMemoryAllocation":
            continue
        buffer = model.buffers[metadata.buffer]
        values = np.frombuffer(buffer.data.tobytes(), dtype="<i4").copy()
        original = len(model.subgraphs[0].tensors) - 4
        if (len(values) != original + 3 or list(values[:3]) != [0, 1, original]):
            raise ValueError("Unsupported Vela offline allocation metadata")
        values[2] += 4
        values = np.concatenate((values, np.full(4, -1, dtype="<i4")))
        buffer.data = np.frombuffer(values.astype("<i4").tobytes(), dtype=np.uint8)


def check_tflite_output_execution(model: object) -> dict:
    """Require four terminal DEQUANTIZE nodes connected directly to NPU outputs.

    :param model:   Vela-generated TFLite FlatBuffer model accessor.
    :returns:       Counts of explicit boundary dequantization operations.
    :raises ValueError: CPU fallback or incompatible tensor connections.
    """
    schema = import_module("ai_edge_litert.schema_py_generated")
    graph = model.Subgraphs(0)
    outputs = [graph.Outputs(index) for index in range(graph.OutputsLength())]
    terminals, delegated = [], set()
    for index in range(graph.OperatorsLength()):
        operation = graph.Operators(index)
        code = model.OperatorCodes(operation.OpcodeIndex())
        builtin = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
        if builtin == schema.BuiltinOperator.CUSTOM and code.CustomCode() == b"ethos-u":
            delegated.update(operation.OutputsAsNumpy().tolist())
        elif builtin == schema.BuiltinOperator.DEQUANTIZE:
            terminals.append(operation)
        else:
            raise ValueError("FP32 output mode permits only Ethos-U and terminal DEQUANTIZE")
    if (len(terminals) != 4 or len(outputs) != 4 or graph.InputsLength() != 1
            or graph.Tensors(graph.Inputs(0)).Type() != schema.TensorType.INT8):
        raise ValueError("Expected four terminal NPU-to-FP32 dequantize outputs")
    if (any(graph.Tensors(index).Type() != schema.TensorType.FLOAT32 for index in outputs)
            or Counter(op.Outputs(0) for op in terminals) != Counter(outputs)
            or any(op.InputsLength() != 1 or op.OutputsLength() != 1
                   or op.Inputs(0) not in delegated for op in terminals)):
        raise ValueError("Expected four terminal NPU-to-FP32 dequantize outputs")
    return {"boundary_dequantize_count": 4, "network_cpu_operator_count": 0}


def main():
    """Wrap an existing integer TFLite without recalibration or NPU recompilation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inspection = import_module("model_export.qrcode.tflite_utils").inspect_tflite
    original = inspection(args.model)
    result = dequantize_tflite_outputs(args.model, args.output)
    print(inspection(result, require_npu=bool(original["npu_delegate_count"]),
                     output_type="float32"))


if __name__ == "__main__":
    main()
