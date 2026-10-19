# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Verify full-output dequantization without changing integer network data."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ai_edge_litert import schema_py_generated as schema
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from executorch.exir.schema import DelegateCall, Instruction, KernelCall, Operator
import flatbuffers
import numpy as np

from test_export_model import make_tflite

from model_export.qrcode.output_io import check_pte_execution, dequantize_tflite_outputs
from model_export.qrcode.tflite_utils import inspect_tflite, lite_runner


def pack_model(model: schema.ModelT) -> bytes:
    """Serialize a mutable TFLite fixture.

    :param model:   Mutable FlatBuffer object.
    :returns:       Serialized model bytes.
    """
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def signed_heads() -> tuple[bytes, dict]:
    """Use distinct scales, nonzero zero points and every signed byte value.

    :returns:   Four-head TFLite bytes and corresponding NCHW source metadata.
    """
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(make_tflite(True), 0))
    metadata = {"input": {"shape": [1, 3, 32, 32]}, "outputs": []}
    for index, tensor in enumerate(model.subgraphs[0].tensors[1:], 1):
        tensor.quantization.scale = [0.037123 * index]
        tensor.quantization.zeroPoint = [-17 + index]
        model.buffers[index].data = np.arange(np.prod(tensor.shape), dtype=np.uint8)
        shape = list(tensor.shape)
        metadata["outputs"].append({"shape": [shape[i] for i in (0, 3, 1, 2)]
                                    if len(shape) == 4 else shape})
    return pack_model(model), metadata


class OutputDequantizationTests(unittest.TestCase):
    """Check numerical parity, original storage and strict boundary-only CPU policy."""

    def test_full_output_numerical_parity(self):
        """Compare all four model dequantizers against the existing external conversion."""
        original, metadata = signed_heads()
        with TemporaryDirectory() as temporary:
            source, destination = [Path(temporary) / name for name in ("int8.tflite", "dq.tflite")]
            source.write_bytes(original)
            dequantize_tflite_outputs(source, destination)
            # Constant-only fixtures do not exercise an XNNPACK network delegate.
            with patch("ai_edge_litert.interpreter.Interpreter", side_effect=lambda **kwargs:
                       Interpreter(**kwargs, experimental_op_resolver_type=
                                   OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)):
                integer, integer_contract = lite_runner(source, metadata, True)
                floating, float_contract = lite_runner(destination, metadata, False)
            inputs = np.zeros((1, 3, 32, 32), dtype=np.float32)
            for expected, actual in zip(integer(inputs), floating(inputs)):
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
            self.assertEqual(source.read_bytes(), original)
            self._assert_aligned_buffers(destination.read_bytes())
            self.assertEqual(float_contract["input"], integer_contract["input"])
            self.assertTrue(all(item["dtype"] == "float32" for item in float_contract["outputs"]))
            with self.assertRaises(ValueError):
                lite_runner(destination, metadata, True)

    def _assert_aligned_buffers(self, data: bytes):
        """Check byte-vector offsets without relying on private FlatBuffer accessors.

        :param data:    Serialized model with embedded buffers.
        """
        raw = np.frombuffer(data, dtype=np.uint8)
        model = schema.Model.GetRootAsModel(data, 0)
        for index in range(model.BuffersLength()):
            values = model.Buffers(index).DataAsNumpy()
            if isinstance(values, np.ndarray):
                self.assertEqual((values.ctypes.data - raw.ctypes.data) % 16, 0)

    def test_compiled_network_allows_only_terminal_dequantize(self):
        """Reject unrelated CPU work and dequantization outside final NPU outputs."""
        model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(make_tflite(True), 0))
        code = schema.OperatorCodeT()
        code.builtinCode, code.customCode = schema.BuiltinOperator.CUSTOM, b"ethos-u"
        model.operatorCodes = [code]
        operation = schema.OperatorT()
        operation.opcodeIndex, operation.inputs, operation.outputs = 0, [0], [1, 2, 3, 4]
        model.subgraphs[0].operators = [operation]
        with TemporaryDirectory() as temporary:
            source, destination = [Path(temporary) / name for name in ("npu.tflite", "dq.tflite")]
            source.write_bytes(pack_model(model))
            dequantize_tflite_outputs(source, destination)
            result = inspect_tflite(destination, require_npu=True, output_type="float32")
            self.assertEqual(result["cpu_operator_count"], 4)
            self.assertEqual(result["network_cpu_operator_count"], 0)
            with self.assertRaises(ValueError):
                inspect_tflite(destination, require_npu=True)
            malformed = schema.ModelT.InitFromObj(
                schema.Model.GetRootAsModel(destination.read_bytes(), 0))
            malformed.subgraphs[0].operators[-1].inputs = [0]
            destination.write_bytes(pack_model(malformed))
            with self.assertRaises(ValueError):
                inspect_tflite(destination, require_npu=True, output_type="float32")

    def test_extends_vela_offline_allocation(self):
        """Keep existing offsets while assigning new outputs to the online memory planner."""
        model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(make_tflite(True), 0))
        buffer = schema.BufferT()
        buffer.data = np.frombuffer(np.array([0, 1, 5, -1, 0, 16, 32, 48],
                                            dtype="<i4").tobytes(), dtype=np.uint8)
        metadata = schema.MetadataT()
        metadata.name, metadata.buffer = b"OfflineMemoryAllocation", len(model.buffers)
        model.buffers.append(buffer)
        model.metadata = [metadata]
        with TemporaryDirectory() as temporary:
            source, destination = [Path(temporary) / name for name in ("npu.tflite", "dq.tflite")]
            source.write_bytes(pack_model(model))
            dequantize_tflite_outputs(source, destination)
            result = schema.ModelT.InitFromObj(
                schema.Model.GetRootAsModel(destination.read_bytes(), 0))
            values = np.frombuffer(result.buffers[metadata.buffer].data.tobytes(), dtype="<i4")
            self.assertEqual(values.tolist(), [0, 1, 9, -1, 0, 16, 32, 48, -1, -1, -1, -1])

    def test_pte_rejects_network_fallback_and_nonterminal_dequantize(self):
        """Allow four final dequantizers without relaxing the NPU network requirement."""
        instructions = [Instruction(DelegateCall(0, [0, 1, 2, 3, 4]))]
        instructions += [Instruction(KernelCall(0, [index, index + 10]))
                         for index in range(1, 5)]
        plan = SimpleNamespace(inputs=[0], outputs=[11, 12, 13, 14],
                               delegates=[SimpleNamespace(id="EthosUBackend")],
                               chains=[SimpleNamespace(instructions=instructions)],
                               operators=[Operator("cortex_m::dequantize_per_tensor", "out")])
        self.assertEqual(check_pte_execution(plan, "float32")["boundary_dequantize_count"], 4)
        with self.assertRaises(ValueError):
            check_pte_execution(plan, "int8")
        plan.operators.append(Operator("aten::convolution", "out"))
        with self.assertRaises(ValueError):
            check_pte_execution(plan, "float32")
        plan.operators.pop()
        instructions[-1] = Instruction(KernelCall(0, [0, 14]))
        with self.assertRaises(ValueError):
            check_pte_execution(plan, "float32")


if __name__ == "__main__":
    unittest.main()
