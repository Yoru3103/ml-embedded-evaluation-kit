# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for constant specialization and four-output serialization."""

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ai_edge_litert import schema_py_generated as schema
import flatbuffers
import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort
import torch
from PIL import Image
from executorch.exir.schema import ScalarType

from qrcode_onnx_utils import (
    adapt_model, check_pte_contract, compare_tensors, detection_metrics, validate_outputs)
from qrcode_tflite_utils import inspect_tflite
from yolo_pte_utils import load_image


def make_graph() -> onnx.ModelProto:
    """Build an in-memory graph exercising the model's non-exportable ONNX operators.

    :returns:   Fixed Resize/Slice/Reshape/MatMul/Expand/SE multiplication graph.
    """
    arrays = {"roi": np.array([], dtype=np.float32),
              "scales": np.array([1, 1, 2, 2], dtype=np.float32),
              "starts": np.array([0], dtype=np.int64), "ends": np.array([2], dtype=np.int64),
              "axes": np.array([1], dtype=np.int64),
              "flat": np.array([1, 2], dtype=np.int64),
              "weight": np.array([[1, 2], [3, 4]], dtype=np.float32),
              "se_shape": np.array([1, 2, 1, 1], dtype=np.int64),
              "full_shape": np.array([1, 2, 8, 8], dtype=np.int64)}
    nodes = [helper.make_node("Resize", ["input", "roi", "scales"], ["resized"],
                              mode="nearest", nearest_mode="floor",
                              coordinate_transformation_mode="asymmetric"),
             helper.make_node("Slice", ["resized", "starts", "ends", "axes"], ["sliced"]),
             helper.make_node("GlobalAveragePool", ["sliced"], ["pooled"]),
             helper.make_node("Reshape", ["pooled", "flat"], ["flat_input"]),
             helper.make_node("MatMul", ["flat_input", "weight"], ["projected"]),
             helper.make_node("Reshape", ["projected", "se_shape"], ["se"]),
             helper.make_node("Expand", ["se", "full_shape"], ["expanded"]),
             helper.make_node("Mul", ["sliced", "expanded"], ["raw_output"]),
             helper.make_node("Relu", ["raw_output"], ["output"])]
    graph = helper.make_graph(nodes, "static_se",
                              [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT,
                                                             [1, 3, 4, 4])],
                              [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT,
                                                             [1, 2, 8, 8])],
                              [numpy_helper.from_array(value, name)
                               for name, value in arrays.items()])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)], ir_version=8)


def make_flatbuffer(cpu: bool) -> bytes:
    """Create a TFLite fixture with both used and unused opcode-table entries.

    :param cpu: Include a live CPU operator when true.
    :returns:   Model containing an Ethos-U operation and an unused ADD opcode.
    """
    model = schema.ModelT()
    model.version = 3
    npu, add = schema.OperatorCodeT(), schema.OperatorCodeT()
    npu.builtinCode, npu.customCode = schema.BuiltinOperator.CUSTOM, b"ethos-u"
    add.builtinCode = schema.BuiltinOperator.ADD
    model.operatorCodes = [npu, add]
    graph = schema.SubGraphT()
    tensor = schema.TensorT()
    tensor.type, tensor.shape = schema.TensorType.INT8, [1]
    graph.tensors, graph.inputs, graph.outputs = [tensor], [0], [0]
    operation = schema.OperatorT()
    operation.opcodeIndex = 0
    graph.operators = [operation]
    if cpu:
        operation = schema.OperatorT()
        operation.opcodeIndex = 1
        graph.operators.append(operation)
    model.subgraphs = [graph]
    builder = flatbuffers.Builder(256)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


class QrcodeExportTests(unittest.TestCase):
    """Protect ONNX parity, archive serialization and ordered multi-output contracts."""

    def test_firmware_crop(self):
        """Prepare the same centered RGB crop as the firmware image generator."""
        pixels = np.arange(6 * 4 * 3, dtype=np.uint8).reshape(4, 6, 3)
        with patch("yolo_pte_utils.Image.open", return_value=Image.fromarray(pixels)):
            actual = load_image(Path("unused.png"), 4, "mlek-crop").numpy()
        expected = pixels[:, 1:5].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        np.testing.assert_array_equal(actual, expected)

    def test_static_specialization_parity(self):
        """Export static operators and archive the result without changing ONNX values."""
        graph = make_graph()
        inputs = torch.linspace(-1, 1, 48).reshape(1, 3, 4, 4)
        reference = ort.InferenceSession(graph.SerializeToString()).run(None,
                                                                       {"input": inputs.numpy()})
        model = adapt_model(graph, inputs)
        exported = torch.export.export(model, (inputs,), strict=True)
        with torch.no_grad():
            actual = exported.module(check_guards=False)(inputs).numpy()
        np.testing.assert_allclose(actual, reference[0], atol=1e-6, rtol=1e-6)
        archive = BytesIO()
        torch.export.save(exported, archive)
        restored = torch.export.load(BytesIO(archive.getvalue())).module(check_guards=False)
        with torch.no_grad():
            np.testing.assert_allclose(restored(inputs).numpy(), reference[0], atol=1e-6, rtol=1e-6)

    def test_four_output_contract(self):
        """Require NHWC image storage and reject reordered or floating interfaces."""
        shapes = [[1, 3, 256, 256], [1, 79, 32, 32], [1, 79, 16, 16],
                  [1, 79, 8, 8], [1, 1, 1344]]
        tensors = [SimpleNamespace(sizes=shape, scalar_type=ScalarType.CHAR,
                                   dim_order=[0, 2, 3, 1] if len(shape) == 4
                                   else list(range(len(shape)))) for shape in shapes]
        plan = SimpleNamespace(inputs=[0], outputs=[1, 2, 3, 4],
                               values=[SimpleNamespace(val=tensor) for tensor in tensors])
        metadata = {"input": {"shape": shapes[0]},
                    "outputs": [{"shape": shape} for shape in shapes[1:]]}
        self.assertEqual(len(check_pte_contract(plan, metadata, "channels_last")["outputs"]), 4)
        for order in ([2, 1, 3, 4], [1, 2, 3]):
            plan.outputs = order
            with self.assertRaises(ValueError):
                check_pte_contract(plan, metadata, "channels_last")
        plan.outputs = [1, 2, 3, 4]
        tensors[1].dim_order = [0, 1, 2, 3]
        with self.assertRaises(ValueError):
            check_pte_contract(plan, metadata, "channels_last")
        tensors[1].dim_order = [0, 2, 3, 1]
        tensors[1].scalar_type = ScalarType.FLOAT
        with self.assertRaises(ValueError):
            check_pte_contract(plan, metadata, "channels_last")

    def test_invalid_host_outputs(self):
        """Reject missing outputs, wrong shapes and non-finite runtime predictions."""
        reference = [np.zeros((1, 2), dtype=np.float32)]
        for actual in ([], [np.zeros((2, 1))], [np.full((1, 2), np.nan)]):
            with self.assertRaises(ValueError):
                compare_tensors(reference, actual)

    def test_nhwc_pte_contract(self):
        """Check deployment NHWC shapes while retaining angle and output order."""
        source = [[1, 3, 256, 256], [1, 79, 32, 32], [1, 79, 16, 16],
                  [1, 79, 8, 8], [1, 1, 1344]]
        deployed = [[1, 256, 256, 3], [1, 32, 32, 79], [1, 16, 16, 79],
                    [1, 8, 8, 79], [1, 1, 1344]]
        values = [SimpleNamespace(val=SimpleNamespace(
            sizes=shape, scalar_type=ScalarType.CHAR, dim_order=list(range(len(shape)))))
                  for shape in deployed]
        plan = SimpleNamespace(inputs=[0], outputs=[1, 2, 3, 4], values=values)
        metadata = {"input": {"shape": source[0]},
                    "outputs": [{"shape": shape} for shape in source[1:]]}
        contract = check_pte_contract(plan, metadata, "nhwc")
        self.assertEqual(contract["input"]["shape"], [1, 256, 256, 3])
        self.assertEqual(contract["outputs"][3]["shape"], [1, 1, 1344])
        with self.assertRaises(ValueError):
            check_pte_contract(plan, metadata, "nchw")

    def test_raw_head_detection_metrics(self):
        """Derive confidence from class logits independently of the angle head."""
        heads = [np.zeros((1, 65, size, size), dtype=np.float32) for size in (4, 2, 1)]
        outputs = heads + [np.zeros((1, 1, 21), dtype=np.float32)]
        metrics = detection_metrics(outputs, outputs, 32, 1)
        self.assertEqual(metrics["float_max_score"], 0.5)
        self.assertEqual(metrics["max_absolute_score_error"], 0)
        self.assertEqual(metrics["box_iou_at_float_best"], 1)
        altered = [item.copy() for item in outputs]
        altered[0][:, 64] = 2
        self.assertGreater(detection_metrics(outputs, altered, 32, 1)["max_absolute_score_error"],
                           0.1)

    def test_angle_does_not_change_class_scores(self):
        """Changing angle logits must not rescale detection confidence or class IDs."""
        for bins in (8, 16):
            heads = [np.full((1, 4 * bins + 15, size, size), -10, dtype=np.float32)
                     for size in (4, 2, 1)]
            heads[0][0, 4 * bins + 14, 1, 1] = 2
            outputs = heads + [np.full((1, 1, 21), -4, dtype=np.float32)]
            altered = [item.copy() for item in outputs]
            altered[3].fill(4)
            metrics = detection_metrics(outputs, altered, 32, 15)
            self.assertAlmostEqual(metrics["float_max_score"], 0.880797078, places=6)
            self.assertEqual(metrics["float_best_class"], 14)
            self.assertEqual(metrics["quantized_best_class"], 14)
            self.assertEqual(metrics["max_absolute_score_error"], 0)

    def test_float_parity_near_zero_logits(self):
        """Allow observed FP32 kernel rounding while rejecting larger raw-output changes."""
        reference = np.array([[[0.014]]], dtype=np.float32)
        sample = {"image": "gray.bmp", "input": np.zeros((1, 1, 1, 1), dtype=np.float32),
                  "reference": [reference]}
        actual = reference + 0.000136
        self.assertEqual(len(validate_outputs(lambda _: [actual], [sample], strict=True)), 1)
        actual = reference + 0.001
        with self.assertRaises(AssertionError):
            validate_outputs(lambda _: [actual], [sample], strict=True)

    def test_angle_rounding_keeps_detection_head_limit(self):
        """Accept measured angle kernel error without relaxing the detection heads."""
        angle = np.array([[[0.152]]], dtype=np.float32)
        sample = {"image": "angle_rounding.bmp", "input": np.zeros((1, 1, 1, 1)),
                  "reference": [angle]}
        actual = angle + 0.00023684
        self.assertEqual(len(validate_outputs(lambda _: [actual], [sample], strict=True)), 1)
        sample["reference"] = [angle.reshape((1, 1, 1, 1))]
        with self.assertRaisesRegex(AssertionError, "angle_rounding.bmp.*output=0"):
            validate_outputs(lambda _: [actual.reshape((1, 1, 1, 1))], [sample], strict=True)
        sample["reference"] = [np.array([[[1.014]]], dtype=np.float32)]
        actual = sample["reference"][0] + 0.0005271
        self.assertEqual(len(validate_outputs(lambda _: [actual], [sample], strict=True)), 1)

    def test_actual_npu_operators(self):
        """Ignore unused CPU opcode entries but reject CPU operators in the graph."""
        path = Mock()
        path.read_bytes.return_value = make_flatbuffer(False)
        self.assertEqual(inspect_tflite(path, require_npu=True)["cpu_operator_count"], 0)
        path.read_bytes.return_value = make_flatbuffer(True)
        with self.assertRaises(ValueError):
            inspect_tflite(path, require_npu=True)


if __name__ == "__main__":
    unittest.main()
