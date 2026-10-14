# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Protect the channel split and raw-angle meaning without external model files."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import helper, numpy_helper
from PIL import Image

from qrcode_contract import describe_model
from yolo_pte_utils import load_image


def make_obb_graph(channels: int, bins: int, classes: int) -> onnx.ModelProto:
    """Construct three raw convolution heads and flattened angle branches.

    :param channels:    Input channel count.
    :param bins:        DFL bins per distance.
    :param classes:     Class-logit channel count.
    :returns:           Static in-memory ONNX graph.
    """
    nodes, weights, outputs, angles = [], [], [], []
    for stride in (8, 16, 32):
        size = 32 // stride
        for branch, count in (("box", 4 * bins), ("class", classes), ("angle", 1)):
            name = f"{branch}_{stride}"
            weights.append(numpy_helper.from_array(
                np.zeros((count, channels, 1, 1), dtype=np.float32), name + "_weight"))
            nodes.append(helper.make_node("Conv", ["input", name + "_weight"], [name],
                                          strides=[stride, stride]))
        nodes.append(helper.make_node("Concat", [f"box_{stride}", f"class_{stride}"],
                                      [f"head_{stride}"], axis=1))
        outputs.append(helper.make_tensor_value_info(
            f"head_{stride}", onnx.TensorProto.FLOAT, [1, 4 * bins + classes, size, size]))
        weights.append(numpy_helper.from_array(
            np.array([1, 1, size * size], dtype=np.int64), f"angle_shape_{stride}"))
        angles.append(f"angle_flat_{stride}")
        nodes.append(helper.make_node("Reshape", [f"angle_{stride}", f"angle_shape_{stride}"],
                                      [angles[-1]]))
    nodes.append(helper.make_node("Concat", angles, ["angles"], axis=2))
    outputs.append(helper.make_tensor_value_info("angles", onnx.TensorProto.FLOAT, [1, 1, 21]))
    graph = helper.make_graph(nodes, "obb", [helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [1, channels, 32, 32])], outputs, weights)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)


class QrcodeContractTests(unittest.TestCase):
    """Check graph-defined class counts and deployment image preparation."""

    def test_rgb_and_grayscale_channel_splits(self):
        """Read 16/8-bin contracts and class counts from actual convolution weights."""
        for channels, bins, classes in ((3, 16, 15), (1, 8, 15), (3, 8, 7)):
            with self.subTest(channels=channels, bins=bins, classes=classes):
                _, metadata = describe_model(make_obb_graph(channels, bins, classes))
                self.assertEqual(metadata["input_channels"], channels)
                self.assertEqual(metadata["dfl_bins"], bins)
                self.assertEqual(metadata["num_classes"], classes)
                self.assertEqual(metadata["auxiliary_kind"], "angle_logits")

    def test_activated_angles_are_rejected(self):
        """Do not silently apply sigmoid twice when a graph changes its angle interface."""
        graph = make_obb_graph(3, 16, 15)
        graph.graph.node[-1].output[0] = "raw_angles"
        graph.graph.node.append(helper.make_node("Sigmoid", ["raw_angles"], ["angles"]))
        with self.assertRaisesRegex(ValueError, "raw convolution heads"):
            describe_model(graph)

    def test_center_crop_and_grayscale(self):
        """Match generated RGB samples, then convert to one channel when requested."""
        pixels = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]] * 2,
                          dtype=np.uint8)
        for channels in (3, 1):
            with patch("yolo_pte_utils.Image.open", return_value=Image.fromarray(pixels)):
                prepared = load_image(Path("in-memory.png"), 2, "mlek-crop", channels).numpy()
            expected = Image.fromarray(pixels[:, 1:3])
            if channels == 1:
                expected = expected.convert("L")
            values = np.asarray(expected).reshape(2, 2, channels).transpose(2, 0, 1)
            np.testing.assert_array_equal(prepared, values[None].astype(np.float32) / 255)


if __name__ == "__main__":
    unittest.main()
