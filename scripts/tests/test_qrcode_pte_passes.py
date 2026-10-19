# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Check explicit Conv/ReLU quantization at rounding and saturation boundaries."""

import copy
import unittest

import torch
from qrcode_export_backends import separate_quantized_relu


class QuantizedRelu(torch.nn.Module):
    """Expose a calibrated Conv/ReLU pair with a configurable output zero point."""

    def __init__(self, zero_point: int):
        """Use an identity convolution and a known output grid.

        :param zero_point:  Integer representation of zero in the output.
        """
        super().__init__()
        self.register_buffer("weight", torch.ones(1, 1, 1, 1))
        self.zero_point = zero_point

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return quantized ReLU values, including saturation outside the grid.

        :param inputs:  NCHW floating input.
        :returns:       INT8 post-ReLU values.
        """
        return torch.ops.quantized_decomposed.quantize_per_tensor.default(
            torch.relu(torch.ops.aten.conv2d.default(inputs, self.weight)),
            0.125, self.zero_point, -128, 127, torch.int8)


class QuantizedReluBoundaryTests(unittest.TestCase):
    """Preserve integer outputs and avoid adding repeated quantization boundaries."""

    def test_asymmetric_boundary_preserves_rounding_and_saturation(self):
        """Cover negative, zero, half-step and saturated values on the output grid."""
        inputs = torch.linspace(-32, 32, 4097).reshape(1, 1, 1, -1)
        for zero_point in (-128, -28, 0, 127):
            with self.subTest(zero_point=zero_point):
                original = torch.export.export(QuantizedRelu(zero_point), (inputs,)).module()
                fixed = copy.deepcopy(original)
                self.assertEqual(separate_quantized_relu(fixed), int(zero_point != -128))
                torch.testing.assert_close(fixed(inputs), original(inputs), rtol=0, atol=0)
                self.assertEqual(separate_quantized_relu(fixed), 0)
                fixed.graph.lint()


if __name__ == "__main__":
    unittest.main()
