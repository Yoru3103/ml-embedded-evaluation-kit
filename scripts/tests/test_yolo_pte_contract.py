# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for the original MLEK input/output contract."""

from types import SimpleNamespace
import unittest

import torch

from executorch.exir.schema import ScalarType

from model_export.yolo.pte import check_io_contract
from model_export.yolo.utils import ExportYolo


class DetectionHead(torch.nn.Module):
    """Expose independent boxes and logits to exercise the export adapter."""

    def forward(self, inputs: torch.Tensor) -> tuple:
        """Return known pixel XYWH and logits without combining their ranges."""
        return inputs[:, :4], inputs[:, 4:]


class ExportContractTests(unittest.TestCase):
    """Protect the deployment boundary instead of adding compatibility branches."""

    def test_probabilities_and_normalized_boxes(self):
        """Keep sigmoid and normalization inside the exported model."""
        values = torch.tensor([160., 80., 64., 32., 0., 2.]).reshape(1, 6, 1)
        output = ExportYolo(DetectionHead(), 320)(values)
        self.assertEqual(tuple(output.shape), (1, 6, 1))
        self.assertTrue(output.is_contiguous())
        torch.testing.assert_close(output[:, :4], values[:, :4] / 320)
        torch.testing.assert_close(output[:, 4:], values[:, 4:].sigmoid())

    def test_reject_incompatible_serialized_interfaces(self):
        """Reject interleaved input, split output, wrong class count and int8 I/O."""
        input_tensor = SimpleNamespace(sizes=[1, 3, 320, 320],
                                       dim_order=[0, 1, 2, 3], scalar_type=ScalarType.FLOAT)
        output_tensor = SimpleNamespace(sizes=[1, 14, 2100], dim_order=[0, 1, 2],
                                        scalar_type=ScalarType.FLOAT)
        plan = SimpleNamespace(inputs=[0], outputs=[1],
                               values=[SimpleNamespace(val=input_tensor),
                                       SimpleNamespace(val=output_tensor)])
        self.assertEqual(check_io_contract(plan, 320, 10)["outputs"][0]["shape"], [1, 14, 2100])
        input_tensor.dim_order = [2, 0, 3, 1]
        with self.assertRaises(ValueError):
            check_io_contract(plan, 320, 10)
        input_tensor.dim_order = [0, 1, 2, 3]
        plan.outputs = [1, 1]
        with self.assertRaises(ValueError):
            check_io_contract(plan, 320, 10)
        plan.outputs = [1]
        with self.assertRaises(ValueError):
            check_io_contract(plan, 320, 80)
        output_tensor.scalar_type = ScalarType.CHAR
        with self.assertRaises(ValueError):
            check_io_contract(plan, 320, 10)

    def test_nhwc_input_contract(self):
        """An NHWC PT deployment retains the decoded three-dimensional output."""
        tensors = [SimpleNamespace(sizes=[1, 320, 320, 3], dim_order=[0, 1, 2, 3],
                                    scalar_type=ScalarType.FLOAT),
                   SimpleNamespace(sizes=[1, 14, 2100], dim_order=[0, 1, 2],
                                    scalar_type=ScalarType.FLOAT)]
        plan = SimpleNamespace(inputs=[0], outputs=[1],
                               values=[SimpleNamespace(val=tensor) for tensor in tensors])
        self.assertEqual(check_io_contract(plan, 320, 10, "nhwc")["inputs"][0]["shape"],
                         [1, 320, 320, 3])
        with self.assertRaises(ValueError):
            check_io_contract(plan, 320, 10, "nchw")


if __name__ == "__main__":
    unittest.main()
