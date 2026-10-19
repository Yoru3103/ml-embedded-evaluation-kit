# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Check channel order and angle preservation at the shared deployment boundary."""

from io import BytesIO
import unittest

import torch


from model_export.common.layout import ChannelLastIO, model_input


class RawHeads(torch.nn.Module):
    """Expose channel-sensitive feature maps and a separate angle tensor."""

    def forward(self, inputs: torch.Tensor) -> tuple:
        """Compute three feature maps and channel-zero angle values.

        :param inputs:  Two-channel NCHW tensor.
        :returns:       Ordered feature maps followed by an angle tensor.
        """
        return (inputs[:, :1] * 2, inputs[:, 1:] * 3, inputs,
                inputs[:, :1].reshape(1, 1, -1))


class ModelLayoutTests(unittest.TestCase):
    """Protect NHWC deployment without changing the source model semantics."""

    def test_channel_last_archive(self):
        """An exported adapter keeps pixel channels and angle ordering intact."""
        inputs = torch.tensor([[[[1., 10.], [2., 20.]], [[3., 30.], [4., 40.]]]])
        model = ChannelLastIO(RawHeads())
        exported = torch.export.export(model, (inputs,), strict=True)
        archive = BytesIO()
        torch.export.save(exported, archive)
        restored = torch.export.load(BytesIO(archive.getvalue())).module(check_guards=False)
        outputs = restored(inputs)
        self.assertEqual([list(value.shape) for value in outputs],
                         [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 2], [1, 1, 4]])
        torch.testing.assert_close(outputs[0].flatten(), torch.tensor([2., 4., 6., 8.]))
        torch.testing.assert_close(outputs[1].flatten(), torch.tensor([30., 60., 90., 120.]))
        torch.testing.assert_close(outputs[2], inputs)
        torch.testing.assert_close(outputs[3], torch.tensor([[[1., 2., 3., 4.]]]))

    def test_calibration_input_order(self):
        """Calibration uses contiguous interleaved channels for NHWC inputs."""
        inputs = torch.tensor([[[[1., 2.]], [[10., 20.]], [[100., 200.]]]])
        converted = model_input(inputs, "nhwc")
        self.assertEqual(list(converted.shape), [1, 1, 2, 3])
        torch.testing.assert_close(converted.flatten(),
                                   torch.tensor([1., 10., 100., 2., 20., 200.]))
        self.assertTrue(converted.is_contiguous())
        torch.testing.assert_close(model_input(inputs, "nchw"), inputs)
        with self.assertRaises(ValueError):
            model_input(inputs, "invalid")


if __name__ == "__main__":
    unittest.main()
