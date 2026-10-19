# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Shared NHWC deployment boundary for models with NCHW computation semantics."""

import torch


def model_input(value: torch.Tensor, layout: str) -> torch.Tensor:
    """Arrange a preprocessed NCHW image for the selected deployment interface.

    :param value:       NCHW image tensor.
    :param layout:      ``nhwc`` or ``nchw`` deployment layout.
    :returns:           Contiguous model input.
    :raises ValueError: Unsupported deployment layout.
    """
    if layout == "nhwc":
        return value.permute(0, 2, 3, 1).contiguous()
    if layout == "nchw":
        return value
    raise ValueError(f"Unsupported model layout: {layout}")


class ChannelLastIO(torch.nn.Module):
    """Accept NHWC images and return NHWC feature maps, preserving other tensors."""

    def __init__(self, model: torch.nn.Module):
        """Wrap a model without rewriting its operators or parameters.

        :param model:   Source module using NCHW computation semantics.
        """
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor | tuple | list:
        """Adapt the interface while retaining ordered outputs and angle semantics.

        :param inputs:  NHWC images.
        :returns:       NHWC feature maps and unchanged non-four-dimensional outputs.
        """
        outputs = self.model(inputs.permute(0, 3, 1, 2))
        if isinstance(outputs, torch.Tensor):
            return self._output(outputs)
        converted = [self._output(value) for value in outputs]
        return tuple(converted) if isinstance(outputs, tuple) else converted

    @staticmethod
    def _output(value: torch.Tensor) -> torch.Tensor:
        """Move the channels of four-dimensional features to the last axis.

        :param value:   Feature map or auxiliary output.
        :returns:       Contiguous feature map or unchanged auxiliary tensor.
        """
        return value.permute(0, 2, 3, 1).contiguous() if value.ndim == 4 else value
