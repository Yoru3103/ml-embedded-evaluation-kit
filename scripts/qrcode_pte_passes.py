# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Make calibrated convolution/ReLU boundaries explicit for Ethos-U lowering."""

import torch


def separate_quantized_relu(graph_module: torch.fx.GraphModule) -> int:
    """Split convolution/ReLU patterns that cannot use the backend's fused quantization.

    Reuse the post-ReLU scale and zero point before ReLU. Quantizing before
    clamping at zero preserves the final integer result when zero is exactly
    representable. Both operations then have explicit quantized boundaries.

    :param graph_module:   Calibrated ATen graph to update in place.
    :returns:              Number of inserted quantize/dequantize boundaries.
    """
    quantize = torch.ops.quantized_decomposed.quantize_per_tensor.default
    dequantize = torch.ops.quantized_decomposed.dequantize_per_tensor.default
    convolutions = {torch.ops.aten.conv2d.default, torch.ops.aten.convolution.default}
    changed = 0
    for activation in list(graph_module.graph.nodes):
        if (activation.target not in (torch.ops.aten.relu.default, torch.ops.aten.relu_.default)
                or len(activation.users) != 1):
            continue
        source = activation.args[0]
        output = next(iter(activation.users))
        if source.target not in convolutions or output.target != quantize:
            continue
        _, scale, zero_point, minimum, maximum, dtype = output.args
        if zero_point == minimum or not minimum <= zero_point <= maximum:
            continue
        with graph_module.graph.inserting_before(activation):
            quantized = graph_module.graph.call_function(
                quantize, (source, scale, zero_point, minimum, maximum, dtype))
            restored = graph_module.graph.call_function(
                dequantize, (quantized, scale, zero_point, minimum, maximum, dtype))
        quantized.meta = dict(source.meta)
        quantized.meta["val"] = source.meta["val"].to(dtype=dtype)
        restored.meta = dict(source.meta)
        activation.replace_input_with(source, restored)
        changed += 1
    graph_module.graph.lint()
    graph_module.recompile()
    return changed
