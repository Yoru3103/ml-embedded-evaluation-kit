# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Keep the RMSNorm chain in float for the TinyStories precision bisect."""

from __future__ import annotations

from typing import Iterable

import torch
from torch.fx import GraphModule, Node


def _target(node: Node) -> str:
    """Return the printable target of an FX node.

    :param node: Graph node.
    :returns:    Operator name.
    """
    return node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)


def _user(node: Node, target: str) -> Node:
    """Return the unique user of ``node`` with the requested target.

    :param node:   Producer node.
    :param target: Expected operator name.
    :returns:      Matching user.
    :raises ValueError: If there is not exactly one matching user.
    """
    matches = [user for user in node.users if _target(user) == target]
    if len(matches) != 1:
        raise ValueError(f'{node.name} has {len(matches)} {target} users')
    return matches[0]


def _remove(nodes: Iterable[Node]):
    """Erase nodes that no longer have users."""
    for node in nodes:
        if len(node.users) == 0:
            node.graph.erase_node(node)


def remove_rmsnorm_quantization(module: GraphModule) -> int:
    """Remove Q/DQ pairs around the mean/add/rsqrt/mul RMSNorm chain.

    :param module: Quantized exported graph module.
    :returns:      Number of transformed RMSNorm chains.
    :raises ValueError: If the expected pattern is not present.
    """
    count = 0
    for rsqrt in [node for node in module.graph.nodes if _target(node) == 'rsqrt.default']:
        add_dq = rsqrt.args[0]
        add_q = add_dq.args[0]
        add = add_q.args[0]
        view1_dq, eps_dq = add.args
        view1_q = view1_dq.args[0]
        view1 = view1_q.args[0]
        mul1_dq = view1.args[0]
        mul1_q = mul1_dq.args[0]
        mul1 = mul1_q.args[0]
        sum_dq, mean_dq = mul1.args
        sum_q = sum_dq.args[0]
        sum_node = sum_q.args[0]
        view_dq = sum_node.args[0]
        rsqrt_q = _user(rsqrt, 'quantize_per_tensor.default')
        rsqrt_dq = _user(rsqrt_q, 'dequantize_per_tensor.default')
        mul2 = _user(rsqrt_dq, 'mul.Tensor')
        mul2_q = _user(mul2, 'quantize_per_tensor.default')
        mul2_dq = _user(mul2_q, 'dequantize_per_tensor.default')

        # Keep the reduction on the NPU; the delegate boundary is the sum Q/DQ.
        mul1.args = (sum_dq, mean_dq)
        view1.args = (mul1, [1, 1, 1])
        add.args = (view1, eps_dq)
        rsqrt.args = (add,)
        mul2.args = (mul2.args[0], rsqrt)
        mul2_dq.replace_all_uses_with(mul2)

        _remove([mul1_q, mul1_dq, view1_q, view1_dq,
                 add_q, add_dq, rsqrt_q, rsqrt_dq, mul2_q, mul2_dq])
        count += 1
    if count == 0:
        raise ValueError('No RMSNorm chain found')
    module.recompile()
    return count
