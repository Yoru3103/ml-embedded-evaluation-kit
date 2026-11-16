# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Keep the attention softmax chain in float for the TinyStories experiment.

The quantized graph surrounds ``amax/sub/exp/sum/reciprocal/mul`` with
``quantize_per_tensor`` / ``dequantize_per_tensor`` pairs.  Removing those
internal pairs lets the Arm partitioner leave the softmax chain on the CPU
while the QK and PV matmuls stay on the Ethos-U delegate.
"""

from __future__ import annotations

from typing import Iterable

import torch
from executorch.exir.pass_base import ExportPass
from torch.fx import GraphModule, Node
from torch.fx.passes.infra.pass_base import PassResult


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


def _quant_pair(node: Node) -> tuple[Node, Node]:
    """Return the quantize/dequantize pair following a float node.

    :param node: Float producer.
    :returns:    Quantize node and its dequantize user.
    """
    quantize = _user(node, 'quantize_per_tensor.default')
    dequantize = _user(quantize, 'dequantize_per_tensor.default')
    return quantize, dequantize


def _remove(nodes: Iterable[Node]):
    """Erase nodes that no longer have users.

    :param nodes: Candidate nodes in reverse topological order.
    """
    for node in nodes:
        if len(node.users) == 0:
            node.graph.erase_node(node)


def remove_softmax_quantization(module: GraphModule) -> int:
    """Remove quantize/dequantize pairs inside every attention softmax.

    The graph is rewired so ``amax``, ``sub``, ``exp``, ``sum``,
    ``reciprocal`` and ``mul`` consume and produce float tensors.  The
    quantized scores entering the chain and the quantized probabilities
    leaving it remain, so the delegate boundary stays int16.

    :param module: Exported quantized graph module.
    :returns:      Number of transformed softmax chains.
    :raises ValueError: If the expected softmax pattern is not present.
    """
    transformed = 0
    exp_nodes = [node for node in module.graph.nodes if _target(node) == 'exp.default']
    for exp in exp_nodes:
        exp_input_dq = exp.args[0]
        exp_input_q = exp_input_dq.args[0]
        sub = exp_input_q.args[0]
        scores_dq, max_dq = sub.args[0], sub.args[1]
        max_q = max_dq.args[0]
        amax = max_q.args[0]
        # The exp output feeds both sum and mul through separate dequantize
        # nodes that share the same quantize node.
        exp_q = _user(exp, 'quantize_per_tensor.default')
        exp_dqs = [user for user in exp_q.users
                   if _target(user) == 'dequantize_per_tensor.default']
        sum_node = next(user for dequantize in exp_dqs for user in dequantize.users
                        if _target(user) == 'sum.dim_IntList')
        sum_q, sum_dq = _quant_pair(sum_node)
        reciprocal = _user(sum_dq, 'reciprocal.default')
        reciprocal_q, reciprocal_dq = _quant_pair(reciprocal)
        mul = next(user for dequantize in exp_dqs for user in dequantize.users
                   if _target(user) == 'mul.Tensor')
        mul_q, mul_dq = _quant_pair(mul)

        amax.args = (scores_dq, [-1], True)
        sub.args = (scores_dq, amax)
        exp.args = (sub,)
        sum_node.args = (exp, [-1], True)
        reciprocal.args = (sum_node,)
        mul.args = (exp, reciprocal)
        mul_dq.replace_all_uses_with(mul)

        doomed = [max_q, max_dq, exp_input_q, exp_input_dq, exp_q,
                  sum_q, sum_dq, reciprocal_q, reciprocal_dq, mul_q, mul_dq]
        doomed.extend(exp_dqs)
        _remove(doomed)
        transformed += 1
    if transformed == 0:
        # Edge lowering may already have fused the decomposition back into a
        # single ``_softmax``/``_safe_softmax`` node.  In that case remove the
        # quantize/dequantize pair around the fused node so the partitioner
        # keeps it on the CPU.
        softmax_nodes = [
            node for node in module.graph.nodes
            if _target(node) in ('_softmax.default', '_safe_softmax.default',
                                 'softmax.int')
        ]
        for node in softmax_nodes:
            output_q = _user(node, 'quantize_per_tensor.default')
            output_dq = _user(output_q, 'dequantize_per_tensor.default')
            output_dq.replace_all_uses_with(node)
            _remove([output_q, output_dq])
            transformed += 1
    if transformed == 0:
        names = sorted({_target(node) for node in module.graph.nodes
                        if 'softmax' in _target(node) or 'exp' in _target(node)})
        raise ValueError(f'No softmax chain found; candidates={names}')
    module.recompile()
    return transformed


class SoftmaxCpuPass(ExportPass):
    """Export pass that moves the attention softmax chain to float."""

    def call(self, graph_module: GraphModule) -> PassResult:
        """Apply the transform to an edge graph.

        :param graph_module: Quantized edge graph module.
        :returns:            Pass result with the modified graph.
        """
        count = remove_softmax_quantization(graph_module)
        print(f'TINYSTORIES_SOFTMAX_CPU: transformed {count} chains', flush=True)
        return PassResult(graph_module, True)


def transform(program: torch.export.ExportedProgram) -> int:
    """Apply the softmax transform to an exported program.

    :param program: Quantized exported program.
    :returns:       Number of transformed softmax chains.
    """
    return remove_softmax_quantization(program.graph_module)
