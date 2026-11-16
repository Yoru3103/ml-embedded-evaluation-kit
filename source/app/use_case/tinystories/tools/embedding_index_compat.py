# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Preserve Long indices for CPU quantized embedding during Arm annotation."""

import torch
from executorch.backends.arm._passes.insert_int32_casts_after_int64_placeholders import (
    InsertInt32CastsAfterInt64PlaceholdersPass,
)


def install():
    """Install a process-local exception for quantized embedding index edges."""
    pass_type = InsertInt32CastsAfterInt64PlaceholdersPass
    original = pass_type._insert_placeholder_i64_to_i32_casts

    def preserve_embedding_indices(
        self: InsertInt32CastsAfterInt64PlaceholdersPass,
        graph_module: torch.fx.GraphModule,
    ) -> bool:
        """Restore Long edges after the normal placeholder conversion.

        :param self:            Arm annotation pass instance.
        :param graph_module:    Graph containing the CPU embedding boundary.
        :returns:               Whether the original pass changed the graph.
        """
        protected = []
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            if "quantized_decomposed.embedding_byte" not in str(node.target):
                continue
            indices = node.args[5]
            if (
                isinstance(indices, torch.fx.Node)
                and indices.op == "placeholder"
                and indices.meta["val"].dtype == torch.int64
            ):
                protected.append((node, indices))

        modified = original(self, graph_module)
        for node, indices in protected:
            args = list(node.args)
            args[5] = indices
            node.args = tuple(args)
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        return modified

    pass_type._insert_placeholder_i64_to_i32_casts = preserve_embedding_indices
