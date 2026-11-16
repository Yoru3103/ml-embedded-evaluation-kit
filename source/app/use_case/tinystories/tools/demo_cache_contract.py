# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Derive demo reset regions from the selected persistent INT16-cache PTE."""

import argparse
from pathlib import Path

from mlek_tools.validation.pte_ops_dump import _load_pte_program


def export(model: Path, header: Path, context: int):
    """Validate cache ownership and generate bounded planned-memory reset offsets.

    :param model:    Selected INT16-cache PTE.
    :param header:   Generated C++ contract destination.
    :param context:  Context length the firmware is compiled for.
    """
    program = _load_pte_program(model.read_bytes())
    plan = next(item for item in program.execution_plan if item.name == 'forward')
    regions = []
    cache_shape = None
    for chain in plan.chains:
        for instruction in chain.instructions:
            call = instruction.instr_args
            if not hasattr(call, 'op_index'):
                continue
            operator = plan.operators[call.op_index]
            if (operator.name, operator.overload) != ('aten::clamp', 'out'):
                continue
            tensor = plan.values[call.args[0]].val
            allocation = tensor.allocation_info
            if allocation is None:
                raise ValueError('Cache has no planned allocation')
            if (tensor.scalar_type.name != 'SHORT' or list(tensor.sizes) != [1, 8, context, 64] or
                    call.args[0] != call.args[-1] or call.args[0] != call.args[-2] or
                    allocation.memory_id != 1):
                raise ValueError('Unsupported demo KV state contract')
            if cache_shape is None:
                cache_shape = list(tensor.sizes)
            if list(tensor.sizes) != cache_shape:
                raise ValueError('Cache shapes differ')
            offset = allocation.memory_offset_low + (allocation.memory_offset_high << 32)
            regions.append((offset, 2 * 8 * context * 64))
    regions.sort()
    if not regions:
        # Floor / LUT variants do not contain the aten::clamp cache fast path.
        # Fall back to every INT16 [1,8,context,64] allocation in the planned
        # cache arena; zeroing all aliases is safe for the demo reset.
        seen = {}
        for value in plan.values:
            tensor = value.val
            if not hasattr(tensor, 'scalar_type'):
                continue
            allocation = tensor.allocation_info
            if (tensor.scalar_type.name != 'SHORT' or list(tensor.sizes) != [1, 8, context, 64]
                    or allocation is None or allocation.memory_id != 1):
                continue
            offset = allocation.memory_offset_low + (allocation.memory_offset_high << 32)
            seen[offset] = 2 * 8 * context * 64
        regions = sorted(seen.items())
    arena = plan.non_const_buffer_sizes
    if len(arena) != 2 or arena[0] != 0:
        raise ValueError('Expected exactly one planned cache arena')
    if not all((len(regions) in (16, 32), len(set(regions)) == len(regions),
                all(offset + size <= arena[1] for offset, size in regions),
                all(left[0] + left[1] <= right[0]
                    for left, right in zip(regions, regions[1:])))):
        raise ValueError('Expected sixteen disjoint caches in one planned arena')
    text = ('// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates\n'
            '// <open-source-office@arm.com>\n// SPDX-License-Identifier: Apache-2.0\n'
            '#pragma once\n#include <cstddef>\nnamespace arm::app::tinystories {\n'
            'struct CacheRegion { size_t offset; size_t bytes; };\n'
            f'constexpr size_t kContextTokens = {context};\n'
            f'constexpr size_t kPlannedCacheArenaBytes = {arena[1]};\n'
            'constexpr CacheRegion kCacheRegions[] = {\n')
    text += ''.join(f'    {{{offset}, {size}}},\n' for offset, size in regions)
    header.write_text(text + '};\n}\n', encoding='utf-8')
    print(f'TinyStories demo reset: {len(regions)} caches; context={context}; '
          f'arena={arena[1]} bytes')


def main():
    """Generate a reset contract for the exact configured PTE."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('header', type=Path)
    parser.add_argument('--context', type=int, default=64)
    args = parser.parse_args()
    export(args.model, args.header, args.context)


if __name__ == '__main__':
    main()
