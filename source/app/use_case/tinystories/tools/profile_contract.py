# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Derive expected per-inference profiling call counts from the actual PTE."""

import argparse
from collections import Counter
from pathlib import Path

from mlek_tools.validation.pte_ops_dump import _load_pte_program
from validate_delegate_io import read_io


def write_contract(model_path: Path, header_path: Path):
    """Generate counts in TinyStoriesProfiler::Stage order.

    :param model_path:   PTE that will be embedded in the firmware.
    :param header_path:  Generated C++ header destination.
    :raises ValueError:  Unexpected operators or delegate topology.
    """
    program = _load_pte_program(model_path.read_bytes())
    if program is None:
        raise ValueError('Unable to load PTE')
    plan = next(plan for plan in program.execution_plan if plan.name == 'forward')
    stages = {
        'quantized_decomposed::embedding_byte.dtype_out': 0,
        'quantized_decomposed::quantize_per_tensor.out': 1,
        'quantized_decomposed::dequantize_per_tensor.out': 2,
        'aten::index.Tensor_out': 3,
        'aten::copy_.': 4,
        'aten::clamp.out': 5,
    }
    counts = Counter()
    for chain in plan.chains:
        for instruction in chain.instructions:
            call = instruction.instr_args
            if hasattr(call, 'op_index'):
                name = (plan.operators[call.op_index].name + '.' +
                        plan.operators[call.op_index].overload)
                if name not in stages:
                    raise ValueError(f'Unclassified native operator: {name}')
                counts[stages[name]] += 1
            elif hasattr(call, 'delegate_index'):
                delegate = plan.delegates[call.delegate_index]
                if delegate.id != 'EthosUBackend':
                    raise ValueError(f'Unexpected delegate: {delegate.id}')
                payload = bytes(program.backend_delegate_data[delegate.processed.index].data)
                descriptors = read_io(payload)
                counts[7] += 1
                counts[8] += len(descriptors['inputs'])
                counts[9] += len(descriptors['outputs'])
                counts[10] += 1
    expected = [counts[index] for index in range(11)]
    if any(counts[index] == 0 for index in range(4)) or counts[7] != 1:
        raise ValueError(f'Unexpected TinyStories operator topology: {expected}')
    header = ('// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates\n'
              '// <open-source-office@arm.com>\n// SPDX-License-Identifier: Apache-2.0\n'
              '// Generated from the embedded PTE by tools/profile_contract.py.\n'
              '#pragma once\n#include <cstdint>\nnamespace arm::app::tinystories {\n'
              'constexpr uint32_t kProfileExpectedCalls[] = {'
              + ', '.join(map(str, expected)) + '};\n}\n')
    header_path.write_text(header, encoding='utf-8')
    print(f'TinyStories profiling contract: {expected}; native calls={sum(expected[:7])}')


def main():
    """Read the PTE and write its generated profiling contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('header', type=Path)
    args = parser.parse_args()
    write_contract(args.model, args.header)


if __name__ == '__main__':
    main()
