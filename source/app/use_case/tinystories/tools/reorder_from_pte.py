# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Reorder one delegate TOSA JSON to the ExecuTorch argument shape/dtype order."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'dependencies/executorch'))
import executorch  # noqa: E402

executorch.__path__ = [str(ROOT / 'dependencies/executorch'), *executorch.__path__]
from mlek_tools.validation.pte_ops_dump import _load_pte_program  # noqa: E402

from validate_delegate_io import read_io  # noqa: E402

TYPE_MAP = {
    'BOOL': 'BOOL', 'BYTE': 'INT8', 'CHAR': 'INT8', 'SHORT': 'INT16',
    'INT': 'INT32', 'LONG': 'INT64', 'HALF': 'FP16', 'FLOAT': 'FP32',
}


def main():
    """Reorder TOSA inputs and outputs to the ET delegate argument order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pte', type=Path)
    parser.add_argument('index', type=int)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    args = parser.parse_args()

    program = _load_pte_program(args.pte.read_bytes())
    call = None
    delegate = None
    current = 0
    for plan in program.execution_plan:
        for chain in plan.chains:
            for instruction in chain.instructions:
                candidate = instruction.instr_args
                if not hasattr(candidate, 'delegate_index'):
                    continue
                item = plan.delegates[candidate.delegate_index]
                if item.id != 'EthosUBackend':
                    continue
                if current == args.index:
                    call = candidate
                    delegate = item
                    break
                current += 1
            if call is not None:
                break
        if call is not None:
            break
    if call is None or delegate is None:
        raise ValueError(f'Delegate {args.index} not found')
    descriptors = read_io(bytes(program.backend_delegate_data[delegate.processed.index].data))
    input_count = len(descriptors['inputs'])
    # The PTE API exposes values through the execution plan.
    values = plan.values
    et_inputs = [values[a].val for a in call.args[:input_count]]
    et_outputs = [values[a].val for a in call.args[input_count:]]

    document = json.loads(args.source.read_text(encoding='utf-8'))
    graph = document['regions'][0]['blocks'][0]
    tensors = {tensor['name']: tensor for tensor in graph['tensors']}

    def reorder(names, expected, label):
        remaining = list(names)
        ordered = []
        for tensor in expected:
            shape = list(tensor.sizes)
            dtype = TYPE_MAP[tensor.scalar_type.name]
            for name in remaining:
                item = tensors[name]
                if item['shape'] == shape and item['type'] == dtype:
                    ordered.append(name)
                    remaining.remove(name)
                    break
            else:
                raise ValueError(f'{label}: no match for {shape} {dtype}; remaining={remaining}')
        if remaining:
            raise ValueError(f'{label}: unused {remaining}')
        return ordered

    graph['inputs'] = reorder(graph['inputs'], et_inputs, 'inputs')
    graph['outputs'] = reorder(graph['outputs'], et_outputs, 'outputs')
    args.target.write_text(json.dumps(document), encoding='utf-8')
    print(f'delegate {args.index}: inputs={len(et_inputs)} outputs={len(et_outputs)}')


if __name__ == '__main__':
    main()
