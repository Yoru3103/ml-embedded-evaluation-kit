# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Terminalize full KV-cache outputs in one delegate TOSA JSON document."""

import argparse
import json
from pathlib import Path


def main():
    """Add a transpose pair for each [1,8,context,64] graph output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    parser.add_argument('--context', type=int, default=64)
    args = parser.parse_args()

    document = json.loads(args.source.read_text(encoding='utf-8'))
    graph = document['regions'][0]['blocks'][0]
    tensors = {tensor['name']: tensor for tensor in graph['tensors']}
    outputs = graph['outputs']
    cache_shape = [1, 8, args.context, 64]
    permutation = [0, 1, 3, 2]
    swapped_shape = [cache_shape[axis] for axis in permutation]
    added = 0
    for index, name in enumerate(list(outputs)):
        tensor = tensors.get(name)
        if tensor is None or tensor.get('shape') != cache_shape:
            continue
        swapped = f'ethos_u_terminal_{index:02d}_swap'
        terminal = f'ethos_u_terminal_{index:02d}_out'
        graph['operators'].extend([
            {
                'op': 'TRANSPOSE',
                'attribute_type': 'TransposeAttribute',
                'attribute': {'perms': permutation},
                'inputs': [name],
                'outputs': [swapped],
                'location': {'text': ''},
            },
            {
                'op': 'TRANSPOSE',
                'attribute_type': 'TransposeAttribute',
                'attribute': {'perms': permutation},
                'inputs': [swapped],
                'outputs': [terminal],
                'location': {'text': ''},
            },
        ])
        for tensor_name, shape in ((swapped, swapped_shape), (terminal, cache_shape)):
            graph['tensors'].append({
                'name': tensor_name,
                'shape': shape,
                'type': 'INT16',
                'data': [],
                'variable_name': '',
            })
        outputs[index] = terminal
        added += 1
    if added == 0:
        raise ValueError('No full KV output found')
    args.target.write_text(json.dumps(document), encoding='utf-8')
    print(f'terminalized {added} outputs; graph outputs={outputs}')


if __name__ == '__main__':
    main()
