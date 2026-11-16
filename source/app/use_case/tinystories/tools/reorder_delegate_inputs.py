# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Reorder TOSA graph inputs to match the ExecuTorch delegate argument order."""

import argparse
import json
from pathlib import Path

ORDERS = {
    '01': [[512], [512], [1, 8, 64, 64], [1, 8, 64, 64], [1, 1, 512], [1, 64],
           [1, 32], [1, 32], [1, 1, 512], [1], [1, 8, 1, 64]],
    '04': [[512], [512], [1, 8, 64, 64], [1, 8, 64, 64], [1, 64], [1, 32], [1, 32],
           [1, 1, 512], [1, 1, 512], [1], [1, 8, 1, 64]],
    '06': [[1, 32], [1, 32], [1, 1, 512], [1, 1, 512], [1, 8, 64, 64], [1],
           [1, 8, 1, 64], [512], [512], [1, 8, 64, 64], [1, 64]],
}


def main():
    """Reorder the graph input array for one delegate JSON document."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    args = parser.parse_args()
    document = json.loads(args.source.read_text(encoding='utf-8'))
    graph = document['regions'][0]['blocks'][0]
    tensors = {tensor['name']: tensor for tensor in graph['tensors']}
    index = args.source.name.split('_')[1]
    order = ORDERS[index]
    remaining = list(graph['inputs'])
    reordered = []
    for shape in order:
        for name in remaining:
            if tensors[name]['shape'] == shape:
                reordered.append(name)
                remaining.remove(name)
                break
        else:
            raise ValueError(f'No input with shape {shape}; remaining={remaining}')
    if remaining:
        raise ValueError(f'Unused inputs: {remaining}')
    graph['inputs'] = reordered
    args.target.write_text(json.dumps(document), encoding='utf-8')
    print('reordered', reordered)


if __name__ == '__main__':
    main()
