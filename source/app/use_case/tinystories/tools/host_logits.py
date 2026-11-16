# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Print the same top-five logits the TEST firmware can dump with LOGIT_DUMP."""

import argparse
import importlib
from pathlib import Path

import torch
from pytorch_tokenizers import get_tokenizer

importlib.import_module('executorch.exir.passes._quant_patterns_and_replacements')
importlib.import_module('torchao.quantization.pt2e.quantize_pt2e')

PROMPT = [1, 9038, 2501, 263, 931, 297, 263, 2319, 5720, 29892]
POSITIONS = 25


def main():
    """Run the fixed TEST sequence and print one line per position."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('tokenizer', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = torch.export.load(args.checkpoint).module(check_guards=False)
    tokenizer = get_tokenizer(str(args.tokenizer))
    buffers = [buffer for name, buffer in model.named_buffers()
               if 'k_cache' in name or 'v_cache' in name]
    if len(buffers) != 16:
        raise ValueError(f'Expected 16 cache buffers, found {len(buffers)}')
    for buffer in buffers:
        buffer.zero_()
    tokens = list(PROMPT)
    with torch.no_grad():
        for position in range(POSITIONS):
            logits = model(torch.tensor([[tokens[position]]], dtype=torch.int64),
                           {'input_pos': torch.tensor([position], dtype=torch.int32)})
            values, indices = torch.topk(logits.reshape(-1).float(), 5)
            values = values.tolist()
            indices = indices.tolist()
            print(f'TINYSTORIES_LOGITS position={position} '
                  f'id0={indices[0]} v0={values[0]:.6f} '
                  f'id1={indices[1]} v1={values[1]:.6f} '
                  f'id2={indices[2]} v2={values[2]:.6f} '
                  f'id3={indices[3]} v3={values[3]:.6f} '
                  f'id4={indices[4]} v4={values[4]:.6f} '
                  f'margin={values[0] - values[1]:.6f}')
            if position >= len(PROMPT) - 1:
                tokens.append(indices[0])
    print(f'# prompt={PROMPT} tokenizer={args.tokenizer} checkpoint={args.checkpoint}')
    print(f'# decoded={tokenizer.decode(tokens)}')


if __name__ == '__main__':
    main()
