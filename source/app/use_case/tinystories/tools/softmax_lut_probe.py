# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Compare exact host softmax with the Ethos-U TABLE-based exp approximation.

The probe runs the fixed TinyStories TEST prompt through the quantized PT2,
optionally replacing every ``aten.exp`` with the 513-entry INT16 table and
linear interpolation used by Vela.  It reports top-1/top-5 per position and the
generated token sequence, so the board divergence can be attributed to the exp
lookup table before any firmware change.
"""

import argparse
import importlib
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from pytorch_tokenizers import get_tokenizer
from torch.utils._python_dispatch import TorchDispatchMode
from tosa.TosaGraph import TosaGraph

importlib.import_module('executorch.exir.passes._quant_patterns_and_replacements')
importlib.import_module('torchao.quantization.pt2e.quantize_pt2e')

PROMPT = [1, 9038, 2501, 263, 931, 297, 263, 2319, 5720, 29892]
OUTPUT_SCALE = 0.000244140625
TABLE_POINTS = 513


def load_table(path: Path, name: str) -> np.ndarray:
    """Read one 513-entry INT16 table from the exported TOSA graph.

    :param path: TOSA flatbuffer captured before Vela.
    :param name: Tensor name of the table constant.
    :returns:    Table values as float32.
    """
    buffer = path.read_bytes()
    graph = TosaGraph.GetRootAsTosaGraph(buffer, 0)
    block = graph.Regions(0).Blocks(0)
    for index in range(block.TensorsLength()):
        tensor = block.Tensors(index)
        if tensor.Name().decode() == name:
            raw = tensor.DataAsNumpy().tobytes()
            return np.frombuffer(raw, dtype='<i2').astype(np.float32)
    raise ValueError(f'Table {name} not found in {path}')


def synthesise_table(scale: float) -> np.ndarray:
    """Build the table Vela would generate for a given exp input scale.

    :param scale: Physical value of one INT16 code.
    :returns:     513 INT16-valued samples of exp over the code domain.
    """
    codes = -32768.0 + (65535.0 / (TABLE_POINTS - 1)) * np.arange(TABLE_POINTS)
    values = np.exp(np.minimum(codes * scale, 80.0)) * 32767.0
    return np.clip(np.rint(values), 0, 32767).astype(np.float32)


class ExpLutMode(TorchDispatchMode):
    """Replace aten.exp with a quantized table lookup and linear interpolation."""

    def __init__(self, table: np.ndarray, input_scale: float):
        self.table = torch.tensor(table, dtype=torch.float32)
        self.input_scale = float(input_scale)
        self.calls = 0

    def _lookup(self, values: torch.Tensor) -> torch.Tensor:
        quantized = torch.round(values / self.input_scale).clamp(-32768, 32767)
        position = (quantized + 32768.0) * ((TABLE_POINTS - 1) / 65535.0)
        lower = torch.floor(position).long().clamp(0, TABLE_POINTS - 2)
        fraction = (position - lower.float()).clamp(0.0, 1.0)
        result = (self.table[lower] * (1.0 - fraction) +
                  self.table[lower + 1] * fraction) / 32767.0
        return torch.round(result / OUTPUT_SCALE) * OUTPUT_SCALE

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.exp.default:
            self.calls += 1
            return self._lookup(args[0].float())
        return func(*args, **(kwargs or {}))


def reset_cache(model: torch.nn.Module):
    """Zero the sixteen persistent KV buffers.

    :param model: Exported host graph.
    """
    buffers = [buffer for name, buffer in model.named_buffers()
               if 'k_cache' in name or 'v_cache' in name]
    if len(buffers) != 16:
        raise ValueError(f'Expected 16 cache buffers, found {len(buffers)}')
    for buffer in buffers:
        buffer.zero_()


def run(model: torch.nn.Module, lut: ExpLutMode | None, max_new: int) -> dict:
    """Run the fixed TEST sequence and record logits and token choices.

    :param model:   Exported host graph.
    :param lut:     Optional exp table emulation.
    :param max_new: Number of continuation tokens.
    :returns:       Position records and decoded token ids.
    """
    reset_cache(model)
    tokens = list(PROMPT)
    records = []
    context = lut if lut is not None else torch.no_grad()
    with torch.no_grad(), context:
        for position in range(len(PROMPT) + max_new - 1):
            logits = model(
                torch.tensor([[tokens[position]]], dtype=torch.int64),
                {'input_pos': torch.tensor([position], dtype=torch.int32)},
            ).reshape(-1).float()
            values, indices = torch.topk(logits, 5)
            record = {
                'position': position,
                'top1': int(indices[0]),
                'top5': [int(value) for value in indices],
                'values': [float(value) for value in values],
                'margin': float(values[0] - values[1]),
            }
            records.append(record)
            if position >= len(PROMPT) - 1:
                tokens.append(record['top1'])
    return {'records': records, 'tokens': tokens}


def parse_board_log(path: Path) -> dict[int, list[int]]:
    """Parse TINYSTORIES_LOGITS lines from a captured UART TEST log.

    :param path: Raw UART capture.
    :returns:    Position to top-5 token ids.
    """
    text = path.read_bytes().decode('utf-8', 'replace')
    pattern = re.compile(r'TINYSTORIES_LOGITS position=(\d+) id0=(-?\d+) v0=[-\d.]+ '
                         r'id1=(-?\d+) v1=[-\d.]+ id2=(-?\d+) v2=[-\d.]+ '
                         r'id3=(-?\d+) v3=[-\d.]+ id4=(-?\d+) v4=[-\d.]+')
    result = {}
    for match in pattern.finditer(text):
        result[int(match.group(1))] = [int(match.group(i)) for i in range(2, 7)]
    return result


def compare(name: str, base: dict, other: dict, tokenizer) -> dict:
    """Compare two runs over the prompt and generated positions.

    :param name:     Comparison label.
    :param base:     Reference run.
    :param other:    Candidate run.
    :param tokenizer: Tokenizer used to decode the final text.
    :returns:        Summary with top-1 matches, divergence and overlap.
    """
    base_records = base['records']
    other_records = other['records']
    prompt_matches = sum(
        base_records[i]['top1'] == other_records[i]['top1'] for i in range(len(PROMPT)))
    prompt_overlap = sum(
        len(set(base_records[i]['top5']) & set(other_records[i]['top5']))
        for i in range(len(PROMPT))) / len(PROMPT)
    divergence = next(
        (i for i in range(len(base_records)) if base_records[i]['top1'] != other_records[i]['top1']),
        None)
    generated_base = [record['top1'] for record in base_records[len(PROMPT) - 1:]]
    generated_other = [record['top1'] for record in other_records[len(PROMPT) - 1:]]
    generated_matches = sum(a == b for a, b in zip(generated_base, generated_other))
    return {
        'name': name,
        'prompt_top1_matches': prompt_matches,
        'prompt_top1_total': len(PROMPT),
        'prompt_top5_overlap_mean': prompt_overlap,
        'first_divergence': divergence,
        'generated_matches': generated_matches,
        'generated_total': len(generated_base),
        'generated_base': generated_base,
        'generated_other': generated_other,
        'text': tokenizer.decode(base['tokens']),
        'other_text': tokenizer.decode(other['tokens']),
    }


def main():
    """Run exact, LUT and optional -30 LUT variants and write a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('tokenizer', type=Path)
    parser.add_argument('tosa', type=Path)
    parser.add_argument('--exp-scale', type=float, default=0.00815)
    parser.add_argument('--neg-inf', type=float, default=None,
                        help='Also emulate this mask value with a synthesised table.')
    parser.add_argument('--max-new', type=int, default=16)
    parser.add_argument('--board-log', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    torch.set_num_threads(4)
    print('loading checkpoint...', flush=True)
    model = torch.export.load(args.checkpoint).module(check_guards=False)
    tokenizer = get_tokenizer(str(args.tokenizer))
    print('reading exp table...', flush=True)
    table = load_table(args.tosa, 'b_aten_exp_default_table_constant')
    print(f'table: {table.size} values, range {table.min():.0f}..{table.max():.0f}', flush=True)

    print('running exact host...', flush=True)
    exact = run(model, None, args.max_new)
    print('exact done', flush=True)
    print('running exp LUT...', flush=True)
    lut_mode = ExpLutMode(table, args.exp_scale)
    lut = run(model, lut_mode, args.max_new)
    print(f'lut done, intercepted {lut_mode.calls} exp calls', flush=True)
    results = [compare('LUT vs exact', exact, lut, tokenizer)]
    candidates = {'exact': exact, 'lut': lut}

    if args.neg_inf is not None:
        scale = abs(args.neg_inf) / 32767.0
        print(f'running exp LUT with neg_inf={args.neg_inf} (scale={scale:.9f})...', flush=True)
        fixed = run(model, ExpLutMode(synthesise_table(scale), scale), args.max_new)
        print('neg_inf lut done', flush=True)
        candidates['lut_neg_inf'] = fixed
        results.append(compare(f'LUT({args.neg_inf}) vs exact', exact, fixed, tokenizer))
        results.append(compare(f'LUT({args.neg_inf}) vs LUT', lut, fixed, tokenizer))

    if args.board_log:
        board = parse_board_log(args.board_log)
        for key in ('lut', 'lut_neg_inf'):
            if key not in candidates:
                continue
            records = candidates[key]['records']
            matches = sum(board.get(record['position'], [None])[0] == record['top1']
                          for record in records if record['position'] in board)
            results.append({
                'name': f'{key} vs board',
                'positions': len([r for r in records if r['position'] in board]),
                'top1_matches': matches,
            })

    payload = {
        'settings': {
            'checkpoint': str(args.checkpoint),
            'tosa': str(args.tosa),
            'exp_scale': args.exp_scale,
            'neg_inf': args.neg_inf,
            'table_min': float(table.min()),
            'table_max': float(table.max()),
        },
        'results': results,
        'runs': {
            key: {
                'prompt_top1': [record['top1'] for record in value['records'][:len(PROMPT)]],
                'generated': [record['top1'] for record in value['records'][len(PROMPT) - 1:]],
                'records': value['records'],
                'text': tokenizer.decode(value['tokens']),
            }
            for key, value in candidates.items()
        },
    }
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    for result in results:
        print(json.dumps(result, ensure_ascii=False))
    print('exact text:', tokenizer.decode(exact['tokens']))
    print('lut   text:', tokenizer.decode(lut['tokens']))


if __name__ == '__main__':
    main()
