# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Compare greedy and sampled host generation for the exported TinyStories graph.

The script mirrors the firmware token selection (greedy, top-k, top-p and an
optional repetition penalty) so decode settings can be screened on a host before
they are measured on the board. It never changes the PTE or the firmware.
"""

import argparse
import importlib
import json
import random
from pathlib import Path

import torch
from pytorch_tokenizers import get_tokenizer

# The checkpoint contains CPU quantized-decomposed embedding operators.
importlib.import_module('executorch.exir.passes._quant_patterns_and_replacements')
importlib.import_module('torchao.quantization.pt2e.quantize_pt2e')

EOS_ID = 2


def reset_cache(model: torch.nn.Module):
    """Zero every persistent KV buffer for one independent request.

    :param model: Exported host graph.
    :raises ValueError: If the graph does not expose sixteen caches.
    """
    buffers = [buffer for name, buffer in model.named_buffers()
               if 'k_cache' in name or 'v_cache' in name]
    if len(buffers) != 16:
        raise ValueError(f'Expected 16 cache buffers, found {len(buffers)}')
    for buffer in buffers:
        buffer.zero_()


def apply_repetition_penalty(logits: torch.Tensor, history: list[int], penalty: float):
    """Scale logits of recently used tokens (CTRL-style multiplicative penalty).

    :param logits:  One FP32 logits vector.
    :param history: Generated tokens in this request.
    :param penalty: Penalty factor; 1.0 disables the adjustment.
    :returns:       Adjusted logits vector.
    """
    if penalty <= 1.0:
        return logits
    for token in set(history):
        if logits[token] > 0:
            logits[token] /= penalty
        else:
            logits[token] *= penalty
    return logits


def select_token(logits: torch.Tensor, args, history: list[int], rng: random.Random) -> int:
    """Select one token with the same rules as the firmware sampler.

    :param logits:  One FP32 logits vector.
    :param args:    Parsed command line settings.
    :param history: Generated tokens in this request.
    :param rng:     Deterministic Python generator.
    :returns:       Selected token.
    """
    values = apply_repetition_penalty(logits.clone(), history, args.repetition_penalty)
    if args.mode == 'greedy':
        return int(torch.argmax(values))
    top = torch.topk(values, min(args.top_k, values.numel()))
    scaled = top.values / args.temperature
    probabilities = torch.softmax(scaled, dim=-1)
    cumulative = torch.cumsum(probabilities, dim=-1)
    keep = int(torch.searchsorted(cumulative, torch.tensor(args.top_p))) + 1
    keep = max(1, min(keep, top.values.numel()))
    probabilities = probabilities[:keep]
    probabilities = probabilities / probabilities.sum()
    threshold = rng.random()
    running = 0.0
    for index, probability in enumerate(probabilities.tolist()):
        running += probability
        if threshold < running:
            return int(top.indices[index])
    return int(top.indices[keep - 1])


def generate(model: torch.nn.Module, tokenizer, prompt: str, args) -> dict:
    """Run one greedy or sampled continuation.

    :param model:     Exported host graph.
    :param tokenizer: SentencePiece tokenizer.
    :param prompt:    Text prompt.
    :param args:      Parsed command line settings.
    :returns:         Tokens, text and quality counters.
    """
    reset_cache(model)
    prompt_tokens = tokenizer.encode(prompt, bos=True, eos=False)
    budget = min(args.max_new, args.context - len(prompt_tokens))
    tokens = list(prompt_tokens)
    generated = []
    rng = random.Random(args.seed)
    with torch.no_grad():
        for position in range(len(prompt_tokens) + budget - 1):
            logits = model(torch.tensor([[tokens[position]]], dtype=torch.int64),
                           {'input_pos': torch.tensor([position], dtype=torch.int32)})
            if position < len(prompt_tokens) - 1:
                continue
            next_token = select_token(logits.reshape(-1).float(), args, generated, rng)
            tokens.append(next_token)
            generated.append(next_token)
            if next_token == EOS_ID:
                break
    text = tokenizer.decode(tokens)
    grams = list(zip(generated, generated[1:], generated[2:], generated[3:]))
    distinct = len(set(grams)) / len(grams) if grams else 0.0
    return {'prompt': prompt, 'prompt_tokens': len(tokens) - len(generated),
            'generated': generated, 'text': text,
            'eos': EOS_ID in generated, 'tokens': len(generated),
            'distinct_4': distinct, 'repeat_4': 1.0 - distinct}


def main():
    """Score one decode configuration across the supplied prompts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('tokenizer', type=Path)
    parser.add_argument('--prompt', action='append', required=True)
    parser.add_argument('--mode', choices=('greedy', 'sample'), default='sample')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--top-k', type=int, default=64)
    parser.add_argument('--repetition-penalty', type=float, default=1.0)
    parser.add_argument('--max-new', type=int, default=128)
    parser.add_argument('--context', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = torch.export.load(args.checkpoint).module(check_guards=False)
    tokenizer = get_tokenizer(str(args.tokenizer))
    results = [generate(model, tokenizer, prompt, args) for prompt in args.prompt]
    settings = {key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()}
    settings['checkpoint'] = str(args.checkpoint)
    settings['tokenizer'] = str(args.tokenizer)
    payload = {'settings': settings, 'results': results}
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    for result in results:
        print(json.dumps({key: result[key] for key in
                          ('prompt', 'tokens', 'eos', 'distinct_4', 'repeat_4')}))
        print(result['text'])
        print('-' * 80)


if __name__ == '__main__':
    main()
