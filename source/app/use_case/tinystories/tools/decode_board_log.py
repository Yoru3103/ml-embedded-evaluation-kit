# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Compare a TinyStories UART log with the host PT2 reference and decode its tokens."""

import argparse
import json
import re
from pathlib import Path

from pytorch_tokenizers import get_tokenizer


def main():
    """Require complete ordered prompt/generation records and decode board token IDs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('tokenizer', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    text = args.log.read_text(encoding='utf-8-sig', errors='replace')
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    prompts = [(int(p), int(t), int(v)) for p, t, v in re.findall(
        r'TINYSTORIES position=(\d+) input=(\d+) top1=(\d+)', text)]
    generated = [(int(i), int(t)) for i, t in re.findall(
        r'TINYSTORIES_GENERATED index=(\d+) token=(\d+)', text)]
    if [p for p, _, _ in prompts] != list(range(len(reference['prompt_tokens']))):
        raise ValueError('Expected exactly one complete, ordered prompt run in the log')
    if [t for _, t, _ in prompts] != reference['prompt_tokens']:
        raise ValueError('Board prompt differs from the host reference')
    if not generated or [i for i, _ in generated] != list(range(len(generated))):
        raise ValueError('Missing or unordered generated token records')
    actual = [t for _, t in generated]
    expected = reference['generated_tokens']
    if len(actual) != len(expected) and actual[-1] != 2:
        raise ValueError('Generation log is incomplete and does not end with EOS')
    tokenizer = get_tokenizer(str(args.tokenizer))
    result = {
        'prompt_matches': sum(v == e for (_, _, v), e in
                              zip(prompts, reference['prompt_top1'])),
        'prompt_count': len(prompts),
        'prompt_mismatches': [{'position': p, 'actual': v, 'expected': e}
                              for (p, _, v), e in zip(prompts, reference['prompt_top1'])
                              if v != e],
        'generated_tokens': actual,
        'generation_matches': sum(a == b for a, b in zip(actual, expected)),
        'generation_exact': actual == expected,
        'board_text': tokenizer.decode(reference['prompt_tokens'] + actual),
        'reference_text': reference['text'],
        'timing_records': re.findall(r'TINYSTORIES_TIME [^\r\n]+', text),
        'reference_kind': reference['reference_kind'],
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered)


if __name__ == '__main__':
    main()
