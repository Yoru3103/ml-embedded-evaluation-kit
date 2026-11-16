# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Compare the actual board encoder and decoder against SentencePiece."""

import argparse
import random
import subprocess
from pathlib import Path

import sentencepiece as spm


def make_cases() -> list[str]:
    """Return deterministic natural, Unicode and randomized BPE test prompts."""
    cases = ['Once upon a time in a small village,', 'Hello world!', ' ', '  a   b  ',
             'a' * 256, '🙂', '中文提示词', '<s>hello</s>', '▁', 'é e\u0301',
             'Once upon a time, there was a little robot.', 'A cat found a magic key.']
    generator = random.Random(42)
    alphabet = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?\'"-'
    alphabet += '中文猫故事🙂🚀éαЖ\u2581'
    for _ in range(1500):
        text = ''.join(generator.choice(alphabet) for _ in range(generator.randrange(1, 90)))
        cases.append(text)
    return cases


def check_line(tokenizer: spm.SentencePieceProcessor, text: str, line: str):
    """Compare one response with the original SentencePiece implementation.

    :param tokenizer:  Original tokenizer processor.
    :param text:       Input prompt.
    :param line:       C++ harness response.
    """
    expected = tokenizer.Encode(text, add_bos=True)
    if line == 'ERR':
        raise AssertionError(f'Unexpected encoder rejection: {text!r}')
    ids, decoded = line.split('\t')
    actual = list(map(int, ids.split(',')))
    if actual != expected:
        raise AssertionError(f'BPE mismatch for {text!r}: {actual} != {expected}')
    if bytes.fromhex(decoded).decode('utf-8') != tokenizer.Decode(expected):
        raise AssertionError(f'Decoder mismatch for {text!r}')


def verify(executable: Path, model: Path, encoder: Path, decoder: Path):
    """Exercise BPE merges, whitespace, byte fallback and prompt length boundaries.

    :param executable:  Compiled TokenizerCli test harness.
    :param model:       Original SentencePiece tokenizer model.
    :param encoder:     Exported encoder table.
    :param decoder:     Exported decoder table.
    """
    tokenizer = spm.SentencePieceProcessor(model_file=str(model))
    cases = make_cases()
    invalid = [b'', b'\xff', b'\xed\xa0\x80', b'\xf0\x80\x80\x80', b'a' * 257]
    data = [text.encode('utf-8') for text in cases] + invalid
    result = subprocess.run([str(executable.resolve()), str(encoder), str(decoder)],
                            input='\n'.join(item.hex() for item in data) + '\n',
                            text=True, capture_output=True, check=True)
    lines = result.stdout.splitlines()
    if len(lines) != len(data):
        raise AssertionError('Tokenizer harness response count mismatch')
    for text, line in zip(cases, lines):
        check_line(tokenizer, text, line)
    if lines[len(cases):] != ['ERR'] * len(invalid):
        raise AssertionError('Expected malformed/oversized input rejection')
    print(f'PASS: {len(cases)} prompts match SentencePiece IDs and text; '
          f'{len(invalid)} invalid inputs rejected')
    check_capacity(executable, tokenizer, encoder, decoder)


def check_capacity(executable: Path, tokenizer: spm.SentencePieceProcessor,
                   encoder: Path, decoder: Path):
    """Check the actual 63-token prompt bound without truncating accepted inputs.

    :param executable:  Compiled tokenizer harness.
    :param tokenizer:   Original SentencePiece processor.
    :param encoder:     Exported encoder table.
    :param decoder:     Exported decoder table.
    """
    cases = ['Once upon a time in a small village,', ' a' * 62, ' a' * 63]
    result = subprocess.run([str(executable.resolve()), str(encoder), str(decoder), '63'],
                            input='\n'.join(text.encode().hex() for text in cases) + '\n',
                            text=True, capture_output=True, check=True)
    lines = result.stdout.splitlines()
    if len(lines) != len(cases):
        raise AssertionError('Capacity test response count mismatch')
    for text, line in zip(cases, lines):
        if len(tokenizer.Encode(text, add_bos=True)) > 63:
            if line != 'ERR':
                raise AssertionError('Oversized token input was accepted or truncated')
        else:
            check_line(tokenizer, text, line)
    print('PASS: 63-token capacity accepts complete prompts and rejects overflow')


def main():
    """Verify exported tables using the actual C++ implementation."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('executable', 'model', 'encoder', 'decoder'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    verify(args.executable, args.model, args.encoder, args.decoder)


if __name__ == '__main__':
    main()
