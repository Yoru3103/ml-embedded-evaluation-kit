# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Export all SentencePiece token surfaces for the TinyStories C++ decoder."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2


def main():
    """Write a bounded offset table and token surfaces, preserving byte fallbacks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    model_bytes = args.model.read_bytes()
    model = sentencepiece_model_pb2.ModelProto()
    model.ParseFromString(model_bytes)
    if (model.HasField('denormalizer_spec') or not model.normalizer_spec.add_dummy_prefix
            or model.normalizer_spec.name != 'identity'
            or model.normalizer_spec.remove_extra_whitespaces):
        raise ValueError('Unsupported tokenizer normalization configuration')
    tokenizer = spm.SentencePieceProcessor(model_proto=model_bytes)
    surfaces = bytearray()
    offsets = []
    for token in range(tokenizer.vocab_size()):
        offsets.append(len(surfaces))
        flag = 0
        piece = tokenizer.id_to_piece(token)
        if tokenizer.is_control(token) or tokenizer.is_unused(token):
            surface = b''
        elif tokenizer.is_byte(token):
            surface = bytes([int(piece[3:5], 16)])
        elif tokenizer.is_unknown(token):
            surface = model.trainer_spec.unk_surface.encode('utf-8')
        else:
            flag = 1
            surface = piece.replace('\u2581', ' ').encode('utf-8')
        surfaces.append(flag)
        surfaces.extend(surface)
    offsets.append(len(surfaces))
    binary = b'TSV1' + struct.pack('<I', tokenizer.vocab_size())
    binary += struct.pack(f'<{len(offsets)}I', *offsets) + surfaces
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(binary)
    print(json.dumps({'vocabulary': tokenizer.vocab_size(), 'bytes': len(binary),
                      'tokenizer_sha256': hashlib.sha256(model_bytes).hexdigest(),
                      'table_sha256': hashlib.sha256(binary).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
