# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Export the exact TinyStories identity-normalized SentencePiece BPE vocabulary."""

import argparse
import struct
from pathlib import Path

from sentencepiece import sentencepiece_model_pb2


def export(model_path: Path, output: Path):
    """Export raw pieces, scores, normal-piece search order and byte token IDs.

    :param model_path:  SentencePiece model.
    :param output:      Board encoder table destination.
    """
    model = getattr(sentencepiece_model_pb2, 'ModelProto')()
    model.ParseFromString(model_path.read_bytes())
    spec = model.normalizer_spec
    if not all((model.trainer_spec.model_type == 2, model.trainer_spec.byte_fallback,
                spec.name == 'identity', spec.add_dummy_prefix,
                not spec.remove_extra_whitespaces, spec.escape_whitespaces,
                not model.HasField('denormalizer_spec'),
                all(piece.type != 4 for piece in model.pieces))):
        raise ValueError('Unsupported tokenizer: require identity BPE with byte fallback')
    pieces = [piece.piece.encode('utf-8') for piece in model.pieces]
    normal = sorted((index for index, piece in enumerate(model.pieces) if piece.type == 1),
                    key=lambda index: pieces[index])
    byte_ids = [-1] * 256
    for index, piece in enumerate(model.pieces):
        if piece.type == 6:
            byte_ids[int(piece.piece[3:5], 16)] = index
    if len(pieces) != 32000 or min(byte_ids) < 0 or model.trainer_spec.bos_id != 1:
        raise ValueError('Unexpected vocabulary, BOS or byte fallback IDs')
    offsets = [0]
    for piece in pieces:
        offsets.append(offsets[-1] + len(piece))
    binary = b'TSE1' + struct.pack('<II', len(pieces), len(normal))
    binary += struct.pack(f'<{len(offsets)}I', *offsets)
    binary += struct.pack(f'<{len(pieces)}f', *(piece.score for piece in model.pieces))
    binary += struct.pack(f'<{len(normal)}I', *normal)
    binary += struct.pack('<256I', *byte_ids) + b''.join(pieces)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(binary)
    print(f'TinyStories encoder table: {len(binary)} bytes; {len(normal)} normal pieces')


def main():
    """Export the specified tokenizer model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    export(args.model, args.output)


if __name__ == '__main__':
    main()
