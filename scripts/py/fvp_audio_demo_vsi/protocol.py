# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Fixed PCM16 frames shared by the host server and the FVP sub-interpreter.

Sample positions are absolute source sample offsets, starting at zero, rather
than simulation or wall-clock time. Divide by SAMPLE_RATE for seconds. The
24-byte header maps directly to firmware Block fields (not C struct padding).
"""
from dataclasses import dataclass
import struct

SAMPLE_RATE = 16000
BLOCK_SAMPLES = 320
BLOCK_BYTES = BLOCK_SAMPLES * 2
MAGIC = 0x41554431
EOS = 1
DISCONTINUITY = 2
ERROR = 4
AUTHKEY = b"fvp_audio_demo"
REQUEST = b"NEXT"
HEADER = struct.Struct("<IQIII")
FRAME_BYTES = HEADER.size + BLOCK_BYTES


@dataclass(frozen=True)
class AudioBlock:
    """One source interval and the number of queued samples remaining behind it."""

    first_sample: int = 0
    pcm: bytes = b""
    flags: int = 0
    backlog_samples: int = 0

    @property
    def samples(self) -> int:
        """Return the valid sample count, excluding DMA padding."""
        return len(self.pcm) // 2

    def encode(self) -> bytes:
        """Validate and serialize a fixed-size, zero-padded transport frame."""
        if not isinstance(self.first_sample, int) or not 0 <= self.first_sample < 1 << 64:
            raise ValueError("first_sample must be a representable uint64")
        if not isinstance(self.backlog_samples, int) or not 0 <= self.backlog_samples < 1 << 32:
            raise ValueError("backlog_samples must be a representable uint32")
        if not isinstance(self.flags, int) or not 0 <= self.flags < 1 << 32:
            raise ValueError("flags must be a representable uint32")
        if len(self.pcm) % 2 or len(self.pcm) > BLOCK_BYTES:
            raise ValueError("PCM must contain at most 320 complete PCM16 samples")
        if self.flags & ~(EOS | DISCONTINUITY | ERROR):
            raise ValueError("Unknown audio flags")
        if self.flags & ERROR and not self.flags & EOS:
            raise ValueError("ERROR requires EOS")
        if self.first_sample + self.samples >= 1 << 64:
            raise ValueError("Audio interval exceeds uint64 sample positions")
        if not self.samples and not self.flags & EOS:
            raise ValueError("Empty block requires EOS")
        return HEADER.pack(MAGIC, self.first_sample, self.samples, self.flags,
                           self.backlog_samples) + self.pcm.ljust(BLOCK_BYTES, b"\0")

    @classmethod
    def decode(cls, frame: bytes) -> "AudioBlock":
        """Validate a received frame before publishing metadata to firmware.

        :param frame:       Complete connection payload, including DMA padding.
        :returns:           Validated source block without padding.
        :raises ValueError: Invalid length, magic, sample count or flags.
        """
        if len(frame) != FRAME_BYTES:
            raise ValueError("Invalid audio frame length")
        magic, first, samples, flags, backlog = HEADER.unpack_from(frame)
        if magic != MAGIC or samples > BLOCK_SAMPLES:
            raise ValueError("Invalid audio magic or sample count")
        block = cls(first, frame[HEADER.size:HEADER.size + samples * 2], flags, backlog)
        block.encode()
        return block
