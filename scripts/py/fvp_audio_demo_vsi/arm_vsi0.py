# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Official VSI callbacks for the custom application pull/DMA protocol.

Point mps4_board.v_path at this directory. Set FVP_AUDIO_DEMO_PORT in the FVP
environment to override 6010. Only standard-library modules run inside FVP.
Register 0=1 fetches one frame; registers 1..6 publish metadata; register 7 is
the DMA completion handshake. No timer-driven audio capture or host VAD runs.
"""
# The FVP looks up these exact callback names from its official VSI Python API.
# pylint: disable=invalid-name
import importlib
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Connection
import os
import sys
import time
from typing import Optional

if __package__:
    from . import protocol
else:
    protocol = importlib.import_module("protocol")

Regs = [0] * 64


class PullBridge:
    """Keep connection and one-shot payload state separate from VSI callbacks."""

    def __init__(self):
        """Construct an idle bridge without connecting or starting acquisition."""
        self.connection: Optional[Connection] = None
        self.payload = b""
        self.terminal: Optional[protocol.AudioBlock] = None
        self.next_sample = 0

    def close(self):
        """Release the socket, including when the FVP reinitializes VSI0."""
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def fetch(self) -> protocol.AudioBlock:
        """Pull lazily, converting transport failures into a visible terminal error."""
        self.payload = b""
        if self.terminal is not None:
            return self.terminal
        try:
            port = int(os.environ.get("FVP_AUDIO_DEMO_PORT", "6010"))
            if self.connection is None:
                self.connection = Client(("127.0.0.1", port), authkey=protocol.AUTHKEY)
            self.connection.send_bytes(protocol.REQUEST)
            block = protocol.AudioBlock.decode(self.connection.recv_bytes(protocol.FRAME_BYTES))
            self.next_sample = block.first_sample + block.samples
        except (OSError, EOFError, ValueError, AuthenticationError) as error:
            print("FVP audio bridge failed: " + str(error) + ". Start the host with "
                  "python scripts/py/fvp_audio_demo.py --wav <16k-mono-pcm16.wav> "
                  "--port " + os.environ.get("FVP_AUDIO_DEMO_PORT", "6010")
                  + " (localhost); check FVP_AUDIO_DEMO_PORT.", file=sys.stderr, flush=True)
            block = protocol.AudioBlock(self.next_sample, flags=protocol.ERROR | protocol.EOS)
        if block.flags & protocol.EOS:
            self.terminal = protocol.AudioBlock(block.first_sample + block.samples,
                                               flags=block.flags)
            self.close()
        self.payload = block.pcm
        return block


BRIDGE = PullBridge()


def init():
    """Publish readiness before connecting; reset transport on model initialization."""
    BRIDGE.close()
    BRIDGE.payload = b""
    BRIDGE.terminal = None
    BRIDGE.next_sample = 0
    Regs[:] = [0] * 64
    Regs[1] = protocol.MAGIC


def rdIRQ() -> int:
    """Return no interrupt requests for the synchronous pull adapter."""
    return 0


def wrIRQ(value: int) -> int:
    """Accept and echo the unused interrupt register write."""
    return value


def wrTimer(unused_index: int, value: int) -> int:
    """Echo timer writes; the model itself schedules the DMA trigger."""
    return value


def timerEvent():
    """Leave audio acquisition to explicit register-zero pulls."""


def wrDMA(unused_index: int, value: int) -> int:
    """Echo DMA writes; one-shot data is provided by rdDataDMA."""
    return value


def rdDataDMA(size: int) -> bytearray:
    """Consume staged PCM once, pad to DMA size, then acknowledge register 7."""
    data = bytearray(BRIDGE.payload[:size].ljust(size, b"\0"))
    BRIDGE.payload = b""
    Regs[7] = 1
    return data


def wrDataDMA(unused_data: bytearray, unused_size: int):
    """Ignore memory-to-peripheral transfers on this input-only adapter."""


def rdRegs(index: int) -> int:
    """Read model user-register state, including DMA completion."""
    if index == 8:
        return int(time.monotonic() * 1000) & 0xFFFFFFFF
    return Regs[index]


def wrRegs(index: int, value: int) -> int:
    """Echo writes; register zero requests synchronously publish a source block."""
    Regs[index] = value
    if index == 0 and value == 1:
        block = BRIDGE.fetch()
        Regs[1:7] = [protocol.MAGIC, block.first_sample & 0xFFFFFFFF,
                     block.first_sample >> 32, block.samples, block.flags,
                     block.backlog_samples]
    return value


init()
