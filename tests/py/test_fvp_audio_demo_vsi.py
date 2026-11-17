# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Validate VSI callback compatibility, failure reporting and the binary contract."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "py"
sys.path.insert(0, str(SCRIPT_DIR))

arm_vsi0 = importlib.import_module("fvp_audio_demo_vsi.arm_vsi0")
protocol = importlib.import_module("fvp_audio_demo_vsi.protocol")


class ProtocolTests(unittest.TestCase):
    """Protect sample widths, frame size and firmware flag semantics."""

    def test_roundtrip_and_padding(self):
        """Carry 64-bit sample timestamps and partial tails without changing valid PCM."""
        block = protocol.AudioBlock((1 << 40) + 9, b"\xff\x7f" * 17,
                                    protocol.DISCONTINUITY, 123)
        frame = block.encode()
        self.assertEqual(len(frame), 664)
        self.assertEqual(protocol.AudioBlock.decode(frame), block)
        self.assertEqual(frame[24 + 34:], bytes(640 - 34))

    def test_reject_invalid_frames(self):
        """Reject truncated frames, unknown flags and out-of-bounds sample counts."""
        valid = protocol.AudioBlock(pcm=b"\0\0").encode()
        unknown_flag = 1 << protocol.WAITING.bit_length()
        frames = [valid[:-1], b"FAIL" + valid[4:], valid + b"\0",
                  protocol.HEADER.pack(protocol.MAGIC, 0, 321, 0, 0) + bytes(640),
                  protocol.HEADER.pack(protocol.MAGIC, 0, 0, 0, 0) + bytes(640),
                  protocol.HEADER.pack(protocol.MAGIC, 0, 1, unknown_flag, 0) + bytes(640)]
        for frame in frames:
            with self.subTest(frame=frame[:24]), self.assertRaises(ValueError):
                protocol.AudioBlock.decode(frame)

    def test_unsigned_bounds_and_terminal_errors(self):
        """Reject signed/out-of-range fields and ERROR without EOS before packing."""
        blocks = [protocol.AudioBlock(-1, b"\0\0"),
                  protocol.AudioBlock(1 << 64, b"\0\0"),
                  protocol.AudioBlock((1 << 64) - 1, b"\0\0"),
                  protocol.AudioBlock(pcm=b"\0\0", backlog_samples=-1),
                  protocol.AudioBlock(pcm=b"\0\0", backlog_samples=1 << 32),
                  protocol.AudioBlock(pcm=b"\0\0", flags=-1),
                  protocol.AudioBlock(pcm=b"\0\0", flags=1 << 32),
                  protocol.AudioBlock(pcm=b"\0\0", flags=protocol.ERROR),
                  protocol.AudioBlock(pcm=b"\0")]
        for block in blocks:
            with self.subTest(block=block), self.assertRaises(ValueError):
                block.encode()
        block = protocol.AudioBlock((1 << 64) - 1, flags=protocol.EOS,
                                    backlog_samples=(1 << 32) - 1)
        self.assertEqual(protocol.AudioBlock.decode(block.encode()), block)


class CallbackTests(unittest.TestCase):
    """Exercise official names directly, including standalone v_path imports."""

    def tearDown(self):
        """Ensure no staged audio or terminal state leaks between tests."""
        arm_vsi0.init()

    def test_lazy_magic_and_noop_callbacks(self):
        """Initialization never connects and optional callbacks have official returns."""
        with patch.object(arm_vsi0, "Client") as client:
            arm_vsi0.init()
            self.assertEqual(arm_vsi0.rdRegs(1), protocol.MAGIC)
            client.assert_not_called()
        self.assertEqual(arm_vsi0.rdIRQ(), 0)
        self.assertEqual(arm_vsi0.wrIRQ(7), 7)
        self.assertEqual(arm_vsi0.wrTimer(0, 9), 9)
        self.assertEqual(arm_vsi0.wrDMA(0, 8), 8)
        self.assertIsNone(arm_vsi0.timerEvent())
        self.assertIsNone(arm_vsi0.wrDataDMA(bytearray(640), 640))
        self.assertEqual(arm_vsi0.wrRegs(7, 0), 0)
        self.assertEqual(arm_vsi0.rdRegs(7), 0)

    def test_metadata_and_completion(self):
        """Split the absolute position into low/high registers and consume DMA once."""
        block = protocol.AudioBlock((1 << 40) + 23, b"\x01\0" * 320, 2, 640)
        client = Mock()
        client.recv_bytes.return_value = block.encode()
        with patch.object(arm_vsi0, "Client", return_value=client):
            arm_vsi0.init()
            arm_vsi0.wrRegs(0, 1)
        self.assertEqual(arm_vsi0.Regs[1:7], [protocol.MAGIC, 23, 256, 320, 2, 640])
        arm_vsi0.wrRegs(7, 0)
        self.assertEqual(arm_vsi0.rdDataDMA(640), block.pcm)
        self.assertEqual(arm_vsi0.rdRegs(7), 1)
        self.assertEqual(arm_vsi0.rdDataDMA(640), bytes(640))

    def test_connection_failure_is_visible(self):
        """Never hide a missing host behind apparently valid silent audio."""
        output = io.StringIO()
        with patch.object(arm_vsi0, "Client", side_effect=OSError("refused")) as client, \
                contextlib.redirect_stderr(output):
            arm_vsi0.init()
            arm_vsi0.wrRegs(0, 1)
            arm_vsi0.wrRegs(0, 1)
        self.assertEqual(arm_vsi0.rdRegs(5), protocol.ERROR | protocol.EOS)
        self.assertEqual(arm_vsi0.rdRegs(4), 0)
        self.assertIn("fvp_audio_demo.py --wav", output.getvalue())
        self.assertIn("6010", output.getvalue())
        client.assert_called_once()

    def test_disconnect_after_valid_block_preserves_position(self):
        """A broken host connection ends at the last valid absolute sample offset."""
        client = Mock()
        client.recv_bytes.side_effect = [protocol.AudioBlock(100, b"\0\0" * 320).encode(),
                                         EOFError()]
        with patch.object(arm_vsi0, "Client", return_value=client), \
                contextlib.redirect_stderr(io.StringIO()):
            arm_vsi0.init()
            arm_vsi0.wrRegs(0, 1)
            arm_vsi0.wrRegs(0, 1)
        self.assertEqual(arm_vsi0.rdRegs(2), 420)
        self.assertEqual(arm_vsi0.rdRegs(5), protocol.ERROR | protocol.EOS)
        self.assertEqual(arm_vsi0.rdDataDMA(640), bytes(640))

    def test_standalone_fvp_import(self):
        """Load arm_vsi0 with only v_path present, as the model does."""
        vsi_path = SCRIPT_DIR / "fvp_audio_demo_vsi"
        with patch.object(sys, "path", [str(vsi_path), *sys.path]):
            spec = importlib.util.spec_from_file_location("arm_vsi0", vsi_path / "arm_vsi0.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module.rdRegs(1), protocol.MAGIC)
            self.assertIsNone(module.BRIDGE.connection)


if __name__ == "__main__":
    unittest.main()
