# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Exercise binary framing, corruption handling and the real C++ header writer."""
import io
import os
from pathlib import Path
import subprocess
import unittest
import zlib

from mlek_tools.pico_face.receive import (
    HEADER, MAGIC, OUTPUT_DIRECTORY, read_frame, resolve_output_prefix,
)


class OutputPathTests(unittest.TestCase):
    """Keep both old command examples and new defaults inside the image directory."""

    def test_relative_and_legacy_prefixes(self):
        """Resolve equivalent user arguments to the same isolated output prefix."""
        expected = OUTPUT_DIRECTORY / "face"
        for value in ("face", "out/face", "build/logs/pico_face_logs/out/face"):
            with self.subTest(value=value):
                self.assertEqual(resolve_output_prefix(Path(value), 3, 8), expected)
        self.assertEqual(resolve_output_prefix(expected, 3, 8), expected)

    def test_default_includes_seed_and_steps(self):
        """Keep the automatic name independent of the shell working directory."""
        output = resolve_output_prefix(None, 7, 8)
        self.assertEqual(output.parent, OUTPUT_DIRECTORY)
        self.assertTrue(output.name.startswith("face_seed7_steps8_"))

    def test_rejects_escape(self):
        """Reject traversal and absolute paths outside the dedicated output directory."""
        for value in (Path("../face"), OUTPUT_DIRECTORY.parent / "face"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_output_prefix(value, 3, 8)


class FragmentedStream(io.BytesIO):
    """Model a serial stream that returns only a few bytes per read."""
    def read(self, size: int = -1) -> bytes:
        """Return at most seven bytes, regardless of the caller's request."""
        return super().read(min(size, 7))


class SerialTests(unittest.TestCase):
    """Check that log text and arbitrary RGB bytes survive framing unchanged."""
    def setUp(self):
        """Build an independent wire-format fixture with every possible byte value."""
        self.rgb = bytes(range(256)) * 192
        self.header = HEADER.pack(MAGIC, 1, 128, 128, 3, 49152,
                                  3, 4, 1, 4.0, zlib.crc32(self.rgb), 123456789)

    def test_logs_and_fragmentation(self):
        """Skip preceding logs and preserve LF, NUL and XON/XOFF in the payload."""
        frame = read_frame(FragmentedStream(b"logs\nPFIMAGE0\n" + self.header + self.rgb))
        self.assertEqual(frame.rgb, self.rgb)
        self.assertEqual((frame.seed, frame.steps, frame.class_id), (3, 4, 1))
        self.assertEqual(frame.generation_us, 123456789)

    def test_crc_rejects_changed_data(self):
        """Reject even a single changed byte rather than saving a damaged image."""
        bad = self.rgb[:-1] + bytes([self.rgb[-1] ^ 1])
        with self.assertRaisesRegex(ValueError, "CRC mismatch"):
            read_frame(io.BytesIO(self.header + bad))

    def test_truncated_payload(self):
        """Fail clearly when the serial capture ends inside the image."""
        with self.assertRaises(EOFError):
            read_frame(io.BytesIO(self.header + self.rgb[:-1]))

    def test_rejects_invalid_length_before_reading(self):
        """Reject an unbounded length before attempting any payload allocation."""
        header = HEADER.pack(MAGIC, 1, 128, 128, 3, 0xffffffff,
                             3, 4, 1, 4.0, 0, 0)
        with self.assertRaises(ValueError):
            read_frame(io.BytesIO(header))

    def test_cpp_encoder_interoperability(self):
        """Decode an actual frame emitted by the firmware's C++ header encoder."""
        fixture = os.environ.get("PICO_FACE_FRAME_FIXTURE")
        if not fixture:
            self.skipTest("Build pico_face_serial_fixture for the integration test")
        wire = subprocess.run([fixture], check=True, capture_output=True).stdout
        frame = read_frame(FragmentedStream(wire))
        self.assertEqual(wire[:HEADER.size], self.header)
        self.assertEqual(frame.rgb, self.rgb)


if __name__ == "__main__":
    unittest.main()
