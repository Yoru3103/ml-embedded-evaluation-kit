# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Microphone threshold, short noise rejection and hysteresis checks."""
import importlib
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/py"))
LevelGate = importlib.import_module("audio_level_gate").LevelGate


def block(amplitude: int) -> bytes:
    """Make one constant-energy 20 ms PCM16 interval.

    :param amplitude:   Signed PCM16 amplitude.
    :returns:           Little-endian audio block.
    """
    return struct.pack("<320h", *([amplitude] * 320))


class LevelGateTests(unittest.TestCase):
    """Verify energy decisions while preserving transport sample counts."""

    def test_dbfs_and_threshold(self):
        """Digital level uses full-scale RMS and suppresses quieter background."""
        gate = LevelGate(-26)
        self.assertEqual(gate.process(block(1000)), bytes(640))
        self.assertAlmostEqual(gate.level, -30.309, places=2)
        self.assertEqual(gate.process(block(2000)), block(2000))

    def test_impulse_does_not_latch(self):
        """One loud interval cannot relax the subsequent start threshold."""
        gate = LevelGate(-26)
        gate.process(block(4000))
        self.assertFalse(gate.active)
        self.assertEqual(gate.process(block(1000)), bytes(640))

    def test_hysteresis_and_release(self):
        """Speech continuation survives lower energy; quiet restores the start gate."""
        gate = LevelGate(-26)
        for _ in range(6):
            gate.process(block(2000))
        self.assertTrue(gate.active)
        self.assertEqual(gate.process(block(1000)), block(1000))
        for _ in range(25):
            self.assertEqual(gate.process(block(0)), bytes(640))
        self.assertFalse(gate.active)
        self.assertEqual(gate.process(block(1000)), bytes(640))


if __name__ == "__main__":
    unittest.main()
