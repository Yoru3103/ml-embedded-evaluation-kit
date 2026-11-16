# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Synthetic checks for safe edge selection, fallback and waveform preservation."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[3] / "scripts/py/denoise_asr_wav.py"
SPEC = importlib.util.spec_from_file_location("denoise_asr_wav", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DenoiseTests(unittest.TestCase):
    """Verify behavior for unknown speech boundaries and unsuitable noise profiles."""

    def test_tail_selection_when_head_is_loud(self):
        """A loud start must be excluded while a sustained quiet tail is usable."""
        random = np.random.default_rng(42)
        audio = random.normal(0, 0.001, 48000)
        audio[:32000] += 0.15 * np.sin(2 * np.pi * 400 * np.arange(32000) / 16000)
        processed, report = MODULE.denoise(audio, MODULE.NoiseOptions())
        self.assertTrue(report["applied"])
        self.assertTrue(all(region["start_seconds"] >= 2 for region in report["regions"]))
        self.assertEqual(len(processed), len(audio))
        self.assertTrue(np.isfinite(processed).all())
        self.assertLess(np.mean(processed[-8000:] ** 2), np.mean(audio[-8000:] ** 2))
        _, head = MODULE.denoise(audio, MODULE.NoiseOptions(source="head"))
        self.assertFalse(head["applied"])

    def test_no_profile_is_exact_bypass(self):
        """Do not modify speech-like audio when neither edge passes the gate."""
        audio = 0.2 * np.sin(2 * np.pi * 300 * np.arange(32000) / 16000)
        processed, report = MODULE.denoise(audio, MODULE.NoiseOptions())
        self.assertFalse(report["applied"])
        np.testing.assert_array_equal(audio, processed)

    def test_short_quiet_gap_is_not_profile(self):
        """Brief low-level gaps cannot satisfy the sustained-noise condition."""
        audio = np.full(32000, 0.1)
        audio[:800] = 0
        _, report = MODULE.denoise(audio, MODULE.NoiseOptions())
        self.assertFalse(report["applied"])

    def test_zero_attenuation_reconstructs(self):
        """Overlap-add must preserve alignment and length, including a partial hop."""
        audio = np.random.default_rng(7).normal(0, 0.001, 32137)
        processed, report = MODULE.denoise(audio, MODULE.NoiseOptions(reduction_db=0))
        self.assertTrue(report["applied"])
        np.testing.assert_allclose(audio, processed, atol=1e-12, rtol=1e-10)

    def test_silence_is_finite(self):
        """An all-zero noise profile must produce finite all-zero output."""
        processed, _ = MODULE.denoise(np.zeros(16000), MODULE.NoiseOptions())
        np.testing.assert_array_equal(processed, np.zeros(16000))


if __name__ == "__main__":
    unittest.main()
