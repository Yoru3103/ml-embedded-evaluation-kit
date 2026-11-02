# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""PCM16 microphone energy gate with digital dBFS thresholds."""
from array import array
import math
import sys


class LevelGate:
    """Retain speech continuation down to 6 dB below the opening threshold."""

    def __init__(self, threshold_dbfs: float, release_ms: int = 500):
        """Set the digital threshold and quiet release interval.

        :param threshold_dbfs:  RMS level relative to full-scale PCM16.
        :param release_ms:      Quiet interval before restoring the start threshold.
        """
        self.threshold = threshold_dbfs
        self.release_samples = release_ms * 16
        self.active = False
        self.loud_samples = 0
        self.quiet_samples = 0
        self.level = -120.0

    def reset(self):
        """Discard speech history after a capture discontinuity."""
        self.active = False
        self.loud_samples = 0
        self.quiet_samples = 0
        self.level = -120.0

    def process(self, pcm: bytes) -> bytes:
        """Gate low-level input without removing samples or changing timestamps.

        :param pcm: Little-endian mono PCM16 samples.
        :returns:   Original samples above threshold, zero samples below it.
        """
        samples = array("h")
        samples.frombytes(pcm)
        if sys.byteorder != "little":
            samples.byteswap()
        rms = math.sqrt(sum(value * value for value in samples) / max(1, len(samples)))
        self.level = 20 * math.log10(max(rms, 0.032768) / 32768)
        loud = self.level >= self.threshold - (6 if self.active else 0)
        if loud:
            self.loud_samples += len(samples)
            self.quiet_samples = 0
            if self.loud_samples >= 1920:
                self.active = True
        else:
            self.loud_samples = 0
            self.quiet_samples += len(samples)
            if self.quiet_samples >= self.release_samples:
                self.active = False
        return pcm if loud else bytes(len(pcm))
