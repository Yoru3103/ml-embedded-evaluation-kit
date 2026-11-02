# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Bounded, lazy audio sources; microphone capture lives outside FVP Python."""
from collections import deque
from dataclasses import replace
import importlib
from pathlib import Path
import threading
import time
from typing import Any, Optional
import wave
from audio_level_gate import LevelGate

from fvp_audio_demo_vsi.protocol import (
    AudioBlock, BLOCK_SAMPLES, DISCONTINUITY, EOS, ERROR, SAMPLE_RATE,
)


class BlockQueue:
    """Bound memory with WAV backpressure or oldest-block microphone eviction."""

    def __init__(self, queue_ms: int, drop_oldest: bool):
        """Create a queue rounded down to complete 20 ms blocks.

        :param queue_ms:    Maximum queued audio duration, at least 20 ms.
        :param drop_oldest: Evict stale microphone blocks instead of blocking.
        """
        if queue_ms < 20:
            raise ValueError("--queue-ms must be at least 20")
        self.capacity = queue_ms // 20
        self.drop_oldest = drop_oldest
        self.blocks: deque[AudioBlock] = deque()
        self.condition = threading.Condition()
        self.terminal: Optional[AudioBlock] = None
        self.stopped = False
        self.expected_sample = 0

    def put(self, block: AudioBlock) -> bool:
        """Enqueue a bounded PCM block, waiting only for file-source backpressure.

        :param block:   Source PCM interval with its original absolute position.
        :returns:       False when shutdown interrupts a blocked producer.
        """
        block.encode()
        with self.condition:
            while len(self.blocks) >= self.capacity and not self.stopped:
                if self.drop_oldest:
                    discarded = self.blocks.popleft()
                    if discarded.flags & DISCONTINUITY:
                        if self.blocks:
                            self.blocks[0] = replace(
                                self.blocks[0], flags=self.blocks[0].flags | DISCONTINUITY)
                        else:
                            block = replace(block, flags=block.flags | DISCONTINUITY)
                    break
                self.condition.wait()
            if self.stopped:
                return False
            self.blocks.append(block)
            self.condition.notify_all()
            return True

    def finish(self, first_sample: int, flags: int = 0):
        """Publish a stable terminal block after draining queued source samples.

        :param first_sample:    Absolute position just after acquisition ended.
        :param flags:           Optional ERROR flag for failed acquisition.
        """
        with self.condition:
            self.terminal = AudioBlock(first_sample, flags=flags | EOS)
            self.condition.notify_all()

    def get(self) -> AudioBlock:
        """Wait for source data, marking sequence gaps and reporting queued samples."""
        with self.condition:
            while not self.blocks and self.terminal is None and not self.stopped:
                self.condition.wait()
            if self.blocks:
                block = self.blocks.popleft()
            else:
                block = self.terminal or AudioBlock(self.expected_sample, flags=EOS)
            if block.first_sample != self.expected_sample:
                block = replace(block, flags=block.flags | DISCONTINUITY)
            self.expected_sample = block.first_sample + block.samples
            block = replace(block, backlog_samples=sum(item.samples for item in self.blocks))
            self.condition.notify_all()
            return block

    def close(self):
        """Wake blocked producers and consumers during disconnect or shutdown."""
        with self.condition:
            self.stopped = True
            self.blocks.clear()
            self.condition.notify_all()


def sounddevice_module() -> Any:
    """Import optional microphone support with an actionable installation hint."""
    try:
        return importlib.import_module("sounddevice")
    except ImportError as error:
        raise RuntimeError(
            "Microphone support requires: python -m pip install sounddevice") from error


class AudioSource:
    """Start one bounded producer on the first pull, preserving source sample time."""

    def __init__(self, options: Any):
        """Store CLI options without opening the WAV or microphone.

        :param options: Parsed wav, device, duration, realtime and queue_ms options.
        """
        self.options = options
        self.queue = BlockQueue(options.queue_ms, drop_oldest=options.wav is None)
        self.stop = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.position = 0
        self.error: Optional[str] = None

    def pull(self) -> AudioBlock:
        """Start acquisition exactly once and block until the next source interval."""
        if self.thread is None:
            self.thread = threading.Thread(target=self._produce, name="audio-source", daemon=True)
            self.thread.start()
        return self.queue.get()

    def close(self):
        """Stop acquisition and wake a WAV producer blocked by queue backpressure."""
        self.stop.set()
        self.queue.close()
        if self.thread is not None:
            self.thread.join(timeout=2)

    def _produce(self):
        """Drain the selected source and make all source failures visible as ERROR."""
        flags = 0
        try:
            if self.options.wav is not None:
                self._wav()
            else:
                self._microphone()
        except (OSError, ValueError, RuntimeError, wave.Error, ImportError) as error:
            self.error = str(error)
            flags = ERROR
        finally:
            self.queue.finish(self.position, flags)

    def _remaining(self) -> int:
        """Return the next bounded read size respecting an optional duration limit."""
        if self.options.duration is None:
            return BLOCK_SAMPLES
        return max(0, min(BLOCK_SAMPLES,
                          int(self.options.duration * SAMPLE_RATE) - self.position))

    def _wav(self):
        """Read PCM16 mono 16 kHz WAV incrementally; never drop file samples."""
        with wave.open(str(Path(self.options.wav)), "rb") as source:
            if (source.getnchannels(), source.getsampwidth(), source.getframerate(),
                    source.getcomptype()) != (1, 2, SAMPLE_RATE, "NONE"):
                raise ValueError("WAV must be uncompressed mono 16 kHz signed PCM16")
            started = time.monotonic()
            while not self.stop.is_set() and self._remaining():
                count = self._remaining()
                pcm = source.readframes(count)
                if not pcm:
                    if self.position < source.getnframes():
                        raise ValueError("Truncated WAV data")
                    break
                if len(pcm) % 2:
                    raise ValueError("Truncated PCM16 sample")
                end = self.position + len(pcm) // 2
                if self.options.realtime:
                    delay = started + end / SAMPLE_RATE - time.monotonic()
                    if delay > 0 and self.stop.wait(delay):
                        break
                if not self.queue.put(AudioBlock(self.position, pcm)):
                    break
                self.position = end

    def _microphone(self):
        """Capture fixed-size blocks with optional digital level gating."""
        audio = sounddevice_module()
        threshold = getattr(self.options, "vad_dbfs", None)
        meter = getattr(self.options, "level_meter", False)
        gate = LevelGate(threshold if threshold is not None else -90) if (
            threshold is not None or meter) else None
        last_meter = 0.0
        try:
            with audio.RawInputStream(
                    samplerate=SAMPLE_RATE, blocksize=BLOCK_SAMPLES,
                    channels=1, dtype="int16", device=self.options.device) as source:
                while not self.stop.is_set() and self._remaining():
                    pcm, overflow = source.read(self._remaining())
                    if overflow and gate is not None:
                        gate.reset()
                    original = bytes(pcm)
                    filtered = gate.process(original) if gate is not None else original
                    if meter and time.monotonic() - last_meter >= 1:
                        print(f"MIC RMS={gate.level:.1f} dBFS "
                              f"gate={threshold} active={gate.active}", flush=True)
                        last_meter = time.monotonic()
                    outgoing = filtered if threshold is not None else original
                    block = AudioBlock(self.position, outgoing, DISCONTINUITY if overflow else 0)
                    if not self.queue.put(block):
                        break
                    self.position += block.samples
        except audio.PortAudioError as error:
            raise RuntimeError(f"Microphone capture failed: {error}") from error


def list_devices(device: Optional[int]) -> str:
    """Enumerate optional audio hardware without opening a capture stream.

    :param device:  Optional sounddevice device index, or all devices when absent.
    :returns:       Printable device information.
    """
    audio = sounddevice_module()
    try:
        return str(audio.query_devices(device))
    except audio.PortAudioError as error:
        raise RuntimeError(f"Audio device enumeration failed: {error}") from error
