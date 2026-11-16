# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Host-source and real localhost transport regressions; no microphone is opened."""
import contextlib
import importlib
import io
from multiprocessing.connection import Client, Listener
from pathlib import Path
import struct
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch
import wave

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "py"
sys.path.insert(0, str(SCRIPT_DIR))

fvp_audio_demo = importlib.import_module("fvp_audio_demo")
AudioSource = importlib.import_module("fvp_audio_demo_sources").AudioSource
BlockQueue = importlib.import_module("fvp_audio_demo_sources").BlockQueue
arm_vsi0 = importlib.import_module("fvp_audio_demo_vsi.arm_vsi0")
protocol = importlib.import_module("fvp_audio_demo_vsi.protocol")


def write_wav(path: Path, count: int, rate: int = 16000) -> bytes:
    """Create a deterministic project-local WAV fixture.

    :param path:    Temporary path inside the repository.
    :param count:   Number of PCM16 samples to generate.
    :param rate:    WAV sampling frequency.
    :returns:       Exact PCM bytes expected from the source.
    """
    pcm = b"".join(struct.pack("<h", index % 32768) for index in range(count))
    with wave.Wave_write(str(path)) as output:
        output.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        output.writeframes(pcm)
    return pcm


class QueueTests(unittest.TestCase):
    """Exercise overflow, WAV backpressure and shutdown without real-time capture."""

    def test_microphone_drop_preserves_absolute_time(self):
        """Evict oldest blocks, marking the next delivered interval as discontinuous."""
        queue = BlockQueue(40, True)
        for first in (0, 320, 640):
            queue.put(protocol.AudioBlock(first, b"\1\0" * 320))
        block = queue.get()
        self.assertEqual((block.first_sample, block.backlog_samples), (320, 320))
        self.assertEqual(block.flags, protocol.DISCONTINUITY)
        self.assertEqual(queue.get().flags, 0)

    def test_wave_backpressure_and_close(self):
        """A full file queue blocks until consumption; close wakes its producer."""
        queue = BlockQueue(20, False)
        queue.put(protocol.AudioBlock(0, b"\0\0" * 320))
        entered = threading.Event()
        finished = threading.Event()

        def produce():
            """Attempt one write beyond queue capacity."""
            entered.set()
            queue.put(protocol.AudioBlock(320, b"\0\0" * 320))
            finished.set()

        thread = threading.Thread(target=produce, daemon=True)
        thread.start()
        self.assertTrue(entered.wait(1))
        self.assertFalse(finished.wait(0.02))
        self.assertEqual(queue.get().first_sample, 0)
        self.assertTrue(finished.wait(1))
        self.assertEqual(queue.get().first_sample, 320)
        thread.join(1)
        queue.put(protocol.AudioBlock(640, b"\0\0" * 320))
        finished.clear()
        thread = threading.Thread(target=produce, daemon=True)
        thread.start()
        queue.close()
        thread.join(1)
        self.assertFalse(thread.is_alive())

    def test_partial_backlog_and_terminal(self):
        """Count valid samples rather than padding and drain queued data before EOS."""
        queue = BlockQueue(40, True)
        queue.put(protocol.AudioBlock(0, b"\0\0" * 320))
        queue.put(protocol.AudioBlock(320, b"\0\0" * 7))
        queue.finish(327)
        self.assertEqual(queue.get().backlog_samples, 7)
        self.assertEqual(queue.get().samples, 7)
        self.assertEqual(queue.get(), protocol.AudioBlock(327, flags=protocol.EOS))
        self.assertEqual(queue.get(), protocol.AudioBlock(327, flags=protocol.EOS))

    def test_evicted_overflow_marker_reaches_first_survivor(self):
        """Preserve hardware overflow flags when their original block is evicted."""
        queue = BlockQueue(40, True)
        queue.put(protocol.AudioBlock(0, b"\0\0" * 320, protocol.DISCONTINUITY))
        queue.put(protocol.AudioBlock(320, b"\0\0" * 320))
        queue.put(protocol.AudioBlock(640, b"\0\0" * 320))
        self.assertEqual(queue.get().flags, protocol.DISCONTINUITY)
        self.assertEqual(queue.get().flags, 0)


class SourceTests(unittest.TestCase):
    """Validate bounded file reads, lazy startup, pacing and explicit source errors."""

    def test_microphone_preserves_quiet_samples_by_default(self):
        """Forward even very quiet speech unchanged, including with the level meter."""
        pcm = struct.pack("<320h", *range(-160, 160))
        for extra in ([], ["--level-meter"]):
            options = fvp_audio_demo.parse_args(
                ["--microphone", "--duration", "0.02", *extra])
            source = AudioSource(options)
            audio = MagicMock()
            stream = audio.RawInputStream.return_value.__enter__.return_value
            stream.read.return_value = (pcm, False)
            try:
                with patch("fvp_audio_demo_sources.sounddevice_module", return_value=audio):
                    actual = source.pull().pcm
                self.assertEqual(actual, pcm)
            finally:
                source.close()

    def test_lazy_lossless_wav_and_duration(self):
        """Retain every WAV sample across a one-block queue, including the short tail."""
        with tempfile.TemporaryDirectory(dir=SCRIPT_DIR) as directory:
            path = Path(directory) / "source.wav"
            expected = write_wav(path, 997)
            options = fvp_audio_demo.parse_args(["--wav", str(path), "--queue-ms", "20"])
            source = AudioSource(options)
            try:
                self.assertIsNone(source.thread)
                blocks = []
                while True:
                    block = source.pull()
                    if block.flags & protocol.EOS:
                        break
                    blocks.append(block)
                self.assertEqual(b"".join(item.pcm for item in blocks), expected)
                self.assertEqual([item.first_sample for item in blocks], [0, 320, 640, 960])
                self.assertEqual(block.first_sample, 997)
                self.assertTrue(all(item.flags == 0 for item in blocks))
            finally:
                source.close()
            options.duration = 0.021
            source = AudioSource(options)
            try:
                self.assertEqual(source.pull().samples, 320)
                self.assertEqual(source.pull().samples, 16)
                self.assertEqual(source.pull().first_sample, 336)
            finally:
                source.close()

    def test_invalid_wav_is_error(self):
        """Reject unsupported input instead of forwarding incorrectly timed PCM."""
        with tempfile.TemporaryDirectory(dir=SCRIPT_DIR) as directory:
            path = Path(directory) / "wrong.wav"
            write_wav(path, 100, rate=8000)
            source = AudioSource(fvp_audio_demo.parse_args(["--wav", str(path)]))
            try:
                block = source.pull()
                self.assertEqual(block.flags, protocol.EOS | protocol.ERROR)
                self.assertIn("16 kHz", source.error)
            finally:
                source.close()

    def test_realtime_paces_wav(self):
        """Pacing uses source samples and an interruptible host-clock wait."""
        with tempfile.TemporaryDirectory(dir=SCRIPT_DIR) as directory:
            path = Path(directory) / "paced.wav"
            write_wav(path, 320)
            source = AudioSource(fvp_audio_demo.parse_args(["--wav", str(path), "--realtime"]))
            try:
                with patch("fvp_audio_demo_sources.time.monotonic", return_value=1), \
                        patch.object(source.stop, "wait", return_value=False) as wait:
                    self.assertEqual(source.pull().samples, 320)
                self.assertAlmostEqual(wait.call_args.args[0], 0.02)
            finally:
                source.close()

    def test_mocked_microphone_duration_and_overflow(self):
        """Use a fake stream to check duration and overflow without touching hardware."""
        audio = MagicMock()
        audio.PortAudioError = RuntimeError
        stream = audio.RawInputStream.return_value.__enter__.return_value
        stream.read.side_effect = [(bytes(640), True), (bytes(32), False)]
        options = fvp_audio_demo.parse_args(["--microphone", "--duration", "0.021"])
        source = AudioSource(options)
        try:
            with patch("fvp_audio_demo_sources.sounddevice_module", return_value=audio):
                first = source.pull()
                second = source.pull()
                terminal = source.pull()
            self.assertEqual((first.samples, first.flags), (320, protocol.DISCONTINUITY))
            self.assertEqual((second.first_sample, second.samples), (320, 16))
            self.assertEqual((terminal.first_sample, terminal.flags), (336, protocol.EOS))
            self.assertEqual([call.args[0] for call in stream.read.call_args_list], [320, 16])
        finally:
            source.close()

    def test_mocked_microphone_failure(self):
        """Report backend errors through the terminal protocol without capture."""
        audio = MagicMock()
        audio.PortAudioError = RuntimeError
        audio.RawInputStream.side_effect = RuntimeError("device unavailable")
        source = AudioSource(fvp_audio_demo.parse_args(["--microphone"]))
        try:
            with patch("fvp_audio_demo_sources.sounddevice_module", return_value=audio):
                block = source.pull()
            self.assertEqual(block.flags, protocol.ERROR | protocol.EOS)
            self.assertIn("device unavailable", source.error)
        finally:
            source.close()

    def test_cli_rejects_invalid_limits(self):
        """Reject malformed bounds before opening sockets or source devices."""
        for arguments in (["--queue-ms", "19"], ["--port", "0"],
                          ["--duration", "nan"], ["--duration", "0"], ["--realtime"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                fvp_audio_demo.parse_args(["--microphone", *arguments])


class TransportTests(unittest.TestCase):
    """Check the actual authenticated socket path from host source to VSI DMA."""

    def test_localhost_bridge(self):
        """Transfer exact WAV bytes, metadata, EOS and DMA completion end to end."""
        with tempfile.TemporaryDirectory(dir=SCRIPT_DIR) as directory:
            path = Path(directory) / "transport.wav"
            expected = write_wav(path, 327)
            source = AudioSource(fvp_audio_demo.parse_args(["--wav", str(path)]))
            results = []
            with Listener(("127.0.0.1", 0), authkey=protocol.AUTHKEY) as listener:

                def serve():
                    """Accept the test bridge and run the production request loop."""
                    with listener.accept() as connection:
                        results.append(fvp_audio_demo.serve_connection(connection, source))

                thread = threading.Thread(target=serve, daemon=True)
                thread.start()
                try:
                    with patch.dict("os.environ", {
                            "FVP_AUDIO_DEMO_PORT": str(listener.address[1])}):
                        arm_vsi0.init()
                        self.assertIsNone(arm_vsi0.BRIDGE.connection)
                        self.assertEqual(arm_vsi0.rdRegs(1), protocol.MAGIC)
                        captured = bytearray()
                        for first, count in ((0, 320), (320, 7)):
                            arm_vsi0.wrRegs(0, 1)
                            self.assertEqual(arm_vsi0.rdRegs(2), first)
                            self.assertEqual(arm_vsi0.rdRegs(4), count)
                            arm_vsi0.wrRegs(7, 0)
                            data = arm_vsi0.rdDataDMA(640)
                            self.assertEqual(arm_vsi0.rdRegs(7), 1)
                            captured.extend(data[:count * 2])
                            self.assertEqual(data[count * 2:], bytes(640 - count * 2))
                            self.assertEqual(arm_vsi0.rdDataDMA(640), bytes(640))
                        self.assertEqual(captured, expected)
                        arm_vsi0.wrRegs(0, 1)
                        self.assertEqual(arm_vsi0.rdRegs(5), protocol.EOS)
                        self.assertEqual(arm_vsi0.rdRegs(2), 327)
                        arm_vsi0.wrRegs(0, 1)
                        self.assertEqual(arm_vsi0.rdRegs(5), protocol.EOS)
                finally:
                    arm_vsi0.init()
                    source.close()
                thread.join(2)
                self.assertFalse(thread.is_alive())
                self.assertEqual(results, [True])

    def test_bad_request_does_not_acquire(self):
        """Protocol errors fail explicitly before the lazy producer starts."""
        source = AudioSource(fvp_audio_demo.parse_args(["--wav", "unused.wav"]))
        with Listener(("127.0.0.1", 0), authkey=protocol.AUTHKEY) as listener:

            def serve():
                """Accept one malformed client."""
                with listener.accept() as connection, contextlib.redirect_stderr(io.StringIO()):
                    fvp_audio_demo.serve_connection(connection, source)

            thread = threading.Thread(target=serve, daemon=True)
            thread.start()
            with Client(listener.address, authkey=protocol.AUTHKEY) as client:
                client.send_bytes(b"BAD")
                block = protocol.AudioBlock.decode(client.recv_bytes())
                self.assertEqual(block.flags, protocol.EOS | protocol.ERROR)
            thread.join(2)
            self.assertIsNone(source.thread)


if __name__ == "__main__":
    unittest.main()
