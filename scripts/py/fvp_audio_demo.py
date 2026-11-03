# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Serve lazy bounded audio to the application-level FVP VSI0 pull adapter.

Examples::

    python scripts/py/fvp_audio_demo.py --wav speech.wav --realtime
    python scripts/py/fvp_audio_demo.py --list-devices
    python scripts/py/fvp_audio_demo.py --microphone --device 1 --queue-ms 1000

Set FVP mps4_board.v_path to scripts/py/fvp_audio_demo_vsi (absolute path).
The optional sounddevice package is needed only for microphone/device listing.
"""
import argparse
import math
from multiprocessing.connection import Connection, Listener
from pathlib import Path
import sys
import threading
from typing import Optional

from fvp_audio_demo_sources import AudioSource, list_devices, sounddevice_module
from record_audio_demo import record_utterance
from fvp_audio_demo_vsi.protocol import AUTHKEY, AudioBlock, EOS, ERROR, REQUEST


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse and validate host options without starting acquisition.

    :param argv:    Optional argument list for programmatic callers.
    :returns:       Validated CLI source and transport options.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--wav", type=Path, help="mono 16 kHz PCM16 WAV (no implicit conversion)")
    source.add_argument("--microphone", action="store_true", help="capture a live microphone")
    source.add_argument("--list-devices", action="store_true", help="enumerate; do not capture")
    parser.add_argument("--device", type=int, help="sounddevice input device index")
    parser.add_argument("--record-utterance", action="store_true",
                        help="record one sentence with Start/Stop buttons before sending")
    parser.add_argument("--auto-utterances", action="store_true",
                        help="Chinese ASR continuous raw microphone with target-side VAD")
    parser.add_argument("--vad-dbfs", type=float, help="microphone start energy gate in dBFS")
    parser.add_argument("--level-meter", action="store_true",
                        help="print microphone RMS once a second")
    parser.add_argument("--port", type=int, default=6010)
    parser.add_argument("--realtime", action="store_true", help="pace WAV reads at source rate")
    parser.add_argument("--queue-ms", type=int, default=1000,
                        help="bounded audio queue; >=20 ms, rounded down to 20 ms blocks")
    parser.add_argument("--duration", type=float, help="maximum acquired audio seconds")
    options = parser.parse_args(argv)
    if options.auto_utterances and (not options.microphone or options.record_utterance
                                    or options.vad_dbfs is not None):
        parser.error("--auto-utterances requires raw microphone without button recording")
    if options.record_utterance and not options.microphone:
        parser.error("--record-utterance requires --microphone")
    if options.vad_dbfs is not None and (
            not math.isfinite(options.vad_dbfs) or not -90 <= options.vad_dbfs <= -1):
        parser.error("--vad-dbfs must be between -90 and -1")
    if (options.vad_dbfs is not None or options.level_meter) and not options.microphone:
        parser.error("--vad-dbfs and --level-meter require --microphone")
    if not (options.wav or options.microphone or options.list_devices):
        parser.error("select --wav, --microphone or --list-devices")
    if not 1 <= options.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if options.queue_ms < 20:
        parser.error("--queue-ms must be at least 20")
    if options.duration is not None and (not math.isfinite(options.duration)
                                         or options.duration * 16000 < 1):
        parser.error("--duration must be finite and at least one sample (1/16000 second)")
    if options.device is not None and not (options.microphone or options.list_devices):
        parser.error("--device requires --microphone or --list-devices")
    if options.realtime and options.wav is None:
        parser.error("--realtime requires --wav")
    return options


def serve_connection(connection: Connection, source: AudioSource) -> bool:
    """Serve bounded binary requests until EOS or disconnect.

    :param connection:  Authenticated localhost connection owned by the caller.
    :param source:      Lazy source; no acquisition happens before NEXT.
    :returns:           True on clean EOS, False on source failure or disconnect.
    """
    try:
        while True:
            request = connection.recv_bytes(len(REQUEST))
            if request != REQUEST:
                connection.send_bytes(AudioBlock(flags=EOS | ERROR).encode())
                print("Invalid audio request; expected NEXT", file=sys.stderr, flush=True)
                return False
            block = source.pull()
            connection.send_bytes(block.encode())
            if block.flags & EOS:
                if source.error:
                    print("Audio source failed: " + source.error, file=sys.stderr, flush=True)
                return not bool(block.flags & ERROR)
    except (OSError, EOFError) as error:
        print("Audio client disconnected: " + str(error), file=sys.stderr, flush=True)
        return False
    finally:
        source.close()


def main(argv: Optional[list[str]] = None) -> int:
    """Run device enumeration or a single-client host server.

    :param argv:    Optional command-line arguments.
    :returns:       Zero on clean EOS, nonzero on source/transport failure.
    """
    options = parse_args(argv)
    try:
        if options.list_devices:
            print(list_devices(options.device))
            return 0
        source = AudioSource(options)
        try:
            with Listener(("127.0.0.1", options.port), authkey=AUTHKEY) as listener:
                print(f"Audio listening on 127.0.0.1:{options.port}; capture starts on NEXT",
                      flush=True)
                with listener.accept() as connection:
                    if options.record_utterance:
                        # Service idle pulls while Tk remains on the main thread.
                        result = []
                        worker = threading.Thread(
                            target=lambda: result.append(serve_connection(connection, source)),
                            name="audio-button-server", daemon=True)
                        worker.start()
                        try:
                            options.recorded_pcm = record_utterance(
                                sounddevice_module(), options.device,
                                min(options.duration or 30, 30), source.stop)
                            while worker.is_alive():
                                worker.join(timeout=0.1)
                            return 0 if result and result[0] else 1
                        finally:
                            source.close()
                    return 0 if serve_connection(connection, source) else 1
        finally:
            source.close()
    except (OSError, RuntimeError) as error:
        print("Audio host failed: " + str(error), file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
