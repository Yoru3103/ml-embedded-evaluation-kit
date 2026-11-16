#  SPDX-FileCopyrightText:  Copyright 2021, 2023, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""
This script can be used with the noise reduction use case to save
the dumped noise reduced audio to a wav file.

Example use:
python rnnoise_dump_extractor.py --dump_file output.bin --output_dir ./denoised_wavs/
"""

import argparse
from pathlib import Path
import struct
from typing import BinaryIO
import wave


def read_exact(dump_file: BinaryIO, size: int) -> bytes:
    """
    Read a complete field from the dump.

    :param dump_file:   Open binary dump.
    :param size:        Number of bytes to read.
    :returns:           The requested bytes.
    :raises ValueError: If the dump ends before the field is complete.
    """
    data = dump_file.read(size)
    if len(data) != size:
        raise ValueError("Truncated RNNoise dump")
    return data


def extract(dump_file: BinaryIO, output_dir: str, export_npy: bool):
    """
    Extract audio files from an RNNoise memory dump.

    :param dump_file:   Open binary dump.
    :param output_dir:  Destination directory.
    :param export_npy:  Also export each clip as a NumPy array.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    while True:
        filename_length, = struct.unpack("<i", read_exact(dump_file, 4))
        if filename_length == -1:
            return
        if not 1 <= filename_length <= len("loop4294967295"):
            raise ValueError("Invalid RNNoise clip name length")

        filename = read_exact(dump_file, filename_length).decode("ascii")
        if not filename.startswith("loop") or not filename[4:].isdigit():
            raise ValueError(f"Invalid RNNoise clip name: {filename!r}")

        audio_clip_length, = struct.unpack("<I", read_exact(dump_file, 4))
        if not audio_clip_length or audio_clip_length % 2:
            raise ValueError("Invalid RNNoise PCM byte count")
        audio_clip = read_exact(dump_file, audio_clip_length)
        output_base = destination / f"denoised_{filename}"
        output_wav = output_base.with_suffix(".wav")
        with wave.open(str(output_wav), "wb") as wav_file:
            wav_file.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
            wav_file.writeframes(audio_clip)
        print(f"{output_wav} written to disk")

        if export_npy:
            # NumPy is required only for the optional array export.
            import numpy as np

            output_npy = output_base.with_suffix(".npy")
            np.save(output_npy, np.frombuffer(audio_clip, dtype="<i2"))
            print(f"{output_npy} written to disk")


def main():
    """Parse CLI arguments and extract the requested audio files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_file", type=Path, required=True, help="RNNoise memory dump.")
    parser.add_argument("--output_dir", required=True, help="Directory for extracted WAV files.")
    parser.add_argument("--export_npy", action="store_true", help="Also save NumPy arrays.")
    args = parser.parse_args()

    try:
        with args.dump_file.open("rb") as dump_file:
            extract(dump_file, args.output_dir, args.export_npy)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
