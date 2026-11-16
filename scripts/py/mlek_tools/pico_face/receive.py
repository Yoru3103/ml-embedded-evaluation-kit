# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Request an FPGA RGB frame and save CRC-verified RGB, PNG and metadata."""
import argparse
import dataclasses
from datetime import datetime
import json
import math
from pathlib import Path
import struct
import time
from typing import BinaryIO
import zlib

MAGIC = b"PFIMAGE1"
HEADER = struct.Struct("<8s8IfIQ")
OUTPUT_DIRECTORY = Path(__file__).resolve().parents[4] / "build/logs/pico_face_logs/out"


def resolve_output_prefix(requested: Path | None, seed: int, steps: int) -> Path:
    """Resolve image names inside the dedicated output directory, including legacy paths."""
    if requested is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        requested = Path(f"face_seed{seed}_steps{steps}_{stamp}")
    if not requested.is_absolute():
        parts = requested.parts
        if tuple(part.lower() for part in parts[:4]) == (
                "build", "logs", "pico_face_logs", "out"):
            parts = parts[4:]
        elif parts and parts[0].lower() == "out":
            parts = parts[1:]
        requested = OUTPUT_DIRECTORY.joinpath(*parts)
    resolved = requested.resolve()
    if resolved == OUTPUT_DIRECTORY or OUTPUT_DIRECTORY not in resolved.parents:
        raise ValueError(f"Image output must stay inside {OUTPUT_DIRECTORY}")
    return resolved


@dataclasses.dataclass(frozen=True)
class Frame:
    """Validated image and generation parameters received from the board."""
    seed: int
    steps: int
    class_id: int
    guidance: float
    crc32: int
    generation_us: int
    rgb: bytes


def read_exact(stream: BinaryIO, count: int, deadline: float) -> bytes:
    """Read a bounded byte count, handling serial timeouts and fragmented reads."""
    chunks = bytearray()
    while len(chunks) < count:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out after {len(chunks)}/{count} bytes")
        chunk = stream.read(count - len(chunks))
        if chunk:
            chunks.extend(chunk)
        elif not getattr(stream, "is_open", False):
            raise EOFError(f"Truncated stream after {len(chunks)}/{count} bytes")
    return bytes(chunks)


def find_marker(stream: BinaryIO, marker: bytes, deadline: float):
    """Skip text logs until the requested binary or readiness marker arrives."""
    window = bytearray()
    while True:
        window.extend(read_exact(stream, 1, deadline))
        if window.endswith(marker):
            return
        if len(window) > len(marker):
            del window[:-len(marker)]


def read_frame(stream: BinaryIO, timeout: float = 90.0) -> Frame:
    """Read one version-1 frame, validate bounds and CRC before returning it."""
    deadline = time.monotonic() + timeout
    find_marker(stream, MAGIC, deadline)
    header = MAGIC + read_exact(stream, HEADER.size - len(MAGIC), deadline)
    (_, version, width, height, channels, size, seed, steps, class_id,
     guidance, crc, generation_us) = HEADER.unpack(header)
    if (version, width, height, channels, size) != (1, 128, 128, 3, 49152):
        raise ValueError("Unsupported version, shape or RGB payload size")
    if (steps not in (1, 2, 4, 8) or class_id > 4 or
            not math.isfinite(guidance) or guidance < 0):
        raise ValueError("Invalid generation parameters")
    rgb = read_exact(stream, size, deadline)
    actual_crc = zlib.crc32(rgb)
    if actual_crc != crc:
        raise ValueError(f"CRC mismatch: received {actual_crc:08x}, expected {crc:08x}")
    return Frame(seed, steps, class_id, guidance, crc, generation_us, rgb)


def save_frame(frame: Frame, prefix: Path):
    """Save only a validated frame as RGB888, PNG and JSON metadata."""
    from PIL import Image

    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".rgb").write_bytes(frame.rgb)
    Image.frombytes("RGB", (128, 128), frame.rgb).save(prefix.with_suffix(".png"))
    metadata = dataclasses.asdict(frame)
    del metadata["rgb"]
    metadata.update(width=128, height=128, channels=3, layout="HWC RGB888")
    metadata["crc32"] = f"{frame.crc32:08x}"
    prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {prefix.with_suffix('.png')}; CRC32={frame.crc32:08x}; "
          f"generation={frame.generation_us / 1000:.3f} ms")


def main():
    """Open a serial port, request the stored image and verify it before saving."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="Board application UART, for example COM7")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--wait-ready", action="store_true",
                        help="Start before board reset and wait for SERIAL_IMAGE_READY")
    parser.add_argument("--list", action="store_true", help="List available serial ports")
    parser.add_argument("--capture", type=Path, help="Decode an existing binary UART capture")
    parser.add_argument("--output", type=Path, help="Image name inside build/logs/pico_face_logs/out")
    args = parser.parse_args()
    if args.capture:
        with args.capture.open("rb") as stream:
            frame = read_frame(stream, args.timeout)
            save_frame(frame, resolve_output_prefix(args.output, frame.seed, frame.steps))
        return
    try:
        import serial
        from serial.tools import list_ports
    except ImportError as error:
        raise SystemExit("Install the receiver dependencies: python -m pip install pyserial pillow") from error
    if args.list:
        for port in list_ports.comports():
            print(f"{port.device}: {port.description}")
        return
    if not args.port:
        parser.error("Supply --port COMx, --list or --capture")
    print("Close other terminals using this port. Waiting for the FPGA frame...")
    with serial.Serial(args.port, args.baud, timeout=0.5, write_timeout=5,
                       xonxoff=False, rtscts=False, dsrdtr=False) as stream:
        stream.reset_input_buffer()
        if args.wait_ready:
            find_marker(stream, b"SERIAL_IMAGE_READY", time.monotonic() + args.timeout)
        stream.write(b"I")
        stream.flush()
        frame = read_frame(stream, args.timeout)
        save_frame(frame, resolve_output_prefix(args.output, frame.seed, frame.steps))


if __name__ == "__main__":
    try:
        main()
    except (OSError, EOFError, ValueError) as error:
        raise SystemExit(str(error)) from error
