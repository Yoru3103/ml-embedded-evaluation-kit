# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Publish an atomic grayscale image request for the semihosted QR detector."""
import argparse
from pathlib import Path
import secrets
import struct
import tempfile
import time
from typing import Optional
import zlib

from PIL import Image, ImageOps

HEADER = struct.Struct("<4sIIIII")


def prepare_image(path: Path, width: int, height: int) -> Image.Image:
    """Apply EXIF orientation, grayscale conversion and centered letterboxing.

    :param path:    Input JPG, PNG, BMP or other Pillow-readable image.
    :param width:   Model image width, a positive multiple of 32.
    :param height:  Model image height, a positive multiple of 32.
    :returns:       Eight-bit grayscale model input with gray-114 padding.
    """
    if not 32 <= width <= 1024 or not 32 <= height <= 1024 or width % 32 or height % 32:
        raise ValueError("Image dimensions must be multiples of 32 in 32..1024")
    with Image.open(path) as source:
        return prepare_pixels(source, width, height)


def prepare_pixels(source: Image.Image, width: int, height: int, mode: str = "L") -> Image.Image:
    """Prepare an in-memory image using the same contract as file input.

    :param source:  Source image; the caller retains ownership.
    :param width:   Model width, a multiple of 32 in 32..1024.
    :param height:  Model height, a multiple of 32 in 32..1024.
    :param mode:    Pixel format, L for QR or RGB for hand pose.
    :returns:       Prepared image with centered gray-114 padding.
    """
    if not 32 <= width <= 1024 or not 32 <= height <= 1024 or width % 32 or height % 32:
        raise ValueError("Image dimensions must be multiples of 32 in 32..1024")
    if mode not in ("L", "RGB"):
        raise ValueError("Expected L or RGB pixel format")
    gray = ImageOps.exif_transpose(source).convert(mode)
    ratio = min(width / gray.width, height / gray.height)
    resized_width = min(width, max(1, round(gray.width * ratio)))
    resized_height = min(height, max(1, round(gray.height * ratio)))
    resized = gray.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    output = Image.new(mode, (width, height), 114 if mode == "L" else (114, 114, 114))
    output.paste(resized, ((width - resized_width) // 2, (height - resized_height) // 2))
    return output


def encode_request(image: Image.Image, request_id: int) -> bytes:
    """Encode an explicit little-endian header followed by grayscale pixels.

    :param image:       Prepared grayscale model input.
    :param request_id:  Unsigned 32-bit identifier, changed for each new image.
    :returns:           Header and payload bytes, including the payload CRC32.
    """
    if image.mode != "L" or not 0 <= request_id <= 0xFFFFFFFF:
        raise ValueError("Expected grayscale image and uint32 request ID")
    pixels = image.tobytes()
    return HEADER.pack(b"QRD1", request_id, image.width, image.height, len(pixels),
                       zlib.crc32(pixels)) + pixels


def publish_request(destination: Path, packet: bytes):
    """Replace a request atomically, retrying brief Windows reader-sharing conflicts.

    :param destination: Target file in the FVP working directory.
    :param packet:      Complete encoded request; never expose a partial payload.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(packet)
        for attempt in range(21):
            try:
                temporary.replace(destination)
                return
            except PermissionError:
                if attempt == 20:
                    raise
                time.sleep(0.05)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Optional[list[str]] = None) -> int:
    """Prepare and publish one file without starting or rebuilding FVP.

    :param argv:    Optional CLI argument list.
    :returns:       Zero on successful publication, one on preparation failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--preview", type=Path)
    options = parser.parse_args(argv)
    try:
        if options.output.resolve() == options.input.resolve():
            raise ValueError("Output must not overwrite the input image")
        if options.preview is not None and options.preview.resolve() == options.output.resolve():
            raise ValueError("Preview and request paths must differ")
        image = prepare_image(options.input, options.width, options.height)
        request_id = secrets.randbits(32)
        if options.preview is not None:
            if options.preview.resolve() == options.input.resolve():
                raise ValueError("Preview must not overwrite the input image")
            options.preview.parent.mkdir(parents=True, exist_ok=True)
            image.save(options.preview, format="PNG")
        publish_request(options.output, encode_request(image, request_id))
        print(f"QR image published: id={request_id} size={image.width}x{image.height} "
              f"file={options.input.name}", flush=True)
        return 0
    except (OSError, ValueError) as error:
        print(f"QR image preparation failed: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
