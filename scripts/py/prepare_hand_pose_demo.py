# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Publish an RGB hand pose frame using the existing atomic image transport."""
import argparse
from pathlib import Path
import secrets
from typing import Optional
import zlib

from PIL import Image, ImageOps

from prepare_qrcode_demo import HEADER, prepare_pixels, publish_request


def prepare_hand_pixels(source: Image.Image, size: int, framing: str = "square") -> Image.Image:
    """Use a square field of view without stretching the hand.

    :param source:  Camera or file image; never modified in place.
    :param size:    Model width and height.
    :param framing: Center square crop, or full-frame letterbox.
    :returns:       RGB model input with the selected field of view.
    """
    if framing not in ("square", "letterbox"):
        raise ValueError("Expected square or letterbox framing")
    oriented = ImageOps.exif_transpose(source)
    if framing == "square":
        side = min(oriented.size)
        left = (oriented.width - side) // 2
        top = (oriented.height - side) // 2
        oriented = oriented.crop((left, top, left + side, top + side))
    return prepare_pixels(oriented, size, size, "RGB")


def encode_request(image: Image.Image, request_id: int, magic: bytes = b"HPD1") -> bytes:
    """Encode packed RGB pixels for a semihosted image request.

    :param image:       Prepared RGB model image.
    :param request_id:  Unique unsigned 32-bit capture identifier.
    :param magic:       Four-byte firmware protocol identifier.
    :returns:           Header, pixel count and CRC-protected RGB payload.
    """
    if image.mode != "RGB" or not 0 <= request_id <= 0xFFFFFFFF or len(magic) != 4:
        raise ValueError("Expected RGB image, uint32 request ID and four-byte magic")
    pixels = image.tobytes()
    return HEADER.pack(magic, request_id, image.width, image.height, len(pixels),
                       zlib.crc32(pixels)) + pixels


def main(argv: Optional[list[str]] = None) -> int:
    """Publish an image file for the hand pose firmware.

    :param argv:    Optional CLI argument list.
    :returns:       Zero on success, one on image/preprocessing failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--framing", choices=("square", "letterbox"), default="square")
    parser.add_argument("--preview", type=Path)
    options = parser.parse_args(argv)
    try:
        paths = [options.input.resolve(), options.output.resolve()]
        if options.preview is not None:
            paths.append(options.preview.resolve())
        if len(set(paths)) != len(paths):
            raise ValueError("Input, request and preview paths must differ")
        with Image.open(options.input) as source:
            image = prepare_hand_pixels(source, options.size, options.framing)
        if options.preview is not None:
            options.preview.parent.mkdir(parents=True, exist_ok=True)
            image.save(options.preview, format="PNG")
        request = secrets.randbits(32)
        publish_request(options.output, encode_request(image, request))
        print(f"Hand image published: id={request} size={image.width}x{image.height}", flush=True)
        return 0
    except (OSError, ValueError) as error:
        print(f"Hand image preparation failed: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
