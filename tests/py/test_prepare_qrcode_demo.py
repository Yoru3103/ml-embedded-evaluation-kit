# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Small host-side acceptance tests for QR image publication."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zlib

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "prepare_qrcode_demo", ROOT / "scripts/py/prepare_qrcode_demo.py")
DEMO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEMO)


class ImageRequestTests(unittest.TestCase):
    """Exercise preprocessing and the host/firmware packet contract."""

    def test_image_formats_and_letterbox(self):
        """Keep aspect ratio and pad rather than stretch a wide image."""
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            for extension in ("png", "jpg", "bmp"):
                source = Path(directory) / f"input.{extension}"
                Image.new("RGB", (64, 32), "white").save(source)
                original = source.read_bytes()
                image = DEMO.prepare_image(source, 32, 32)
                self.assertEqual((image.mode, image.size), ("L", (32, 32)))
                self.assertEqual(image.getpixel((0, 0)), 114)
                self.assertEqual(image.getpixel((16, 16)), 255)
                self.assertEqual(source.read_bytes(), original)

    def test_packet_layout_and_atomic_replacement(self):
        """Publish two distinct requests and verify length, ID and CRC."""
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            destination = Path(directory) / "request.bin"
            for request_id in (123, 124):
                pixels = Image.new("L", (32, 32), request_id)
                packet = DEMO.encode_request(pixels, request_id)
                DEMO.publish_request(destination, packet)
                received = destination.read_bytes()
                self.assertEqual(received, packet)
                self.assertEqual(DEMO.HEADER.unpack(received[:24]),
                                 (b"QRD1", request_id, 32, 32, 1024,
                                  zlib.crc32(received[24:])))
            self.assertEqual(list(Path(directory).iterdir()), [destination])

    def test_invalid_dimensions_and_packet_input(self):
        """Reject invalid model dimensions, RGB payloads and out-of-range IDs."""
        with self.assertRaises(ValueError):
            DEMO.prepare_image(Path("unused"), 31, 32)
        with self.assertRaises(ValueError):
            DEMO.encode_request(Image.new("RGB", (32, 32)), 1)
        with self.assertRaises(ValueError):
            DEMO.encode_request(Image.new("L", (32, 32)), -1)

    def test_source_overwrite_guard(self):
        """CLI errors must preserve the supplied image."""
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source = Path(directory) / "input.png"
            Image.new("L", (32, 32), 0).save(source)
            original = source.read_bytes()
            self.assertEqual(DEMO.main(["--input", str(source), "--output", str(source)]), 1)
            self.assertEqual(DEMO.main([
                "--input", str(source), "--output", str(Path(directory) / "request.bin"),
                "--preview", str(source)]), 1)
            self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
