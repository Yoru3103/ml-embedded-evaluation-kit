# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Camera interaction acceptance using synthetic frames, without a webcam."""
import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import zlib

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/py"))
try:
    SPEC = importlib.util.spec_from_file_location(
        "qrcode_camera_demo", ROOT / "scripts/py/qrcode_camera_demo.py")
    CAMERA = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(CAMERA)
finally:
    sys.path.pop(0)


class CameraTests(unittest.TestCase):
    """Verify explicit capture, request gating, and display/input separation."""

    def test_square_crop_and_preview_use_identical_model_pixels(self):
        """Crop the center field of view and magnify exactly those submitted pixels."""
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            build = Path(directory)
            (build / "CMakeCache.txt").write_text(
                "yolov8_pose_IMAGE_SIZE:STRING=32\n", encoding="utf-8")
            (build / "hand-pose-fvp.log").touch()
            session = CAMERA.CaptureSession(build, "hand-pose")
            try:
                # Outer columns must disappear; a uniform square center must not get padding.
                frame = np.zeros((32, 64, 3), dtype=np.uint8)
                frame[:, :, 0] = 255
                frame[:, 16:48] = (0, 0, 255)
                cv = Mock()
                cv.cvtColor.side_effect = lambda source, _code: (
                    np.asarray(source)[:, :, ::-1].copy())
                cv.resize.side_effect = lambda source, _size, **_kw: (
                    source.repeat(3, 0).repeat(3, 1))
                cv.copyMakeBorder.side_effect = lambda source, *_args, **_kw: np.pad(
                    source, ((40, 0), (0, 0), (0, 0)))
                display, capture = CAMERA.preview_frame(frame, session, cv)
                self.assertEqual(capture.size, (32, 32))
                self.assertEqual(capture.getpixel((0, 0)), (255, 0, 0))
                self.assertEqual(display.shape, (136, 96, 3))
                self.assertTrue(np.array_equal(display[40::3, ::3, ::-1], np.asarray(capture)))
                session.submit(capture)
                packet = (build / "demo-input/hand-pose-input.bin").read_bytes()
                self.assertEqual(packet[24:], capture.tobytes())
            finally:
                session.close()

    def test_hand_capture_preserves_rgb_and_matches_hand_result(self):
        """Hand frames must keep RGB channels, gray padding and their own result marker."""
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            build = Path(directory)
            (build / "CMakeCache.txt").write_text(
                "yolov8_pose_IMAGE_SIZE:STRING=32\n", encoding="utf-8")
            log = build / "hand-pose-fvp.log"
            log.touch()
            session = CAMERA.CaptureSession(build, "hand-pose", "letterbox")
            try:
                with patch.object(CAMERA.secrets, "randbits", return_value=234):
                    self.assertTrue(session.submit(Image.new("RGB", (64, 32), (255, 0, 0))))
                packet = (build / "demo-input/hand-pose-input.bin").read_bytes()
                self.assertEqual(struct.unpack("<4sIIIII", packet[:24]),
                                 (b"HPD1", 234, 32, 32, 3072, zlib.crc32(packet[24:])))
                self.assertEqual(packet[24:27], bytes((114, 114, 114)))
                offset = 24 + (16 * 32 + 16) * 3
                self.assertEqual(packet[offset:offset + 3], bytes((255, 0, 0)))
                with log.open("a", encoding="utf-8") as handle:
                    handle.write("QRCODE_RESULT id=234 count=1\n")
                session.poll()
                self.assertEqual(session.pending, 234)
                with log.open("a", encoding="utf-8") as handle:
                    handle.write("HAND_POSE_RESULT id=234 count=1\n")
                session.poll()
                self.assertIsNone(session.pending)
            finally:
                session.close()

    def test_capture_waits_for_matching_result(self):
        """Do not publish on startup or overwrite a request still being processed."""
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            build = Path(directory)
            (build / "CMakeCache.txt").write_text(
                "qrcode_IMAGE_WIDTH:STRING=32\nqrcode_IMAGE_HEIGHT:STRING=32\n",
                encoding="utf-8")
            log = build / "qrcode-fvp.log"
            log.write_text("QRCODE_RESULT id=99 count=1\n", encoding="utf-8")
            session = CAMERA.CaptureSession(build)
            try:
                request = build / "demo-input/qrcode-input.bin"
                self.assertFalse(request.exists())
                with patch.object(CAMERA.secrets, "randbits", side_effect=(123, 124)):
                    self.assertTrue(session.submit(Image.new("RGB", (64, 32), "white")))
                    first = request.read_bytes()
                    self.assertFalse(session.submit(Image.new("RGB", (64, 32), "black")))
                    self.assertEqual(request.read_bytes(), first)
                    with log.open("a", encoding="utf-8") as handle:
                        handle.write("QRCODE_RESULT id=99 count=1\n")
                    session.poll()
                    self.assertEqual(session.pending, 123)
                    with log.open("a", encoding="utf-8") as handle:
                        handle.write("QRCODE_RESULT id=123 count=")
                    session.poll()
                    self.assertEqual(session.pending, 123)
                    with log.open("a", encoding="utf-8") as handle:
                        handle.write("0\n")
                    session.poll()
                    self.assertIsNone(session.pending)
                    self.assertTrue(session.submit(Image.new("RGB", (64, 32), "black")))
                    self.assertNotEqual(request.read_bytes(), first)
            finally:
                session.close()

    def test_img_class_file_is_published_as_rgb_capture(self):
        """Send one RGB request through the same path used for camera captures."""
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            build = Path(directory)
            (build / "CMakeCache.txt").write_text(
                "img_class_IMAGE_SIZE:STRING=32\n", encoding="utf-8")
            (build / "img-class-fvp.log").touch()
            image = build / "source.png"
            Image.new("RGB", (64, 32), (255, 0, 0)).save(image)
            with patch.object(CAMERA.secrets, "randbits", return_value=234):
                result = CAMERA.main(["--build-directory", str(build), "--demo", "img-class",
                                      "--image", str(image)])
            self.assertEqual(result, 0)
            packet = (build / "demo-input/img-class-input.bin").read_bytes()
            self.assertEqual(struct.unpack("<4sIIIII", packet[:24]),
                             (b"ICD1", 234, 32, 32, 3072, zlib.crc32(packet[24:])))
            self.assertEqual(packet[24:], bytes((255, 0, 0)) * (32 * 32))

    def test_preview_publishes_only_space_without_overlay(self):
        """A live preview must not auto-submit or burn instructions into model pixels."""
        frame = np.zeros((32, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # Blue in OpenCV's BGR convention.
        camera = Mock()
        camera.read.return_value = (True, frame)
        session = Mock(status="SPACE: capture", size=(256, 256))
        cv = Mock()
        cv.waitKey.side_effect = (-1, ord(" "), ord("q"))
        cv.getWindowProperty.return_value = 1
        cv.cvtColor.side_effect = lambda source, _code: source[:, :, ::-1].copy()
        cv.putText.side_effect = lambda display, *_args: display.fill(17)
        CAMERA.preview(camera, session, cv)
        cv.resizeWindow.assert_called_once_with(CAMERA.WINDOW, 512, 512)
        self.assertEqual(session.submit.call_count, 1)
        submitted = session.submit.call_args.args[0]
        self.assertEqual(submitted.getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(camera.read.call_count, 3)

    def test_camera_failure_releases_resources(self):
        """A failed camera open closes the log and releases capture/window resources."""
        cv = Mock()
        cv.error = RuntimeError
        cv.VideoCapture.return_value.isOpened.return_value = False
        with patch.object(CAMERA.importlib, "import_module", return_value=cv), \
                patch.object(CAMERA, "CaptureSession") as session:
            self.assertEqual(CAMERA.main(["--build-directory", str(ROOT)]), 1)
            cv.VideoCapture.return_value.release.assert_called_once()
            cv.destroyAllWindows.assert_called_once()
            session.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
