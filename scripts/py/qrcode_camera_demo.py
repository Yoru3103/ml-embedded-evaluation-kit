# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Preview a PC camera and submit one frame only when Space is pressed."""
import argparse
import importlib
from pathlib import Path
import re
import secrets
from typing import NamedTuple, Optional

from PIL import Image
import numpy as np

from prepare_qrcode_demo import encode_request, prepare_pixels, publish_request
from prepare_hand_pose_demo import encode_request as encode_hand_request
from prepare_hand_pose_demo import prepare_hand_pixels

WINDOW = "QR camera | SPACE: capture | Q / ESC: exit"


class DemoFormat(NamedTuple):
    """Wire and display settings shared by file and camera workflows."""
    use_case: str
    stem: str
    marker: str
    mode: str
    framing: str = "letterbox"
    rgb_magic: bytes = b""


FORMATS = {
    "qrcode": DemoFormat("qrcode", "qrcode", "QRCODE", "L"),
    "hand-pose": DemoFormat("yolov8_pose", "hand-pose", "HAND_POSE", "RGB", "square", b"HPD1"),
    "img-class": DemoFormat("img_class", "img-class", "IMG_CLASS", "RGB", "square", b"ICD1"),
}


class CaptureSession:
    """Keep a single request in flight while preview continues independently."""

    def __init__(self, build: Path, demo: str = "qrcode", framing: Optional[str] = None):
        """Read model dimensions and open the log without loading old results.

        :param build:   Existing semihosted image demo build directory.
        :param demo:    Image demo selection.
        :param framing: Optional field-of-view override for RGB inputs.
        """
        cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
        dimensions = []
        self.format = FORMATS[demo]
        if framing is not None:
            self.format = self.format._replace(framing=framing)
        for axis in ("WIDTH", "HEIGHT"):
            key = "SIZE" if self.format.mode == "RGB" else axis
            match = re.search(rf"^{self.format.use_case}_IMAGE_{key}:[^=]+=(\d+)$",
                              cache, re.MULTILINE)
            if match is None:
                raise ValueError(f"Missing {demo} image {key}")
            dimensions.append(int(match[1]))
        self.size = (dimensions[0], dimensions[1])
        self.build = build
        self.pending = None
        self.status = "SPACE: capture"
        self.tail = ""
        self.log = (build / f"{self.format.stem}-fvp.log").open(encoding="utf-8", errors="replace")
        self.log.seek(0, 2)

    def poll(self):
        """Accept completion only for this session's current request ID."""
        self.tail = (self.tail + self.log.read())[-16384:]
        if self.pending is not None:
            match = re.search(rf"{self.format.marker}_RESULT id={self.pending} count=(\d+)\b",
                              self.tail)
            if match:
                result = "Classes" if self.format.use_case == "img_class" else "Detections"
                self.status = f"{result}: {match[1]} | SPACE: capture again"
                print(f"Camera result: id={self.pending} count={match[1]}", flush=True)
                self.pending = None

    def prepare(self, frame: Image.Image) -> Image.Image:
        """Prepare the exact image used for both square preview and inference.

        :param frame:   RGB camera frame.
        :returns:       Image at model input dimensions.
        """
        if self.format.mode == "RGB":
            return prepare_hand_pixels(frame, self.size[0], self.format.framing)
        return prepare_pixels(frame, *self.size, self.format.mode)

    def submit(self, frame: Image.Image) -> bool:
        """Publish the displayed frame if no previous capture is awaiting inference.

        :param frame:   Current preview frame in RGB order.
        :returns:       Whether a new request was submitted.
        """
        if self.pending is not None:
            return False
        image = self.prepare(frame)
        request = secrets.randbits(32)
        image.save(self.build / f"{self.format.stem}-input-preview.png")
        packet = (encode_hand_request(image, request, self.format.rgb_magic)
                  if self.format.mode == "RGB" else encode_request(image, request))
        publish_request(self.build / f"demo-input/{self.format.stem}-input.bin",
                        packet)
        self.pending = request
        self.status = "Processing capture..."
        print(f"Camera capture: id={request}", flush=True)
        return True

    def close(self):
        """Release the log handle owned by this session."""
        self.log.close()


def preview_frame(frame: np.ndarray, session: CaptureSession, cv: object) -> tuple:
    """Magnify RGB model pixels threefold, with a separate instruction strip.

    :param frame:   Raw BGR frame from the camera.
    :param session: Capture settings and preprocessing.
    :param cv:      OpenCV module.
    :returns:       Display pixels and the clean RGB image to submit.
    """
    capture = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    display = frame.copy()
    if session.format.mode == "RGB":
        capture = session.prepare(capture)
        display = cv.cvtColor(np.asarray(capture), cv.COLOR_RGB2BGR)
        display = cv.resize(display, (capture.width * 3, capture.height * 3),
                            interpolation=cv.INTER_NEAREST)
        display = cv.copyMakeBorder(display, 40, 0, 0, 0, cv.BORDER_CONSTANT, value=(0, 0, 0))
    return display, capture


def preview(camera: object, session: CaptureSession, cv: object, window: str = WINDOW):
    """Run a responsive preview; the PowerShell supervisor owns FVP shutdown.

    :param camera:  OpenCV capture object.
    :param session: One-in-flight request state.
    :param cv:      OpenCV module, injectable for software acceptance tests.
    :param window:  Preview window title.
    """
    cv.namedWindow(window, cv.WINDOW_NORMAL)
    cv.resizeWindow(window, session.size[0] * 2, session.size[1] * 2)
    while True:
        success, frame = camera.read()
        if not success:
            raise OSError("Camera frame read failed; check device access or camera index")
        session.poll()
        # Overlay only the displayed copy so model input never contains instructions.
        display, capture = preview_frame(frame, session, cv)
        status = session.status
        if session.format.mode == "RGB":
            status = f"LIVE | {status}"
        cv.putText(display, status, (10, 25), cv.FONT_HERSHEY_SIMPLEX,
                   0.6, (0, 255, 0), 2)
        cv.imshow(window, display)
        key = cv.waitKey(20) & 0xFF
        if key in (ord("q"), ord("Q"), 27) or cv.getWindowProperty(window, cv.WND_PROP_VISIBLE) < 1:
            break
        if key == ord(" "):
            session.submit(capture)


def main(argv: Optional[list[str]] = None) -> int:
    """Open the selected camera and guarantee resource release on exit.

    :param argv:    Optional CLI argument list.
    :returns:       Zero on normal exit, one on camera or publication failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf"), default="dshow")
    parser.add_argument("--demo", choices=tuple(FORMATS), default="qrcode")
    parser.add_argument("--framing", choices=("square", "letterbox"))
    parser.add_argument("--image", type=Path, help="Submit a single image file")
    options = parser.parse_args(argv)
    if options.image:
        session = None
        try:
            session = CaptureSession(options.build_directory, options.demo, options.framing)
            with Image.open(options.image) as source:
                session.submit(source)
            return 0
        except (OSError, ValueError) as error:
            print(f"Image preparation failed: {error}", flush=True)
            return 1
        finally:
            if session is not None:
                session.close()
    # Keep image-file mode usable without installing the optional camera dependency.
    cv = importlib.import_module("cv2")
    camera = None
    session = None
    try:
        session = CaptureSession(options.build_directory, options.demo, options.framing)
        backend = {"auto": cv.CAP_ANY, "dshow": cv.CAP_DSHOW, "msmf": cv.CAP_MSMF}
        camera = cv.VideoCapture(options.camera_index, backend[options.backend])
        if not camera.isOpened():
            raise OSError("Cannot open camera; try another index or backend")
        camera.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        print("Camera ready. Focus the preview window; SPACE captures, Q / ESC exits.", flush=True)
        title = f"{options.demo} camera | SPACE: capture | Q / ESC: exit"
        preview(camera, session, cv, title)
        return 0
    except (OSError, ValueError, cv.error) as error:
        print(f"Camera demo failed: {error}", flush=True)
        return 1
    finally:
        if camera is not None:
            camera.release()
        if session is not None:
            session.close()
        cv.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
