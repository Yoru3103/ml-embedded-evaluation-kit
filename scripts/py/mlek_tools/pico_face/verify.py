# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Compare a generated RGB888 frame against a PNG or raw RGB reference."""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: Path) -> np.ndarray:
    """Read a 128x128 RGB image as floating-point HWC values."""
    if path.suffix.lower() in (".bin", ".rgb"):
        data = path.read_bytes()
        if len(data) != 49152:
            raise ValueError(f"Expected 49152 RGB bytes: {path}")
        return np.frombuffer(data, dtype=np.uint8).reshape(128, 128, 3).astype(np.float64)
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64)


def main():
    """Fail if image shapes differ or PSNR falls below the requested threshold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--min-psnr", type=float, default=35.0)
    args = parser.parse_args()
    actual, reference = load_image(args.actual), load_image(args.reference)
    if actual.shape != reference.shape:
        raise SystemExit("Image shapes differ")
    difference = np.abs(actual - reference)
    mse = float(np.mean(difference ** 2))
    psnr = math.inf if mse == 0 else 10 * math.log10(255 ** 2 / mse)
    print(f"PSNR={psnr:.2f} dB; max difference={difference.max():.0f}; "
          f"mean difference={difference.mean():.3f}")
    if psnr < args.min_psnr:
        raise SystemExit(f"PSNR is below {args.min_psnr} dB")


if __name__ == "__main__":
    main()
