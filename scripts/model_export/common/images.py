# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Prepare shared calibration and validation images."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING
import random

import numpy as np
from PIL import Image

from mlek_tools.gen.gen_rgb_cpp import resize_letterbox_image

if TYPE_CHECKING:
    import torch

def find_images(directory: Path, limit: int, seed: int) -> list[Path]:
    """Select a reproducible shuffled sample instead of a class-sorted prefix."""
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    paths = sorted(path for path in directory.rglob("*")
                   if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not paths:
        raise ValueError(f"No calibration images found in {directory}")
    random.Random(seed).shuffle(paths)
    return paths[:limit] if limit else paths


def load_image_array(path: Path, image_size: int | tuple[int, int], preprocessing: str,
                     channels: int = 3) -> np.ndarray:
    """Load an image as normalized NCHW data at the requested model input size.

    :param path:            Source image.
    :param image_size:      Target width and height, or a square side length.
    :param preprocessing:   Center crop or the CMake image generator's letterbox.
    :param channels:        Model input channels, one or three.
    :returns:               Contiguous normalized float32 NCHW array.
    :raises ValueError:     Unsupported image policy or channel count.
    """
    if channels not in (1, 3):
        raise ValueError("Model input must have one or three channels")
    target_size = (image_size, image_size) if isinstance(image_size, int) else image_size
    target_width, target_height = target_size
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Model input width and height must be positive")
    with Image.open(path) as source:
        image = source.convert("RGB" if preprocessing == "mlek-crop" or channels == 3
                               else "L")
    if preprocessing == "mlek-crop":
        ratio = max(target_width / image.width, target_height / image.height)
        image = image.resize((int(image.width * ratio), int(image.height * ratio)),
                             Image.Resampling.BILINEAR)
        left = (image.width - target_width) / 2
        top = (image.height - target_height) / 2
        image = image.crop((left, top, left + target_width, top + target_height))
        if channels == 1:
            image = image.convert("L")
        values = np.asarray(image, dtype=np.float32)
    elif preprocessing == "letterbox":
        values = resize_letterbox_image(image, target_size).reshape(
            (target_height, target_width) if channels == 1
            else (target_height, target_width, channels)).astype(np.float32)
    else:
        raise ValueError(f"Unknown preprocessing: {preprocessing}")
    if channels == 1:
        values = values[None, None]
    else:
        values = values.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(values / 255.0)


def load_image(path: Path, image_size: int | tuple[int, int], preprocessing: str,
               channels: int = 3) -> torch.Tensor:
    """Wrap shared image preparation as a PyTorch tensor when requested.

    :param path:            Source image.
    :param image_size:      Target width and height, or a square side length.
    :param preprocessing:  Resize and crop policy.
    :param channels:        Model input channels, one or three.
    :returns:               Contiguous normalized NCHW tensor.
    """
    return import_module("torch").from_numpy(
        load_image_array(path, image_size, preprocessing, channels))
