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
from PIL import Image, ImageOps

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


def load_image_array(path: Path, image_size: int, preprocessing: str,
                     channels: int = 3) -> np.ndarray:
    """Load RGB or grayscale /255 NCHW data after the deployment resize policy.

    :param path:            Source image.
    :param image_size:      Square input resolution.
    :param preprocessing:   Resize and crop policy.
    :param channels:        Model input channels, one or three.
    :returns:               Contiguous normalized float32 NCHW array.
    :raises ValueError:     Unsupported image policy or channel count.
    """
    with Image.open(path) as source:
        image = source.convert("RGB")
    if preprocessing == "mlek-crop":
        ratio = image_size / min(image.size)
        image = image.resize((int(image.width * ratio), int(image.height * ratio)),
                             Image.Resampling.BILINEAR)
        left = (image.width - image_size) / 2
        top = (image.height - image_size) / 2
        image = image.crop((left, top, left + image_size, top + image_size))
    elif preprocessing == "letterbox":
        ratio = image_size / max(image.size)
        image = image.resize((round(image.width * ratio), round(image.height * ratio)),
                             Image.Resampling.BILINEAR)
        width, height = image_size - image.width, image_size - image.height
        image = ImageOps.expand(image, (width // 2, height // 2,
                                      width - width // 2, height - height // 2),
                                fill=(114, 114, 114))
    else:
        raise ValueError(f"Unknown preprocessing: {preprocessing}")
    if channels == 1:
        image = image.convert("L")
    elif channels != 3:
        raise ValueError("Model input must have one or three channels")
    values = np.asarray(image, dtype=np.float32)
    values = values[None, None] if channels == 1 else values.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(values / 255.0)


def load_image(path: Path, image_size: int, preprocessing: str,
               channels: int = 3) -> torch.Tensor:
    """Wrap shared image preparation as a PyTorch tensor when requested.

    :param path:            Source image.
    :param image_size:      Square input resolution.
    :param preprocessing:  Resize and crop policy.
    :param channels:        Model input channels, one or three.
    :returns:               Contiguous normalized NCHW tensor.
    """
    return import_module("torch").from_numpy(
        load_image_array(path, image_size, preprocessing, channels))
