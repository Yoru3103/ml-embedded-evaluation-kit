# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Render Zint symbols into reproducible grayscale camera-like scenes."""

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

PROFILES = {
    "matched": {"fraction": (0.18, 0.44), "blur": (0.30, 0.75),
                "gain": (0.40, 0.70), "offset": (18, 38), "noise": (1.2, 3.0),
                "shadow": (0.12, 0.40)},
    "clear": {"fraction": (0.26, 0.56), "blur": (0.05, 0.30),
              "gain": (0.65, 0.85), "offset": (12, 25), "noise": (0.5, 1.5),
              "shadow": (0.02, 0.18)},
    "difficult": {"fraction": (0.16, 0.36), "blur": (0.65, 1.00),
                  "gain": (0.35, 0.55), "offset": (28, 45), "noise": (2.0, 3.5),
                  "shadow": (0.20, 0.40)},
}


def scene_parameters(rng: np.random.Generator, size: int,
                       category: str, profile: str) -> dict:
    """Choose source-like geometry and photometric ranges for one scene.

    :param rng:         Per-sample random generator.
    :param size:        Square output resolution.
    :param category:    Checkpoint class name.
    :param profile:     Matched, clear or difficult scene group.
    :returns:           Reproducible scene configuration.
    """
    settings = {key: float(rng.uniform(*bounds)) for key, bounds in PROFILES[profile].items()}
    settings.update(profile=profile, frame_height=int(round(size * rng.choice(
        [0.625, 0.75, 0.875, 1.0], p=[0.15, 0.45, 0.25, 0.15]))))
    # UPNQR has a fixed dense matrix; retain approximately one pixel per module.
    minimum = {"UPN": 0.38, "DOT": 0.24, "MAXI": 0.23,
               "BARCODE": 0.30, "PDF": 0.28, "MPDF": 0.28}.get(category, 0.0)
    settings["fraction"] = max(minimum, settings["fraction"])
    # Thin bars and dots lose their structure sooner than larger matrix cells.
    blur_limit = {"BARCODE": 0.35, "DOT": 0.30, "UPN": 0.45}.get(category)
    if blur_limit is not None:
        settings["blur"] = min(settings["blur"], blur_limit + 0.15 * (profile == "difficult"))
    return settings


def homography(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    """Find a projective transform between two ordered quadrilaterals.

    :param source:      Four source XY corners.
    :param destination: Four destination XY corners.
    :returns:           A three-by-three homogeneous transformation matrix.
    """
    coefficients, values = [], []
    for (src_x, src_y), (dst_x, dst_y) in zip(source, destination):
        coefficients.extend(([src_x, src_y, 1, 0, 0, 0, -dst_x * src_x, -dst_x * src_y],
                             [0, 0, 0, src_x, src_y, 1, -dst_y * src_x, -dst_y * src_y]))
        values.extend((dst_x, dst_y))
    return np.append(np.linalg.solve(coefficients, values), 1).reshape(3, 3)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Project XY points without changing their cyclic order.

    :param points:  Array of XY coordinates.
    :param matrix:  Source-to-destination homography.
    :returns:       Projected XY coordinates.
    """
    projected = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    return projected[:, :2] / projected[:, 2:]


def background(rng: np.random.Generator, size: int) -> Image.Image:
    """Generate a neutral surface with shading and occasional background edges.

    :param rng:    Per-sample random generator.
    :param size:   Square scene resolution.
    :returns:      Eight-bit grayscale background.
    """
    rows, columns = np.indices((size, size), dtype=np.float32)
    surface = (rng.uniform(150, 245) + rng.uniform(-30, 30) * columns / size
               + rng.uniform(-30, 30) * rows / size)
    surface += rng.normal(0, rng.uniform(0.5, 3), surface.shape)
    image = Image.fromarray(np.clip(surface, 0, 255).astype(np.uint8))
    drawing = ImageDraw.Draw(image)
    for _ in range(int(rng.integers(0, 4))):
        drawing.line(tuple(rng.integers(0, size, 4).tolist()),
                     fill=int(rng.integers(70, 200)), width=int(rng.integers(1, 3)))
    return image


def target_corners(rng: np.random.Generator, shape: tuple[int, int],
                    size: int, settings: dict) -> tuple[np.ndarray, float]:
    """Place a complete rotated symbol on the canvas with mild perspective distortion.

    :param rng:         Per-sample random generator.
    :param shape:       Symbol width and height, including quiet zones.
    :param size:        Square scene resolution.
    :param settings:    Source frame height and symbol size fraction.
    :returns:           Destination corners and nominal rotation in degrees.
    """
    dimensions = np.array(shape, dtype=np.float64)
    dimensions *= settings["fraction"] * size / dimensions.max()
    corners = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    corners *= dimensions
    corners += rng.uniform(-0.025, 0.025, (4, 2)) * dimensions
    angle = float(rng.uniform(-180, 180))
    cosine, sine = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    corners = corners @ np.array([[cosine, sine], [-sine, cosine]])
    extent = np.ptp(corners, axis=0)
    # Leave spare pixels so rounding cannot invert the center-sampling interval.
    corners *= min(1.0, ((np.array([size, settings["frame_height"]]) - 16) / extent).min())
    center = np.clip(rng.normal([size * 0.52, settings["frame_height"] * 0.52], size * 0.12),
                     6 - corners.min(axis=0),
                     np.array([size, settings["frame_height"]]) - 6 - corners.max(axis=0))
    return corners + center, angle


def rectangle_corners(box: tuple) -> np.ndarray:
    """Represent an axis-aligned rectangle as four cyclic XY vertices.

    :param box:    Left, top, right and bottom coordinates.
    :returns:      Four corners starting at the upper-left vertex.
    """
    return np.array([[box[0], box[1]], [box[2], box[1]],
                     [box[2], box[3]], [box[0], box[3]]], np.float64)


def place_symbol(symbol: Image.Image, rng: np.random.Generator,
                   size: int, settings: dict) -> tuple[Image.Image, np.ndarray, float]:
    """Project a complete symbol onto a neutral surface.

    :param symbol:      Zint raster with quiet zones and no human-readable text.
    :param rng:         Per-sample random generator.
    :param size:        Square output resolution.
    :param settings:    Source frame geometry and scene group.
    :returns:           Grayscale scene, visible-symbol corners and nominal rotation.
    """
    symbol = symbol.convert("L")
    ink = ImageChops.difference(symbol, Image.new("L", symbol.size, 255)).getbbox()
    if ink is None:
        raise ValueError("Zint produced an empty symbol")
    source = rectangle_corners((0, 0, symbol.width, symbol.height))
    corners, angle = target_corners(rng, symbol.size, size, settings)
    matrix = homography(source, corners)
    inverse = np.linalg.inv(matrix)
    inverse /= inverse[2, 2]
    transform = tuple(inverse.ravel()[:8])
    paper = symbol.transform((size, size), Image.Transform.PERSPECTIVE, transform,
                             resample=Image.Resampling.BICUBIC, fillcolor=255)
    mask = Image.new("L", symbol.size, 255).transform(
        (size, size), Image.Transform.PERSPECTIVE, transform,
        resample=Image.Resampling.BICUBIC, fillcolor=0)
    canvas = background(rng, size)
    canvas.paste(paper, mask=mask)
    return canvas, transform_points(rectangle_corners(ink), matrix), angle


def illumination(rng: np.random.Generator, shape: tuple, depth: float) -> tuple[np.ndarray, dict]:
    """Simulate smooth shading and a broad soft shadow across the printed symbol.

    :param rng:     Per-sample random generator.
    :param shape:   Image height and width.
    :param depth:   Maximum fractional shadow attenuation.
    :returns:       Illumination field and its generating parameters.
    """
    rows, columns = np.indices(shape, dtype=np.float32)
    center = rng.uniform(-0.1, 1.1, 2) * np.array(shape[::-1])
    radius = rng.uniform(0.3, 0.8, 2) * np.array(shape[::-1])
    slope = rng.uniform(-0.15, 0.15, 2)
    distance = ((columns - center[0]) / radius[0]) ** 2 + ((rows - center[1]) / radius[1]) ** 2
    light = (1 - depth * np.exp(-0.5 * distance)
             + slope[0] * (columns / shape[1] - 0.5) + slope[1] * (rows / shape[0] - 0.5))
    return np.clip(light, 0.4, 1.1), {"shadow_center": center.tolist(),
                                     "shadow_radius": radius.tolist(),
                                     "lighting_slope": slope.tolist()}


def degrade_image(canvas: Image.Image, rng: np.random.Generator,
                    settings: dict) -> tuple[Image.Image, dict]:
    """Apply mild sensor noise, brightness changes and optical blur.

    :param canvas:      Grayscale scene.
    :param rng:         Per-sample random generator.
    :param settings:    Photometric ranges and source frame height.
    :returns:           Final image and degradation parameters.
    """
    canvas = canvas.crop((0, 0, canvas.width, settings["frame_height"]))
    canvas = canvas.filter(ImageFilter.GaussianBlur(settings["blur"]))
    pixels = np.asarray(canvas, dtype=np.float32) * settings["gain"] + settings["offset"]
    light, parameters = illumination(rng, pixels.shape, settings["shadow"])
    pixels *= light
    pixels += rng.normal(0, settings["noise"], pixels.shape)
    return (Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)),
            dict(settings, **parameters))


def render_symbol(symbol: Image.Image, rng: np.random.Generator,
                    size: int, category: str, profile: str) -> tuple[Image.Image, np.ndarray, dict]:
    """Render a symbol and return its geometry and scene parameters.

    :param symbol:      Zint raster with quiet zones and no human-readable text.
    :param rng:         Per-sample random generator.
    :param size:        Square output resolution.
    :param category:    Checkpoint class name, used for dense-symbol size limits.
    :param profile:     Matched, clear or difficult scene group.
    :returns:           Final grayscale image, symbol corners and degradation parameters.
    """
    settings = scene_parameters(rng, size, category, profile)
    canvas, polygon, angle = place_symbol(symbol, rng, size, settings)
    image, parameters = degrade_image(canvas, rng, settings)
    image, padding = pad_frame(image, size)
    polygon += [0, padding]
    parameters["rotation_degrees"] = angle
    return image, polygon, parameters


def pad_frame(image: Image.Image, size: int) -> tuple[Image.Image, int]:
    """Apply the same neutral letterbox fill visible in the reference BMPs.

    :param image:   Wide source frame.
    :param size:    Square output resolution.
    :returns:       Padded image and top padding in pixels.
    """
    padding = (size - image.height) // 2
    return (ImageOps.expand(image, (0, padding, 0, size - image.height - padding), fill=114),
            padding)


def render_background(rng: np.random.Generator, size: int,
                        profile: str) -> tuple[Image.Image, dict]:
    """Use the same lighting, noise and letterbox distribution for negative samples.

    :param rng:     Per-sample random generator.
    :param size:    Square output resolution.
    :param profile: Matched, clear or difficult scene group.
    :returns:       Background-only scene and degradation parameters.
    """
    settings = scene_parameters(rng, size, "BACKGROUND", profile)
    image, parameters = degrade_image(background(rng, size), rng, settings)
    image, _ = pad_frame(image, size)
    return image, parameters
