# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Reference Top1 decoding for raw DFL/class heads and OBB angle logits."""

import math

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Compute stable independent class probabilities.

    :param values:  Raw logits.
    :returns:       Sigmoid probabilities.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80, 80)))


def maximum_classes(head: np.ndarray, num_classes: int, bins: int) -> tuple:
    """Read each cell's maximum class and probability.

    :param head:        One NCHW detection head.
    :param num_classes: Class count.
    :param bins:        Bins per distribution.
    :returns:           Per-cell class indices and maximum probabilities.
    """
    logits = head[0, 4 * bins:].reshape(num_classes, -1).astype(np.float64)
    classes = logits.argmax(axis=0)
    return classes, sigmoid(logits[classes, np.arange(logits.shape[1])])


def best_cell(heads: list[np.ndarray], num_classes: int, bins: int, threshold: float) -> dict:
    """Select globally while retaining first-cell tie handling.

    :param heads:       Heads in descending spatial size.
    :param num_classes: Class count.
    :param bins:        Bins per distribution.
    :param threshold:   Inclusive score threshold.
    :returns:           Winning cell and total passing positions.
    """
    offset, count = 0, 0
    winner = {"score": -math.inf}
    for head_index, head in enumerate(heads):
        classes, scores = maximum_classes(head, num_classes, bins)
        count += int((scores >= threshold).sum())
        cell = int(scores.argmax())
        if scores[cell] > winner["score"]:
            winner = {"head_index": head_index, "head": f"p{head_index + 3}",
                      "cell": cell, "global_cell": offset + cell,
                      "class_id": int(classes[cell]), "score": float(scores[cell])}
        offset += scores.size
    if "cell" not in winner:
        raise ValueError("No OBB detection heads")
    winner["candidates_above_threshold"] = count
    return winner


def dfl_distances(head: np.ndarray, cell: int, bins: int) -> np.ndarray:
    """Decode four distribution expectations in grid units.

    :param head:    One NCHW head.
    :param cell:    Selected spatial index.
    :param bins:    Bins per distribution.
    :returns:       Left, top, right and bottom distances.
    """
    logits = head[0, :4 * bins].reshape(4, bins, -1)[:, :, cell].astype(np.float64)
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (probabilities * np.arange(bins)).sum(axis=1) / probabilities.sum(axis=1)


def decode_box(head: np.ndarray, cell: int, distances: np.ndarray,
               angle_raw: float, image_size: int) -> list[float]:
    """Rotate the anchor-to-center displacement and recover pixel dimensions.

    :param head:        Selected NCHW head.
    :param cell:        Selected cell.
    :param distances:   DFL distances in grid units.
    :param angle_raw:   Raw angle logit.
    :param image_size:  Square image dimensions.
    :returns:           Center, width/height and angle in radians.
    """
    angle = (float(sigmoid(np.array(angle_raw))) - 0.25) * math.pi
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    stride = image_size / head.shape[2]
    center = (np.array([cell % head.shape[2] + 0.5, cell // head.shape[2] + 0.5]) +
              rotation @ ((distances[2:] - distances[:2]) / 2)) * stride
    dimensions = (distances[:2] + distances[2:]) * stride
    return [*center.tolist(), *dimensions.tolist(), angle]


def display_envelope(box: list[float], image_size: int) -> list[int]:
    """Generate the firmware's clipped integer display rectangle.

    :param box:         Center, dimensions and angle.
    :param image_size:  Image dimensions.
    :returns:           Clipped x/y/width/height.
    """
    center_x, center_y, width, height, angle = box
    cosine, sine = math.cos(angle), math.sin(angle)
    envelope_width = width * abs(cosine) + height * abs(sine)
    envelope_height = width * abs(sine) + height * abs(cosine)
    x0 = min(image_size - 1, max(0, int(center_x - envelope_width / 2)))
    y0 = min(image_size - 1, max(0, int(center_y - envelope_height / 2)))
    x1 = min(image_size, max(0, int(center_x + envelope_width / 2)))
    y1 = min(image_size, max(0, int(center_y + envelope_height / 2)))
    return [x0, y0, x1 - x0, y1 - y0]


def decode_top1(outputs: list[np.ndarray], image_size: int, num_classes: int,
                dfl_bins: int, threshold: float = 0.25) -> dict:
    """Decode the highest-scoring cell using the firmware's OBB convention.

    :param outputs:     Three NCHW heads plus one flattened angle tensor.
    :param image_size:  Square input dimensions.
    :param num_classes: Number of independent class logits.
    :param dfl_bins:    Distribution bins per box side.
    :param threshold:   Inclusive score threshold.
    :returns:           Highest-scoring cell, rotation and clipped display envelope.
    """
    heads = sorted((value for value in outputs if value.ndim == 4),
                   key=lambda value: value.shape[2], reverse=True)
    angles = next(value for value in outputs if value.ndim == 3).ravel()
    if sum(head.shape[2] ** 2 for head in heads) != angles.size:
        raise ValueError("Invalid raw OBB output shapes")
    winner = best_cell(heads, num_classes, dfl_bins, threshold)
    head = heads[winner["head_index"]]
    winner["angle_raw"] = float(angles[winner["global_cell"]])
    winner["obb_cxcywh_radians"] = decode_box(
        head, winner["cell"], dfl_distances(head, winner["cell"], dfl_bins),
        winner["angle_raw"], image_size)
    winner["display_xywh"] = display_envelope(winner["obb_cxcywh_radians"], image_size)
    winner.update({"y": winner["cell"] // head.shape[2],
                   "x": winner["cell"] % head.shape[2],
                   "detected": winner["score"] >= threshold and
                   min(winner["display_xywh"][2:]) > 0})
    return winner


def box_corners(box: list[float]) -> list[list[float]]:
    """Convert center/size/angle into four positively oriented vertices.

    :param box: Center x/y, width, height and angle in radians.
    :returns:   Four corner coordinates.
    """
    center_x, center_y, width, height, angle = box
    cosine, sine = math.cos(angle), math.sin(angle)
    return [[center_x + x * width * cosine / 2 - y * height * sine / 2,
             center_y + x * width * sine / 2 + y * height * cosine / 2]
            for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def polygon_area(points: list[list[float]]) -> float:
    """Calculate a polygon's area with reduced cancellation error.

    :param points:  Ordered vertices.
    :returns:       Absolute area.
    """
    if len(points) < 3:
        return 0.0
    shifted = [(x - points[0][0], y - points[0][1]) for x, y in points]
    return abs(sum(first[0] * second[1] - second[0] * first[1]
                   for first, second in zip(shifted, shifted[1:] + shifted[:1]))) / 2


def signed_side(point: list[float], start: list[float], end: list[float]) -> float:
    """Calculate an edge cross product.

    :param point:   Polygon vertex.
    :param start:   Edge start.
    :param end:     Edge end.
    :returns:       Signed edge distance scaled by edge length.
    """
    return ((end[0] - start[0]) * (point[1] - start[1]) -
            (end[1] - start[1]) * (point[0] - start[0]))


def clip_edge(subject: list[list[float]], start: list[float], end: list[float]) -> list:
    """Clip a convex polygon against one positive half-plane.

    :param subject: Ordered polygon vertices.
    :param start:   Clipping edge start.
    :param end:     Clipping edge end.
    :returns:       Remaining polygon.
    """
    if not subject:
        return []
    previous = subject[-1]
    previous_cross = signed_side(previous, start, end)
    previous_inside, result = previous_cross >= -1e-10, []
    for point in subject:
        cross = signed_side(point, start, end)
        inside = cross >= -1e-10
        if inside != previous_inside:
            fraction = previous_cross / (previous_cross - cross)
            result.append([previous[0] + fraction * (point[0] - previous[0]),
                           previous[1] + fraction * (point[1] - previous[1])])
        if inside:
            result.append(point)
        previous, previous_cross, previous_inside = point, cross, inside
    return result


def rotated_iou(first: list[float], second: list[float]) -> float:
    """Intersect two rectangles by double-precision convex polygon clipping.

    :param first:   First center/size/angle box.
    :param second:  Second center/size/angle box.
    :returns:       Rotation-aware intersection over union.
    """
    subject, clip = box_corners(first), box_corners(second)
    first_area, second_area = polygon_area(subject), polygon_area(clip)
    for start, end in zip(clip, clip[1:] + clip[:1]):
        subject = clip_edge(subject, start, end)
    intersection = min(polygon_area(subject), first_area, second_area)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0
