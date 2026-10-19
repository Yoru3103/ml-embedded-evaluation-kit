# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Static ONNX adaptation and four-output DFL deployment checks."""

from __future__ import annotations

from importlib import import_module
import operator
from typing import TYPE_CHECKING
from typing import Callable

import numpy as np
import onnx


if TYPE_CHECKING:
    import torch
    from executorch.exir.schema import ExecutionPlan


def constant(model: torch.fx.GraphModule, node: torch.fx.Node) -> torch.Tensor:
    """Resolve a static ONNX initializer before capturing the PyTorch graph.

    :param model:   Converted graph.
    :param node:    Initializer access node.
    :returns:       Tensor stored in the graph module.
    """
    if node.op != "get_attr":
        raise ValueError(f"Expected a constant shape/weight initializer: {node}")
    value = model
    for part in str(node.target).split("."):
        value = getattr(value, part)
    return value


def replace_resize(model: torch.fx.GraphModule, node: torch.fx.Node):
    """Bind the nearest-neighbor resize scales outside torch.export.

    :param model:   Converted graph.
    :param node:    Resize operation with constant scales.
    """
    torch_api = import_module("torch")
    module = model.get_submodule(node.target)
    scales = constant(model, node.args[2]).tolist()
    if (module.onnx_mode != "nearest" or scales[:2] != [1, 1]
            or len(scales) != 4 or any(scale <= 0 for scale in scales)):
        raise ValueError("Only fixed spatial nearest-neighbor Resize is supported")
    if len(node.args) > 3 and constant(model, node.args[3]).numel():
        raise ValueError("Resize sizes alongside scales are unsupported")
    model.set_submodule(node.target, torch_api.nn.Upsample(
        scale_factor=tuple(scales[2:]), mode="nearest"))
    node.args = (node.args[0],)


def replace_slice(model: torch.fx.GraphModule, node: torch.fx.Node):
    """Express fixed positive-step ONNX slices as native Python indexing.

    :param model:   Converted graph.
    :param node:    Slice operation with constant bounds.
    """
    parameters = [constant(model, item).tolist() for item in node.args[1:]]
    starts, ends = parameters[:2]
    axes = parameters[2] if len(parameters) > 2 else list(range(len(starts)))
    steps = parameters[3] if len(parameters) > 3 else [1] * len(starts)
    if min(axes) < 0 or min(steps) <= 0:
        raise ValueError("Only nonnegative Slice axes and positive steps are supported")
    slices = [slice(None)] * (max(axes) + 1)
    for start, end, axis, step in zip(starts, ends, axes, steps):
        slices[axis] = slice(start, end, step)
    node.op, node.target = "call_function", operator.getitem
    node.args = (node.args[0], tuple(slices))


def remove_expand(model: torch.fx.GraphModule, node: torch.fx.Node):
    """Remove SE expansion only when multiplication provides identical broadcasting.

    :param model:   Graph annotated with tensor shapes.
    :param node:    Expand whose consumers multiply by tensors of the expanded shape.
    """
    torch_api = import_module("torch")
    shape = tuple(constant(model, node.args[1]).tolist())
    for user in node.users:
        if user.op != "call_module":
            raise ValueError("Expand must feed an elementwise multiplication")
        module = model.get_submodule(user.target)
        other = user.args[1] if user.args[0] is node else user.args[0]
        if (not isinstance(module, import_module(
                "onnx2torch.node_converters.binary_math_operations").OnnxBinaryMathOperation)
                or module.math_op_function is not torch_api.mul
                or tuple(other.meta["tensor_meta"].shape) != shape):
            raise ValueError("Cannot safely fold this Expand into its consumers")
    node.replace_all_uses_with(node.args[0])
    model.graph.erase_node(node)


def adapt_model(graph: onnx.ModelProto, example: torch.Tensor) -> torch.fx.GraphModule:
    """Convert ONNX and specialize constant operations for both export backends.

    :param graph:   Checked static ONNX graph.
    :param example: Input used for shape propagation.
    :returns:       Exportable model retaining all raw outputs in ONNX order.
    """
    torch_api = import_module("torch")
    converters = {name: getattr(import_module(f"onnx2torch.node_converters.{module}"), name)
                  for module, name in (("resize", "OnnxResize"), ("slice", "OnnxSlice"),
                                       ("expand", "OnnxExpand"), ("reshape", "OnnxReshape"),
                                       ("matmul", "OnnxMatMul"))}
    model = import_module("onnx2torch").convert(graph).eval()
    with torch_api.no_grad():
        import_module("torch.fx.passes.shape_prop").ShapeProp(model).propagate(example)
    for node in list(model.graph.nodes):
        if node.op != "call_module":
            continue
        module = model.get_submodule(node.target)
        if isinstance(module, converters["OnnxResize"]):
            replace_resize(model, node)
        elif isinstance(module, torch_api.nn.ReLU):
            # Conv/ReLU fusion shares output qparams with concatenation branches.
            # Arm's fusion pass requires zero_point == quant_min, which is not
            # guaranteed for those shared qparams. An equivalent clamp keeps the
            # convolution separately quantized and eligible for NPU delegation.
            node.op, node.target = "call_function", torch_api.clamp
            node.kwargs = {"min": 0.0}
        elif isinstance(module, converters["OnnxSlice"]):
            replace_slice(model, node)
        elif isinstance(module, converters["OnnxExpand"]):
            remove_expand(model, node)
        elif isinstance(module, converters["OnnxReshape"]):
            shape = tuple(int(dim) for dim in constant(model, node.args[1]).tolist())
            if 0 in shape:
                raise ValueError("Reshape dimensions copied with zero are unsupported")
            node.op, node.target = "call_method", "reshape"
            node.args = (node.args[0], shape)
        elif isinstance(module, converters["OnnxMatMul"]) and node.args[1].op == "get_attr":
            weight = constant(model, node.args[1])
            if weight.ndim != 2:
                raise ValueError("Expected two-dimensional constant MatMul weights")
            linear = torch_api.nn.Linear(weight.shape[0], weight.shape[1], bias=False)
            linear.weight = torch_api.nn.Parameter(weight.T.contiguous(), requires_grad=False)
            model.set_submodule(node.target, linear)
            node.args = (node.args[0],)
    model.graph.eliminate_dead_code()
    model.delete_all_unused_submodules()
    model.graph.lint()
    model.recompile()
    return model


def compare_tensors(reference: list[np.ndarray], actual: list[np.ndarray]) -> list[dict]:
    """Measure raw-output differences without treating logits as decoded probabilities.

    :param reference:   ONNX outputs in graph order.
    :param actual:      Converted outputs in the same order.
    :returns:           Per-output error statistics.
    """
    if len(reference) != len(actual):
        raise ValueError("Output count changed during conversion")
    results = []
    for expected, observed in zip(reference, actual):
        if (expected.shape != observed.shape or not np.isfinite(observed).all()
                or not np.isfinite(expected).all()):
            raise ValueError("Output shape changed or contains non-finite values")
        difference = np.abs(expected - observed)
        results.append({"shape": list(expected.shape),
                        "max_absolute_error": float(difference.max()),
                        "mean_absolute_error": float(difference.mean())})
    return results


def check_float_outputs(reference: list[np.ndarray], actual: list[np.ndarray], image: str):
    """Check raw floating outputs with a separate bound for the angle-logit branch.

    :param reference:   Ordered ONNX outputs.
    :param actual:      Outputs restored to the same layout.
    :param image:       Validation image path included in failure diagnostics.
    """
    for index, (expected, observed) in enumerate(zip(reference, actual)):
        # Measured across all 543 validation images: the three detection heads
        # need atol < 1e-4, while angle logits need atol <= 4.26e-4 at rtol=1e-4.
        # CPU Conv accumulation differs between runtimes. Bound angle rounding
        # separately without relaxing detection-head or INT8 acceptance limits.
        angle = expected.ndim == 3 and expected.shape[:2] == (1, 1)
        np.testing.assert_allclose(
            observed, expected, rtol=1e-4, atol=5e-4 if angle else 2e-4,
            err_msg=f"FP32 parity: image={image}, output={index}, shape={expected.shape}")


def validate_outputs(predict: Callable, samples: list[dict], strict: bool,
                     max_score_error: float = 0.15, min_box_iou: float = 0.9) -> list[dict]:
    """Check every ordered output against ONNX, optionally requiring FP32 parity.

    :param predict:         Callable receiving a NumPy RGB/grayscale NCHW input.
    :param samples:         Images, inputs and cached ONNX reference outputs.
    :param strict:          Require floating-point numerical parity when true.
    :param max_score_error: Largest permitted absolute detection-score difference.
    :param min_box_iou:     Minimum box overlap at the strongest FP32 anchor.
    :returns:               Per-image, per-output and detection error statistics.
    """
    results = []
    for sample in samples:
        actual = predict(sample["input"])
        errors = compare_tensors(sample["reference"], actual)
        if strict:
            check_float_outputs(sample["reference"], actual, sample["image"])
        result = {"image": sample["image"], "outputs": errors}
        if not strict:
            metrics = detection_metrics(sample["reference"], actual, sample["input"].shape[2],
                                        sample["num_classes"])
            result["detection"] = metrics
            if (metrics["max_absolute_score_error"] > max_score_error
                    or metrics["box_iou_at_float_best"] < min_box_iou):
                raise ValueError(f"Quantization validation failed for {sample['image']}: {metrics}")
        results.append(result)
    return results


def _dfl_distances(head: np.ndarray, dfl_bins: int, stride: float) -> np.ndarray:
    """Decode four discrete distance distributions into pixel distances.

    :param head:        Raw NCHW DFL/class head.
    :param dfl_bins:    Number of bins per distance distribution.
    :param stride:      Pixels per feature-map cell.
    :returns:           Left, top, right and bottom distances for every cell.
    """
    logits = head[0, :4 * dfl_bins].reshape(4, dfl_bins, -1)
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return (probabilities * np.arange(dfl_bins)[None, :, None]).sum(axis=1) * stride


def decode_boxes(head: np.ndarray, image_size: int, angles: np.ndarray | None = None,
                 dfl_bins: int = 16) -> np.ndarray:
    """Decode one head's DFL values to pixel-coordinate XYXY boxes.

    :param head:        Raw NCHW DFL/class head.
    :param image_size:  Input image resolution.
    :param angles:      Decoded angle radians for this feature map.
    :param dfl_bins:    Number of bins per DFL distribution.
    :returns:           Four box coordinates for every cell.
    """
    size = head.shape[2]
    stride = image_size / size
    distances = _dfl_distances(head, dfl_bins, stride)
    rows, columns = np.indices((size, size))
    centers = (np.stack((columns.ravel(), rows.ravel())) + 0.5) * stride
    if angles is None:
        return np.concatenate((centers - distances[:2], centers + distances[2:]), axis=0)
    offsets = (distances[2:] - distances[:2]) * 0.5
    cosine, sine = np.cos(angles), np.sin(angles)
    centers += np.stack((offsets[0] * cosine - offsets[1] * sine,
                         offsets[0] * sine + offsets[1] * cosine))
    dimensions = distances[:2] + distances[2:]
    extents = np.stack((np.abs(cosine) * dimensions[0] + np.abs(sine) * dimensions[1],
                        np.abs(sine) * dimensions[0] + np.abs(cosine) * dimensions[1])) * 0.5
    return np.concatenate((centers - extents, centers + extents), axis=0)


def detection_values(outputs: list[np.ndarray], image_size: int,
                     num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode class-only confidence and envelopes of oriented DFL boxes.

    :param outputs:     Three NCHW DFL/class heads and raw angle logits.
    :param image_size:  Square input resolution in pixels.
    :param num_classes: Class count from the checked model contract.
    :returns:           Class scores and XYXY boxes in descending head-size order.
    """
    heads = sorted((item for item in outputs if item.ndim == 4),
                   key=lambda item: item.shape[2], reverse=True)
    auxiliary = next(item for item in outputs if item.ndim == 3)
    scores, boxes, offset = [], [], 0
    angle_values = (1 / (1 + np.exp(-np.clip(auxiliary[0, 0], -80, 80))) - 0.25) * np.pi
    for head in heads:
        if num_classes <= 0 or (head.shape[1] - num_classes) % 4:
            raise ValueError("Invalid DFL/class channel split")
        bins = (head.shape[1] - num_classes) // 4
        if not 2 <= bins <= 32:
            raise ValueError("Invalid DFL bin count")
        cells = head.shape[2] * head.shape[3]
        boxes.append(decode_boxes(head, image_size, angle_values[offset:offset + cells], bins))
        scores.append(1 / (1 + np.exp(-np.clip(head[0, 4 * bins:].reshape(num_classes, -1),
                                             -80, 80))))
        offset += cells
    if offset != angle_values.size:
        raise ValueError("Angle output size does not match feature-map anchor count")
    return np.concatenate(scores, axis=1), np.concatenate(boxes, axis=1)


def _box_iou(original: np.ndarray, observed: np.ndarray) -> float:
    """Measure the overlap of two display envelopes in XYXY coordinates.

    :param original:    Source-model envelope.
    :param observed:    Converted-model envelope.
    :returns:           Intersection over union.
    """
    intersection = np.maximum(0, np.minimum(original[2:], observed[2:])
                              - np.maximum(original[:2], observed[:2])).prod()
    union = sum((box[2:] - box[:2]).prod() for box in (original, observed)) - intersection
    return float(intersection / union) if union > 0 else 0.0


def detection_metrics(reference: list[np.ndarray], actual: list[np.ndarray],
                      image_size: int, num_classes: int) -> dict:
    """Compare quantized scores and the box at the strongest source-model anchor.

    :param reference:   Original ONNX outputs.
    :param actual:      Dequantized converted outputs in the same order.
    :param image_size:  Input image resolution.
    :param num_classes: Class count from the checked model contract.
    :returns:           Score errors and strongest-anchor box overlap.
    """
    scores, boxes = detection_values(reference, image_size, num_classes)
    observed, quantized_boxes = detection_values(actual, image_size, num_classes)
    location = np.unravel_index(scores.argmax(), scores.shape)
    anchor = int(location[1])
    difference = np.abs(scores - observed)
    return {"float_max_score": float(scores.max()), "quantized_max_score": float(observed.max()),
            "quantized_score_at_float_best": float(observed[location]),
            "max_absolute_score_error": float(difference.max()),
            "mean_absolute_score_error": float(difference.mean()),
            "box_iou_at_float_best": _box_iou(boxes[:, anchor], quantized_boxes[:, anchor]),
            "float_best_class": int(location[0]), "float_best_anchor": anchor,
            "quantized_best_class": int(np.unravel_index(observed.argmax(), observed.shape)[0])}


def check_pte_contract(plan: ExecutionPlan, metadata: dict, layout: str = "nchw",
                       output_type: str = "int8") -> dict:
    """Reject serialized layouts that cannot preserve the four-output ONNX interface.

    :param plan:        Serialized ExecuTorch execution plan.
    :param metadata:    Original ordered tensor metadata.
    :param layout:      NCHW, NHWC, or NCHW logical sizes with channels_last storage.
    :param output_type: INT8 heads or model-side dequantized FP32 heads.
    :returns:           Checked tensor shapes and layouts.
    """
    scalar_type = import_module("executorch.exir.schema").ScalarType
    if output_type not in ("int8", "float32"):
        raise ValueError(f"Unsupported PTE output type: {output_type}")
    shapes = [metadata["input"]["shape"]] + [item["shape"] for item in metadata["outputs"]]
    if layout == "nhwc":
        shapes = [[shape[index] for index in (0, 2, 3, 1)] if len(shape) == 4 else shape
                  for shape in shapes]
    elif layout not in ("nchw", "channels_last"):
        raise ValueError(f"Unsupported PTE layout: {layout}")
    if len(plan.inputs) != 1 or len(plan.outputs) != 4:
        raise ValueError("PTE must retain one input and four outputs")
    result = []
    for index, shape in zip(list(plan.inputs) + list(plan.outputs), shapes):
        tensor = plan.values[index].val
        dtype = "int8" if index in plan.inputs else output_type
        required = scalar_type.CHAR if dtype == "int8" else scalar_type.FLOAT
        order = ([0, 2, 3, 1] if layout == "channels_last" and len(shape) == 4
                 else list(range(len(shape))))
        if (tensor.sizes != shape or tensor.scalar_type != required
                or tensor.dim_order != order):
            raise ValueError(f"PTE requires ordered, contiguous {dtype} {layout.upper()} tensors")
        result.append({"shape": tensor.sizes, "dim_order": tensor.dim_order, "dtype": dtype})
    return {"input": result[0], "outputs": result[1:]}
