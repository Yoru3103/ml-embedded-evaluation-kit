# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Static ONNX adaptation and four-output DFL deployment checks."""

import operator
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
from onnx2torch import convert
from onnx2torch.node_converters.binary_math_operations import OnnxBinaryMathOperation
from onnx2torch.node_converters.expand import OnnxExpand
from onnx2torch.node_converters.matmul import OnnxMatMul
from onnx2torch.node_converters.reshape import OnnxReshape
from onnx2torch.node_converters.resize import OnnxResize
from onnx2torch.node_converters.slice import OnnxSlice
import torch
from torch.fx.passes.shape_prop import ShapeProp
from executorch.exir.schema import ExecutionPlan, ScalarType


def describe_model(path: Path) -> tuple[onnx.ModelProto, dict]:
    """Check the fixed RGB input and three DFL heads plus objectness output.

    :param path:    Source ONNX model.
    :returns:       Checked graph and its ordered input/output metadata.
    """
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 4:
        raise ValueError("Expected one input and four raw DFL/objectness outputs")
    values = list(graph.graph.input) + list(graph.graph.output)
    tensors = []
    for value in values:
        tensor = value.type.tensor_type
        shape = [dim.dim_value for dim in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or any(dim <= 0 for dim in shape):
            raise ValueError("Only static float32 ONNX tensors are supported")
        tensors.append({"name": value.name, "shape": shape})
    shape = tensors[0]["shape"]
    if len(shape) != 4 or shape[:2] != [1, 3] or shape[2] != shape[3] or shape[2] % 32:
        raise ValueError("Expected square RGB NCHW [1,3,H,H], with H a multiple of 32")
    heads = [item["shape"] for item in tensors[1:] if len(item["shape"]) == 4]
    sizes = sorted([head[2] for head in heads], reverse=True)
    if (len(heads) != 3 or sizes != [shape[2] // stride for stride in (8, 16, 32)]
            or any(head[0] != 1 or head[1] <= 64 or head[2] != head[3] for head in heads)
            or len({head[1] for head in heads}) != 1):
        raise ValueError("Expected three NCHW heads with 64 DFL channels plus class logits")
    anchors = sum(size * size for size in sizes)
    auxiliary = [item["shape"] for item in tensors[1:] if len(item["shape"]) != 4]
    if auxiliary != [[1, 1, anchors]]:
        raise ValueError("Expected objectness logits [1,1,total anchors]")
    return graph, {"input": tensors[0], "outputs": tensors[1:],
                   "image_size": shape[2], "num_classes": heads[0][1] - 64}


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
    module = model.get_submodule(node.target)
    scales = constant(model, node.args[2]).tolist()
    if (module.onnx_mode != "nearest" or scales[:2] != [1, 1]
            or len(scales) != 4 or any(scale <= 0 for scale in scales)):
        raise ValueError("Only fixed spatial nearest-neighbor Resize is supported")
    if len(node.args) > 3 and constant(model, node.args[3]).numel():
        raise ValueError("Resize sizes alongside scales are unsupported")
    model.set_submodule(node.target, torch.nn.Upsample(
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
    shape = tuple(constant(model, node.args[1]).tolist())
    for user in node.users:
        if user.op != "call_module":
            raise ValueError("Expand must feed an elementwise multiplication")
        module = model.get_submodule(user.target)
        other = user.args[1] if user.args[0] is node else user.args[0]
        if (not isinstance(module, OnnxBinaryMathOperation)
                or module.math_op_function is not torch.mul
                or tuple(other.meta["tensor_meta"].shape) != shape):
            raise ValueError("Cannot safely fold this Expand into its consumers")
    node.replace_all_uses_with(node.args[0])
    model.graph.erase_node(node)


def adapt_model(graph: onnx.ModelProto, example: torch.Tensor) -> torch.fx.GraphModule:
    """Convert ONNX and specialize constant operations for both export backends.

    :param graph:       Checked static ONNX graph.
    :param example:     Input used for shape propagation.
    :returns:           Exportable model retaining all raw outputs in ONNX order.
    """
    model = convert(graph).eval()
    with torch.no_grad():
        ShapeProp(model).propagate(example)
    for node in list(model.graph.nodes):
        if node.op != "call_module":
            continue
        module = model.get_submodule(node.target)
        if isinstance(module, OnnxResize):
            replace_resize(model, node)
        elif isinstance(module, torch.nn.ReLU):
            # Conv/ReLU fusion shares output qparams with concatenation branches.
            # Arm's fusion pass requires zero_point == quant_min, which is not
            # guaranteed for those shared qparams. An equivalent clamp keeps the
            # convolution separately quantized and eligible for NPU delegation.
            node.op, node.target = "call_function", torch.clamp
            node.kwargs = {"min": 0.0}
        elif isinstance(module, OnnxSlice):
            replace_slice(model, node)
        elif isinstance(module, OnnxExpand):
            remove_expand(model, node)
        elif isinstance(module, OnnxReshape):
            shape = tuple(int(dim) for dim in constant(model, node.args[1]).tolist())
            if 0 in shape:
                raise ValueError("Reshape dimensions copied with zero are unsupported")
            node.op, node.target = "call_method", "reshape"
            node.args = (node.args[0], shape)
        elif isinstance(module, OnnxMatMul) and node.args[1].op == "get_attr":
            weight = constant(model, node.args[1])
            if weight.ndim != 2:
                raise ValueError("Expected two-dimensional constant MatMul weights")
            linear = torch.nn.Linear(weight.shape[0], weight.shape[1], bias=False)
            linear.weight = torch.nn.Parameter(weight.T.contiguous(), requires_grad=False)
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


def validate_outputs(predict: Callable, samples: list[dict], strict: bool,
                     max_score_error: float = 0.15, min_box_iou: float = 0.9) -> list[dict]:
    """Check every ordered output against ONNX, optionally requiring FP32 parity.

    :param predict:         Callable receiving a NumPy RGB NCHW input.
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
            for reference, observed in zip(sample["reference"], actual):
                np.testing.assert_allclose(observed, reference, rtol=1e-4, atol=1e-4)
        result = {"image": sample["image"], "outputs": errors}
        if not strict:
            metrics = detection_metrics(sample["reference"], actual, sample["input"].shape[2])
            result["detection"] = metrics
            if (metrics["max_absolute_score_error"] > max_score_error
                    or metrics["box_iou_at_float_best"] < min_box_iou):
                raise ValueError(f"Quantization validation failed for {sample['image']}: {metrics}")
        results.append(result)
    return results


def decode_boxes(head: np.ndarray, image_size: int) -> np.ndarray:
    """Decode one head's DFL values to pixel-coordinate XYXY boxes.

    :param head:        Raw NCHW head with 64 DFL channels.
    :param image_size:  Input image resolution.
    :returns:           Four box coordinates for every cell.
    """
    size = head.shape[2]
    stride = image_size / size
    logits = head[0, :64].reshape(4, 16, -1)
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    distances = (probabilities * np.arange(16)[None, :, None]).sum(axis=1) * stride
    rows, columns = np.indices((size, size))
    centers = (np.stack((columns.ravel(), rows.ravel())) + 0.5) * stride
    return np.concatenate((centers - distances[:2], centers + distances[2:]), axis=0)


def detection_values(outputs: list[np.ndarray], image_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode scores and DFL boxes using the existing raw-head postprocessing rules.

    :param outputs:     Three NCHW DFL/class heads and objectness logits.
    :param image_size:  Square input resolution in pixels.
    :returns:           Class scores and XYXY boxes in descending head-size order.
    """
    heads = sorted((item for item in outputs if item.ndim == 4),
                   key=lambda item: item.shape[2], reverse=True)
    auxiliary = next(item for item in outputs if item.ndim == 3)
    scores, boxes = [], []
    for head in heads:
        boxes.append(decode_boxes(head, image_size))
        scores.append(1 / (1 + np.exp(-np.clip(head[0, 64:].reshape(head.shape[1] - 64, -1),
                                             -80, 80))))
    objectness = 1 / (1 + np.exp(-np.clip(auxiliary[0], -80, 80)))
    return np.concatenate(scores, axis=1) * objectness, np.concatenate(boxes, axis=1)


def detection_metrics(reference: list[np.ndarray], actual: list[np.ndarray],
                      image_size: int) -> dict:
    """Compare quantized scores and the box at the strongest source-model anchor.

    :param reference:   Original ONNX outputs.
    :param actual:      Dequantized converted outputs in the same order.
    :param image_size:  Input image resolution.
    :returns:           Score errors and strongest-anchor box overlap.
    """
    scores, boxes = detection_values(reference, image_size)
    observed, quantized_boxes = detection_values(actual, image_size)
    location = np.unravel_index(scores.argmax(), scores.shape)
    anchor = int(location[1])
    original_box, quantized_box = boxes[:, anchor], quantized_boxes[:, anchor]
    intersection = np.maximum(0, np.minimum(original_box[2:], quantized_box[2:])
                              - np.maximum(original_box[:2], quantized_box[:2])).prod()
    areas = [(box[2:] - box[:2]).prod() for box in (original_box, quantized_box)]
    union = sum(areas) - intersection
    difference = np.abs(scores - observed)
    return {"float_max_score": float(scores.max()), "quantized_max_score": float(observed.max()),
            "quantized_score_at_float_best": float(observed[location]),
            "max_absolute_score_error": float(difference.max()),
            "mean_absolute_score_error": float(difference.mean()),
            "box_iou_at_float_best": float(intersection / union) if union > 0 else 0.0,
            "float_best_class": int(location[0]), "float_best_anchor": anchor}


def check_pte_contract(plan: ExecutionPlan, metadata: dict) -> dict:
    """Reject serialized layouts that cannot preserve the four-output ONNX interface.

    :param plan:        Serialized ExecuTorch execution plan.
    :param metadata:    Original ordered tensor metadata.
    :returns:           Checked tensor shapes and layouts.
    """
    shapes = [metadata["input"]["shape"]] + [item["shape"] for item in metadata["outputs"]]
    if len(plan.inputs) != 1 or len(plan.outputs) != 4:
        raise ValueError("PTE must retain one input and four outputs")
    result = []
    for index, shape in zip(list(plan.inputs) + list(plan.outputs), shapes):
        tensor = plan.values[index].val
        if (tensor.sizes != shape or tensor.scalar_type != ScalarType.CHAR
                or tensor.dim_order != list(range(len(shape)))):
            raise ValueError("PTE must preserve ordered, contiguous INT8 ONNX tensor layouts")
        result.append({"shape": tensor.sizes, "dim_order": tensor.dim_order, "dtype": "int8"})
    return {"input": result[0], "outputs": result[1:]}
