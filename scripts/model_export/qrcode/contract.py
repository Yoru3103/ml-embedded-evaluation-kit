# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Read raw OBB head contracts from ordinary or QOperator ONNX graphs."""

from pathlib import Path
import onnx


def load_model(path: Path | onnx.ModelProto) -> onnx.ModelProto:
    """Load a private graph and restore missing Microsoft quantized operator domains.

    :param path:    Model path or an in-memory graph.
    :returns:       Graph copy with known QOperator domains restored.
    """
    graph = (onnx.load(str(path)) if isinstance(path, Path)
             else onnx.ModelProto.FromString(path.SerializeToString()))
    restored = False
    for node in graph.graph.node:
        if not node.domain and node.op_type in {"QLinearAdd", "QLinearMul", "QLinearConcat"}:
            node.domain = "com.microsoft"
            restored = True
    if restored and not any(item.domain == "com.microsoft" for item in graph.opset_import):
        graph.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    return graph


def _unwrap(name: str, producers: dict) -> onnx.NodeProto:
    """Follow layout-preserving wrappers to an output's producing operation.

    :param name:        Tensor name.
    :param producers:   Tensor-to-node lookup.
    :returns:           Underlying producing node.
    """
    node = producers[name]
    while node.op_type in {"Identity", "DequantizeLinear", "Reshape"}:
        node = producers[node.input[0]]
    return node


def _branches(node: onnx.NodeProto) -> list[str]:
    """Read concatenated data inputs without QOperator scales and zero points.

    :param node:        Concatenation operation.
    :returns:           Ordered input tensor names.
    :raises ValueError: Unsupported raw-head producer.
    """
    if node.op_type == "Concat":
        return list(node.input)
    if node.op_type == "QLinearConcat":
        return list(node.input[2::3])
    raise ValueError("Expected raw convolution heads joined by Concat or QLinearConcat")


def _channels(name: str, producers: dict, initializers: dict) -> int:
    """Read a terminal convolution's output channel count from its weight tensor.

    :param name:            Branch output tensor.
    :param producers:       Tensor-to-node lookup.
    :param initializers:    Constant tensor lookup.
    :returns:               Number of branch channels.
    :raises ValueError:     Unsupported branch producer.
    """
    node = _unwrap(name, producers)
    if node.op_type not in {"Conv", "QLinearConv"}:
        raise ValueError("Expected unactivated box, class or angle convolution output")
    weight = node.input[1 if node.op_type == "Conv" else 3]
    return int(initializers[weight].dims[0])


def _tensor_interface(graph: onnx.ModelProto) -> list[dict]:
    """Read static floating-point input and output tensor metadata.

    :param graph:       Checked source model.
    :returns:           Input followed by outputs in graph order.
    :raises ValueError: Unsupported tensor interface.
    """
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 4:
        raise ValueError("Expected one input and four raw DFL/class/angle outputs")
    tensors = []
    for value in list(graph.graph.input) + list(graph.graph.output):
        tensor = value.type.tensor_type
        shape = [dim.dim_value for dim in tensor.shape.dim]
        if tensor.elem_type != onnx.TensorProto.FLOAT or any(dim <= 0 for dim in shape):
            raise ValueError("Only static float32 ONNX tensor interfaces are supported")
        tensors.append({"name": value.name, "shape": shape})
    return tensors


def _head_contract(head: dict, producers: dict, initializers: dict) -> tuple[int, int]:
    """Read and validate the DFL/class split of a raw detection head.

    :param head:            Head name and NCHW shape.
    :param producers:       Tensor-to-node lookup.
    :param initializers:    Constant tensor lookup.
    :returns:               DFL bins and class count.
    :raises ValueError:     Unsupported or inconsistent channel split.
    """
    branches = _branches(_unwrap(head["name"], producers))
    if len(branches) != 2:
        raise ValueError("A detection head must concatenate DFL then class logits")
    box_channels, num_classes = [_channels(name, producers, initializers) for name in branches]
    if (box_channels % 4 or not 2 <= box_channels // 4 <= 32 or num_classes <= 0
            or box_channels + num_classes != head["shape"][1]):
        raise ValueError("Invalid DFL/class channel split")
    return box_channels // 4, num_classes


def describe_model(path: Path | onnx.ModelProto) -> tuple[onnx.ModelProto, dict]:
    """Check RGB/grayscale input, three raw DFL/class heads and angle logits.

    :param path:        Source model path or an in-memory model.
    :returns:           Checked graph and its ordered deployment interface.
    :raises ValueError: Unsupported tensor layout or raw-head contract.
    """
    graph = load_model(path)
    onnx.checker.check_model(graph)
    tensors = _tensor_interface(graph)
    shape = tensors[0]["shape"]
    if (len(shape) != 4 or shape[0] != 1 or shape[1] not in (1, 3)
            or shape[2] != shape[3] or shape[2] % 32):
        raise ValueError("Expected square RGB/grayscale NCHW [1,C,H,H]")
    heads = [item for item in tensors[1:] if len(item["shape"]) == 4]
    auxiliary = [item for item in tensors[1:] if len(item["shape"]) != 4]
    sizes = sorted([item["shape"][2] for item in heads], reverse=True)
    if (len(heads) != 3 or sizes != [shape[2] // stride for stride in (8, 16, 32)]
            or any(item["shape"][0] != 1 or item["shape"][2] != item["shape"][3]
                   for item in heads)):
        raise ValueError("Expected three NCHW feature maps at strides 8, 16 and 32")
    anchors = sum(size * size for size in sizes)
    if len(auxiliary) != 1 or auxiliary[0]["shape"] != [1, 1, anchors]:
        raise ValueError("Expected angle logits [1,1,total anchors]")
    producers = {name: node for node in graph.graph.node for name in node.output}
    initializers = {item.name: item for item in graph.graph.initializer}
    contracts = [_head_contract(head, producers, initializers) for head in heads]
    if len(set(contracts)) != 1:
        raise ValueError("Detection heads must have identical DFL and class counts")
    angle_branches = _branches(_unwrap(auxiliary[0]["name"], producers))
    if len(angle_branches) != 3 or any(_channels(name, producers, initializers) != 1
                                     for name in angle_branches):
        raise ValueError("Expected three raw one-channel angle branches")
    bins, num_classes = contracts[0]
    return graph, {"input": tensors[0], "outputs": tensors[1:], "image_size": shape[2],
                   "input_channels": shape[1], "num_classes": num_classes,
                   "dfl_bins": bins, "auxiliary_kind": "angle_logits"}
