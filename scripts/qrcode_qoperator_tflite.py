# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Preserve a static INT8 QOperator graph while translating it to NHWC TFLite.

This frontend supports the operators in the supplied grayscale QR model. It
does not recover floating weights or recalibrate the original quantization.
"""

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import flatbuffers
import numpy as np
import onnx
from onnx import numpy_helper


@dataclass
class Feature:
    """An INT8 tensor with its original NCHW logical shape and quantization."""

    index: int
    shape: tuple[int, ...]
    scale: float
    zero_point: int


@dataclass
class TensorSpec:
    """Physical shape, dtype, constant data and optional scalar quantization."""

    shape: tuple[int, ...]
    dtype: int
    data: np.ndarray | None = None
    quantization: tuple[float, int] | None = None


def convolution_geometry(shape: tuple[int, ...], weight: np.ndarray, attrs: dict) -> dict:
    """Calculate static ONNX output dimensions and equivalent TFLite padding.

    :param shape:   NCHW input dimensions.
    :param weight:  Original OIHW convolution weights.
    :param attrs:   Original convolution attributes.
    :returns:       Strides, dilations, explicit padding and output dimensions.
    """
    strides = attrs.get("strides", [1, 1])
    dilations = attrs.get("dilations", [1, 1])
    pads = attrs.get("pads", [0, 0, 0, 0])
    effective = [(weight.shape[index + 2] - 1) * dilations[index] + 1 for index in range(2)]
    output_hw = [(shape[index + 2] + pads[index] + pads[index + 2] -
                  effective[index]) // strides[index] + 1 for index in range(2)]
    needed = [max(0, (output_hw[index] - 1) * strides[index] + effective[index] -
                  shape[index + 2]) for index in range(2)]
    return {"strides": strides, "dilations": dilations, "pads": pads,
            "output_hw": output_hw, "same_pads": [needed[0] // 2, needed[1] // 2,
            needed[0] - needed[0] // 2, needed[1] - needed[1] // 2]}


class QOperatorTflite:
    """Translate static, per-tensor INT8 operations without changing weights."""

    def __init__(self, graph: onnx.ModelProto):
        """Initialize the source constants and FlatBuffer object model.

        :param graph:   Source ONNX model, left unchanged.
        """
        self.source = graph
        self.schema = import_module("ai_edge_litert.schema_py_generated")
        self.constants = {item.name: numpy_helper.to_array(item)
                          for item in graph.graph.initializer}
        self.features: dict[str, Feature] = {}
        self.model = self.schema.ModelT()
        self.model.version = 3
        self.model.description = b"Original ONNX INT8 QOperator parameters; NHWC; no recalibration"
        self.model.buffers = [self.schema.BufferT()]
        self.model.operatorCodes = []
        self.codes: dict[tuple[int, int], int] = {}
        self.subgraph = self.schema.SubGraphT()
        self.subgraph.name = b"main"
        self.subgraph.tensors = []
        self.subgraph.operators = []
        self.subgraph.inputs = []
        self.subgraph.outputs = []
        self.model.subgraphs = [self.subgraph]

    def constant(self, name: str) -> np.ndarray:
        """Read a source initializer or Constant result.

        :param name:    Original tensor name.
        :returns:       Constant array.
        """
        return self.constants[name]

    def quantization(self, scale: str, zero_point: str) -> tuple[float, int]:
        """Validate scalar signed INT8 quantization.

        :param scale:       Scale initializer name.
        :param zero_point:  Zero-point initializer name.
        :returns:           Positive scale and signed zero point.
        """
        factor, offset = self.constant(scale), self.constant(zero_point)
        if factor.size != 1 or offset.size != 1 or offset.dtype != np.int8:
            raise ValueError("QOperator frontend requires per-tensor INT8 parameters")
        if not np.isfinite(factor.item()) or factor.item() <= 0:
            raise ValueError("Invalid quantization scale")
        return float(factor.item()), int(offset.item())

    def tensor(self, name: str, spec: TensorSpec) -> int:
        """Append a constant or runtime FlatBuffer tensor.

        :param name:    Tensor name.
        :param spec:    Shape, dtype, constant data and quantization.
        :returns:       Tensor index.
        """
        tensor = self.schema.TensorT()
        tensor.name = name.encode()
        tensor.shape = list(spec.shape)
        tensor.type = spec.dtype
        tensor.hasRank = True
        if spec.data is not None:
            buffer = self.schema.BufferT()
            buffer.data = np.frombuffer(np.ascontiguousarray(spec.data).tobytes(), dtype=np.uint8)
            tensor.buffer = len(self.model.buffers)
            self.model.buffers.append(buffer)
        if spec.quantization is not None:
            tensor.quantization = self.schema.QuantizationParametersT()
            tensor.quantization.scale = [spec.quantization[0]]
            tensor.quantization.zeroPoint = [spec.quantization[1]]
        index = len(self.subgraph.tensors)
        self.subgraph.tensors.append(tensor)
        return index

    def feature(self, name: str, shape: tuple[int, ...], scale: float,
                zero_point: int) -> Feature:
        """Append an NHWC INT8 tensor while retaining its logical dimensions.

        :param name:        Original or generated tensor name.
        :param shape:       NCHW dimensions for rank four, unchanged otherwise.
        :param scale:       Quantization scale.
        :param zero_point:  Quantization zero point.
        :returns:           Registered feature description.
        """
        physical = tuple(shape[index] for index in (0, 2, 3, 1)) if len(shape) == 4 else shape
        result = Feature(self.tensor(name, TensorSpec(physical, self.schema.TensorType.INT8,
                                                      quantization=(scale, zero_point))),
                         shape, scale, zero_point)
        self.features[name] = result
        return result

    def operation(self, kind: tuple[str, int], inputs: list[int], output: int, options: Any):
        """Append a TFLite operator and deduplicate its opcode entry.

        :param kind:    BuiltinOperator name and required version.
        :param inputs:  Input tensor indices.
        :param output:  Output tensor index.
        :param options: Builtin options object.
        """
        code = getattr(self.schema.BuiltinOperator, kind[0])
        key = (code, kind[1])
        if key not in self.codes:
            entry = self.schema.OperatorCodeT()
            entry.builtinCode = code
            entry.deprecatedBuiltinCode = min(code, 127)
            entry.version = kind[1]
            self.codes[key] = len(self.model.operatorCodes)
            self.model.operatorCodes.append(entry)
        operator = self.schema.OperatorT()
        operator.opcodeIndex = self.codes[key]
        operator.inputs, operator.outputs = inputs, [output]
        operator.builtinOptions = options
        operator.builtinOptionsType = getattr(self.schema.BuiltinOptions,
                                              type(options).__name__[:-1])
        self.subgraph.operators.append(operator)

    def requantize(self, value: Feature, name: str, scale: float, zero_point: int) -> Feature:
        """Align a concatenation input to its output's original quantization.

        :param value:       Original input feature.
        :param name:        Generated tensor name.
        :param scale:       Required scale.
        :param zero_point:  Required zero point.
        :returns:           Original tensor or an explicit INT8 requantization.
        """
        if value.scale == scale and value.zero_point == zero_point:
            return value
        result = self.feature(name, value.shape, scale, zero_point)
        self.operation(("QUANTIZE", 2), [value.index], result.index, self.schema.QuantizeOptionsT())
        return result

    def pad_convolution(self, node: onnx.NodeProto, source: Feature, geometry: dict) -> Feature:
        """Insert real-zero padding when ONNX padding differs from TFLite SAME.

        :param node:        Original convolution.
        :param source:      Original input feature.
        :param geometry:    Static convolution dimensions, updated with padding mode.
        :returns:           Original or explicitly padded input.
        """
        pads = geometry["pads"]
        geometry["padding"] = self.schema.Padding.VALID
        if pads == geometry["same_pads"] and any(pads):
            geometry["padding"] = self.schema.Padding.SAME
        elif any(pads):
            padding_array = np.array([[0, 0], [pads[0], pads[2]],
                                      [pads[1], pads[3]], [0, 0]], dtype=np.int32)
            padding_index = self.tensor(node.name + "/pads", TensorSpec(
                (4, 2), self.schema.TensorType.INT32, padding_array))
            fill_index = self.tensor(node.name + "/zero", TensorSpec(
                (), self.schema.TensorType.INT8, np.array(source.zero_point, dtype=np.int8),
                (source.scale, source.zero_point)))
            padded_shape = (source.shape[0], source.shape[1],
                            source.shape[2] + pads[0] + pads[2],
                            source.shape[3] + pads[1] + pads[3])
            padded = self.feature(node.name + "/padded", padded_shape,
                                  source.scale, source.zero_point)
            self.operation(("PADV2", 2), [source.index, padding_index, fill_index], padded.index,
                           self.schema.PadV2OptionsT())
            source = padded
        return source

    def convolution_constants(self, node: onnx.NodeProto, source: Feature,
                              depthwise: bool) -> list[int]:
        """Reorder INT8 filters and copy original INT32 biases without recalibration.

        :param node:        Original convolution.
        :param source:      Convolution input feature.
        :param depthwise:   Use the depthwise filter layout.
        :returns:           Filter and bias tensor indices.
        """
        weight = self.constant(node.input[3])
        weight_scale, weight_zero = self.quantization(node.input[4], node.input[5])
        if weight_zero != 0:
            raise ValueError("TFLite convolution requires symmetric INT8 weights")
        physical_weight = (weight.transpose(1, 2, 3, 0) if depthwise
                           else weight.transpose(0, 2, 3, 1))
        weight_index = self.tensor(node.input[3], TensorSpec(
            physical_weight.shape, self.schema.TensorType.INT8, physical_weight, (weight_scale, 0)))
        bias = self.constant(node.input[8]) if len(node.input) > 8 else np.zeros(
            weight.shape[0], dtype=np.int32)
        bias_index = self.tensor(node.name + "/bias", TensorSpec(
            bias.shape, self.schema.TensorType.INT32, bias, (source.scale * weight_scale, 0)))
        return [weight_index, bias_index]

    def convolution(self, node: onnx.NodeProto, attrs: dict):
        """Translate Conv or depthwise Conv, preserving explicit ONNX padding.

        :param node:    QLinearConv node.
        :param attrs:   Decoded node attributes.
        """
        source = self.features[node.input[0]]
        weight = self.constant(node.input[3])
        scale, zero = self.quantization(node.input[6], node.input[7])
        geometry = convolution_geometry(source.shape, weight, attrs)
        source = self.pad_convolution(node, source, geometry)
        groups = attrs.get("group", 1)
        depthwise = groups != 1
        if depthwise and (groups != source.shape[1] or weight.shape[1] != 1):
            raise ValueError("Only standard and depthwise convolutions are supported")
        if depthwise:
            options = self.schema.DepthwiseConv2DOptionsT()
            options.depthMultiplier = weight.shape[0] // groups
            kind = "DEPTHWISE_CONV_2D"
        else:
            options = self.schema.Conv2DOptionsT()
            kind = "CONV_2D"
        parameters = self.convolution_constants(node, source, depthwise)
        result = self.feature(node.output[0],
                              (source.shape[0], weight.shape[0], *geometry["output_hw"]),
                              scale, zero)
        options.padding = geometry["padding"]
        options.strideH, options.strideW = geometry["strides"]
        options.dilationHFactor, options.dilationWFactor = geometry["dilations"]
        self.operation((kind, 3), [source.index, *parameters], result.index, options)

    def constant_node(self, node: onnx.NodeProto, attrs: dict):
        """Translate a Constant node with the original quantization.

        :param node:    Original ONNX node.
        :param attrs:   Decoded node attributes.
        """
        self.constants[node.output[0]] = numpy_helper.to_array(attrs["value"])

    def input_boundary(self, node: onnx.NodeProto, _attrs: dict):
        """Translate a QuantizeLinear node with the original quantization.

        :param node:    Original ONNX node.
        :param _attrs:  Decoded node attributes.
        """
        if node.input[0] != self.source.graph.input[0].name or self.subgraph.inputs:
            raise ValueError("Only one floating-point input quantization boundary is supported")
        shape = tuple(dim.dim_value for dim in
                      self.source.graph.input[0].type.tensor_type.shape.dim)
        scale, zero = self.quantization(node.input[1], node.input[2])
        result = self.feature(node.output[0], shape, scale, zero)
        self.subgraph.tensors[result.index].name = node.input[0].encode()
        self.subgraph.inputs = [result.index]

    def addition(self, node: onnx.NodeProto, _attrs: dict):
        """Translate a QLinearAdd node with the original quantization.

        :param node:    Original ONNX node.
        :param _attrs:  Decoded node attributes.
        """
        first, second = (self.features[node.input[index]] for index in (0, 3))
        scale, zero = self.quantization(node.input[6], node.input[7])
        result = self.feature(node.output[0], np.broadcast_shapes(first.shape, second.shape),
                              scale, zero)
        self.operation(("ADD", 2), [first.index, second.index], result.index,
                       self.schema.AddOptionsT())

    def concatenation(self, node: onnx.NodeProto, attrs: dict):
        """Translate a QLinearConcat node with the original quantization.

        :param node:    Original ONNX node.
        :param attrs:   Decoded node attributes.
        """
        scale, zero = self.quantization(node.input[0], node.input[1])
        values = [self.requantize(self.features[node.input[index]],
                                  node.name + f"/input{index}", scale, zero)
                  for index in range(2, len(node.input), 3)]
        axis = attrs["axis"] % len(values[0].shape)
        shape = list(values[0].shape)
        shape[axis] = sum(value.shape[axis] for value in values)
        result = self.feature(node.output[0], tuple(shape), scale, zero)
        options = self.schema.ConcatenationOptionsT()
        options.axis = (0, 3, 1, 2)[axis] if len(shape) == 4 else axis
        self.operation(("CONCATENATION", 2), [value.index for value in values],
                       result.index, options)

    def pooling(self, node: onnx.NodeProto, attrs: dict):
        """Translate a MaxPool node with the original quantization.

        :param node:    Original ONNX node.
        :param attrs:   Decoded node attributes.
        """
        source = self.features[node.input[0]]
        kernel, strides = attrs["kernel_shape"], attrs["strides"]
        pads = attrs.get("pads", [0, 0, 0, 0])
        if strides != [1, 1] or pads != [kernel[0] // 2, kernel[1] // 2] * 2:
            raise ValueError("Only same-padded stride-one pooling is supported")
        result = self.feature(node.output[0], source.shape, source.scale, source.zero_point)
        options = self.schema.Pool2DOptionsT()
        options.padding = self.schema.Padding.SAME
        options.strideH, options.strideW = strides
        options.filterHeight, options.filterWidth = kernel
        self.operation(("MAX_POOL_2D", 2), [source.index], result.index, options)

    def resize(self, node: onnx.NodeProto, attrs: dict):
        """Translate a Resize node with the original quantization.

        :param node:    Original ONNX node.
        :param attrs:   Decoded node attributes.
        """
        source = self.features[node.input[0]]
        if attrs.get("mode") != b"nearest" or attrs.get("nearest_mode") != b"floor" or \
                attrs.get("coordinate_transformation_mode") != b"asymmetric":
            raise ValueError("Only asymmetric nearest/floor resize is supported")
        scales = self.constant(node.input[2])
        shape = tuple(int(size * factor) for size, factor in zip(source.shape, scales))
        if shape[:2] != source.shape[:2]:
            raise ValueError("Resize may only change spatial dimensions")
        size_index = self.tensor(node.name + "/size", TensorSpec(
            (2,), self.schema.TensorType.INT32, np.array(shape[2:], dtype=np.int32)))
        result = self.feature(node.output[0], shape, source.scale, source.zero_point)
        self.operation(("RESIZE_NEAREST_NEIGHBOR", 3), [source.index, size_index], result.index,
                       self.schema.ResizeNearestNeighborOptionsT())

    def reshape(self, node: onnx.NodeProto, _attrs: dict):
        """Translate a Reshape node with the original quantization.

        :param node:    Original ONNX node.
        :param _attrs:  Decoded node attributes.
        """
        source = self.features[node.input[0]]
        if len(source.shape) != 4 or source.shape[1] != 1:
            raise ValueError("Only the single-channel angle flattening is supported")
        shape = [int(value) for value in self.constant(node.input[1])]
        total = int(np.prod(source.shape))
        if shape != [1, 1, -1]:
            raise ValueError("Expected angle reshape [1,1,-1]")
        shape[-1] = total
        result = self.feature(node.output[0], tuple(shape), source.scale, source.zero_point)
        shape_index = self.tensor(node.name + "/shape", TensorSpec(
            (3,), self.schema.TensorType.INT32, np.array(shape, dtype=np.int32)))
        options = self.schema.ReshapeOptionsT()
        options.newShape = shape
        self.operation(("RESHAPE", 1), [source.index, shape_index], result.index, options)

    def output_boundary(self, node: onnx.NodeProto, _attrs: dict):
        """Translate a DequantizeLinear node with the original quantization.

        :param node:    Original ONNX node.
        :param _attrs:  Decoded node attributes.
        """
        if node.output[0] not in {item.name for item in self.source.graph.output}:
            raise ValueError("Only output dequantization boundaries may be removed")
        source = self.features[node.input[0]]
        self.features[node.output[0]] = source
        self.subgraph.tensors[source.index].name = node.output[0].encode()

    def translate_node(self, node: onnx.NodeProto):
        """Dispatch a supported operation without changing the source graph.

        :param node:    Original ONNX node.
        """
        attrs = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
        handlers = {
            "Constant": self.constant_node,
            "QuantizeLinear": self.input_boundary,
            "QLinearConv": self.convolution,
            "QLinearAdd": self.addition,
            "QLinearConcat": self.concatenation,
            "MaxPool": self.pooling,
            "Resize": self.resize,
            "Reshape": self.reshape,
            "DequantizeLinear": self.output_boundary,
        }
        if node.op_type not in handlers:
            raise ValueError(f"Unsupported QOperator node: {node.name} ({node.op_type})")
        handlers[node.op_type](node, attrs)

    def write(self, path: Path) -> dict:
        """Translate the graph and save a fully integer model.

        :param path:    Destination TFLite file.
        :returns:       Quantization-preservation and interface summary.
        """
        for node in self.source.graph.node:
            self.translate_node(node)
        self.subgraph.outputs = [self.features[item.name].index
                                 for item in self.source.graph.output]
        builder = flatbuffers.Builder(2 * 1024 * 1024)
        builder.Finish(self.model.Pack(builder), file_identifier=b"TFL3")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(builder.Output()))
        return {"path": str(path), "bytes": path.stat().st_size,
                "preserves_original_weights": True, "recalibrated": False,
                "layout": "NHWC", "input_output_dtype": "int8",
                "operator_count": len(self.subgraph.operators)}
