# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

"""Arm CMSIS-ExecuTorch export helpers, adapted from commit 0d600669.

Keep the upstream quantization/guard handling and strict delegate checks.
"""
import operator
import os
import re
import sys
import time
from collections import Counter
from typing import Any
from mlek_tools.pico_face.model import Method
BOUNDARY_OPS = {"quantized_decomposed::quantize_per_tensor",
                "quantized_decomposed::dequantize_per_tensor"}

def quant_config(kind: str) -> Any:
    """Select per-channel mixed-precision quantization for one method."""
    from executorch.backends.arm.quantizer import (
        get_symmetric_a16w8_quantization_config,
        get_symmetric_quantization_config,
    )

    if kind == "a8w8":
        return get_symmetric_quantization_config(is_per_channel=True)
    if kind == "a16w8":
        return get_symmetric_a16w8_quantization_config(is_per_channel=True)
    sys.exit(f"unknown quantization {kind!r}: expected a8w8 or a16w8")


def strip_guards_fn(gm: Any):
    """Drop the dead `_guards_fn` call_module that torch.export emits for some
    models; the Arm annotation passes iterate over every module unconditionally."""
    changed = False
    for node in list(gm.graph.nodes):
        if node.op == "call_module" and str(node.target) == "_guards_fn":
            gm.graph.erase_node(node)
            changed = True
    if changed:
        gm.graph.eliminate_dead_code()
        gm.recompile()


def calibrate(prepared: Any, method: Method):
    """Run the observers over the method's calibration data (or its example inputs)."""
    import torch

    batches = method.calibration() if method.calibration else [method.example_inputs]
    count, t0 = 0, time.time()
    with torch.no_grad():
        for inputs in batches:
            prepared(*inputs)
            count += 1
    print(f"[ai_layer] {method.name}: calibrated on {count} input set(s) "
          f"in {time.time() - t0:.1f} s")


def quantize_method(method: Method, spec: Any) -> Any:
    """The method's module with quantize/dequantize nodes (the fake-quant graph)."""
    import torch
    from executorch.backends.arm.quantizer import EthosUQuantizer
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    graph = torch.export.export(method.module, method.example_inputs).module()
    strip_guards_fn(graph)
    # Quantize the whole graph so the partitioner can move every node into the
    # Ethos-U delegate; only the float <-> integer boundary stays on the CPU.
    quantizer = EthosUQuantizer(spec)
    quantizer.set_global(quant_config(method.quantization))
    prepared = prepare_pt2e(graph, quantizer)
    calibrate(prepared, method)
    return convert_pt2e(prepared)


# ----------------------------------------------------------------------------
# Export


def op_name(target: Any) -> str | None:
    """`aten::mul.Tensor` for an edge or aten op object; None for graph plumbing."""
    if target is operator.getitem:
        return None
    name = getattr(target, "name", None)
    if callable(name):
        try:
            return name()
        except Exception:
            pass
    text = str(target)
    if found := re.search(r"schema = ([a-z_0-9]+::[\w.]+)", text):
        return found.group(1)
    if found := re.match(r"^([a-z_0-9]+::[\w.]+)$", text):
        return found.group(1)
    return None


def count_ops(graph_module: Any) -> tuple[int, Counter]:
    """Count backend delegates and named CPU operators in a graph."""
    delegates, cpu_ops = 0, Counter()
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        if "executorch_call_delegate" in str(node.target):
            delegates += 1
        elif name := op_name(node.target):
            cpu_ops[name] += 1
    return delegates, cpu_ops


def report_partitioning(edge: Any, name: str) -> set[str]:
    """Print the delegate count and CPU operators of one method; return the CPU ops."""
    delegates, cpu_ops = count_ops(edge.exported_program(name).graph_module)
    ops = ", ".join(f"{op} x{n}" if n > 1 else op for op, n in cpu_ops.most_common()) or "none"
    print(f"[ai_layer] {name}: {delegates} Ethos-U delegate(s); CPU operators: {ops}")
    stray = {re.sub(r"\.\w+$", "", op) for op in cpu_ops} - BOUNDARY_OPS
    if delegates != 1 or stray:
        message = (
            f"[ai_layer] warning: {name} is not a single Ethos-U delegate "
            f"({delegates} delegate(s), CPU operators beyond the quantize/dequantize "
            f"boundary: {sorted(stray) or 'none'}); set AI_LAYER_VERBOSE=1 for the reasons"
        )
        if os.environ.get("AI_LAYER_STRICT"):
            sys.exit(message)
        print(message, file=sys.stderr)
    return set(cpu_ops)


def export_model(spec: Any, model: Any, params: dict | None = None) -> tuple[bytes, set[str]]:
    """Quantize, delegate and serialize every method; return the .pte and the CPU operators."""
    import torch
    from executorch.backends.arm.ethosu import EthosUPartitioner
    from executorch.exir import (
        EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower,
    )

    programs = {}
    for method in model.get_methods(**(params or {})):
        print(f"[ai_layer] {method.name}: quantization {method.quantization}")
        programs[method.name] = torch.export.export(
            quantize_method(method, spec), method.example_inputs)

    edge = to_edge_transform_and_lower(
        programs,
        partitioner={name: [EthosUPartitioner(spec)] for name in programs},
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    cpu_ops: set[str] = set()
    for name in programs:
        cpu_ops |= report_partitioning(edge, name)

    program = edge.to_executorch(ExecutorchBackendConfig(extract_delegate_segments=False))
    for name in programs:  # operators to_executorch adds (copies, for example) need components too
        cpu_ops |= set(count_ops(program.exported_program(name).graph_module)[1])
    return bytes(program.buffer), cpu_ops


def c_number(value: float) -> str:
    """Format one float literal without changing its float32 value."""
    text = f"{float(value):.9g}"
    if not any(ch in text for ch in ".en"):  # e, inf, nan
        text += ".0"
    return text + "f"


def c_initializer(values: Any, indent: str = "    ") -> list[str]:
    """Lines of a nested brace initializer for a float array of any rank."""
    if values.ndim == 1:
        items = [c_number(v) for v in values.tolist()]
        return [indent + ", ".join(items[i : i + 8]) + "," for i in range(0, len(items), 8)]
    lines = []
    for row in values:
        lines += [indent + "{", *c_initializer(row, indent + "  "), indent + "},"]
    return lines


def params_header(params: dict[str, Any]) -> str:
    """model_params.h from model.get_params(), names as given: strings, integers
    and floats become #defines, sequences and tensors of numbers `static const
    float` arrays of the same shape."""
    import numpy as np

    lines = [
        "// Generated by create_ai_layer.py from get_params() in model/model.py -- do not edit.",
        "#pragma once",
        "",
    ]
    for name, value in params.items():
        if isinstance(value, str):
            lines.append(f'#define {name} "{value}"')
        elif isinstance(value, (bool, int)):
            lines.append(f"#define {name} {int(value)}")
        elif isinstance(value, float):
            lines.append(f"#define {name} {c_number(value)}")
        else:
            array = np.asarray(value.detach().cpu() if hasattr(value, "detach") else value,
                               dtype=np.float32)
            dims = "".join(f"[{d}]" for d in array.shape)
            lines += ["", f"static const float {name}{dims} = {{", *c_initializer(array), "};"]
    return "\n".join(lines) + "\n"
