# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Rebuild the ExecuTorch host interface and memory plan; retain verified NPU streams."""
from collections import Counter
import hashlib
import math
from pathlib import Path

from executorch.exir import schema
from executorch.exir.emit._emitter import ex_pytree
from executorch.exir._serialize import _deserialize_pte_binary
from executorch.exir._serialize._program import serialize_pte_binary


def calls_by_tensor(plan: object, name: str, argument: int) -> dict:
    """Index exact CPU kernel calls by the tensor at the requested argument."""
    return {call.args[argument]: call
            for chain in plan.chains for instruction in chain.instructions
            if isinstance(call := instruction.instr_args, schema.KernelCall)
            and plan.operators[call.op_index].name == name}


def compact_memory(plan: object):
    """Compact active values and assign disjoint output/intermediate allocations.

    Graph inputs are externally owned. All produced tensors receive unique aligned
    regions: removing DQ extends cache lifetimes, so the old alias plan is invalid.
    """
    active = set(plan.inputs + plan.outputs)
    for chain in plan.chains:
        assert not chain.inputs and not chain.outputs
        for instruction in chain.instructions:
            active.update(instruction.instr_args.args)
    indices = sorted(active)
    remap = {old: new for new, old in enumerate(indices)}
    plan.values = [plan.values[index] for index in indices]
    plan.inputs = [remap[index] for index in plan.inputs]
    plan.outputs = [remap[index] for index in plan.outputs]
    for chain in plan.chains:
        for instruction in chain.instructions:
            call = instruction.instr_args
            call.args = [remap[index] for index in call.args]
    offset = 0
    widths = {schema.ScalarType.CHAR: 1, schema.ScalarType.INT: 4, schema.ScalarType.FLOAT: 4}
    for index, value in enumerate(plan.values):
        tensor = value.val
        if not isinstance(tensor, schema.Tensor):
            continue
        assert tensor.data_buffer_idx == tensor.storage_offset == 0
        if index in plan.inputs:
            tensor.allocation_info = None
        else:
            offset = (offset + 63) // 64 * 64
            tensor.allocation_info = schema.AllocationDetails(1, offset, 0)
            offset += math.prod(tensor.sizes) * widths[tensor.scalar_type]
    plan.non_const_buffer_sizes = [0, (offset + 63) // 64 * 64]


def rewrite_state(plan: object, row: dict, quantize: dict, dequantize: dict) -> set[int]:
    """Expose the already-quantized delegate values for one logical state slot."""
    slot = row["slot"]
    encoder, decoder = quantize[plan.inputs[slot]], dequantize[plan.outputs[slot]]
    for call in [encoder, decoder]:
        assert plan.values[call.args[1]].val.double_val == row["scale"]
        assert plan.values[call.args[2]].val.int_val == row["zero_point"]
        assert [plan.values[call.args[i]].val.int_val for i in [3, 4, 5]] == [-128, 127, 1]
    plan.inputs[slot], plan.outputs[slot] = encoder.args[-1], decoder.args[0]
    for index in [plan.inputs[slot], plan.outputs[slot]]:
        tensor = plan.values[index].val
        assert tensor.scalar_type == schema.ScalarType.CHAR and tensor.sizes == row["shape"]
    return {id(encoder), id(decoder)}


def quantization_getters(parameters: list[dict]) -> list:
    """Emit standard primitive metadata methods required by MLEK's INT8 tensors."""
    methods = []
    for row in parameters:
        for kind in ["input", "output"]:
            for suffix, value in [("scale", row["scale"]), ("zp", row["zero_point"])]:
                _, tree = ex_pytree.tree_flatten(value)
                methods.append(schema.ExecutionPlan(
                    name=f'forward_{kind}{row["slot"]}_{suffix}',
                    container_meta_type=schema.ContainerMetadata("", tree.to_str()),
                    values=[schema.EValue(schema.Double(value) if suffix == "scale"
                                          else schema.Int(value))],
                    inputs=[], outputs=[0], chains=[schema.Chain([], [], [], None)],
                    operators=[], delegates=[], non_const_buffer_sizes=[0]))
    return methods


def export_host(source: Path, target: Path, parameters: list[dict]) -> dict:
    """Remove precisely 73 state Q and 73 state DQ calls, preserving delegate arguments."""
    original = source.read_bytes()
    pte = _deserialize_pte_binary(original)
    assert len(pte.program.execution_plan) == 1
    plan = pte.program.execution_plan[0]
    streams = [hashlib.sha256(entry.data).hexdigest()
               for entry in pte.program.backend_delegate_data]
    quantize = calls_by_tensor(plan, "quantized_decomposed::quantize_per_tensor", 0)
    dequantize = calls_by_tensor(plan, "quantized_decomposed::dequantize_per_tensor", -1)
    assert len(quantize) == len(dequantize) == 74
    removed = set()
    for row in parameters:
        removed.update(rewrite_state(plan, row, quantize, dequantize))
    assert len(removed) == 146
    for chain in plan.chains:
        assert chain.stacktrace is None
        chain.instructions = [instruction for instruction in chain.instructions
                              if id(instruction.instr_args) not in removed]
    counts = Counter(type(instruction.instr_args).__name__
                     for chain in plan.chains for instruction in chain.instructions)
    assert counts == {"KernelCall": 2, "DelegateCall": 2}
    compact_memory(plan)
    pte.program.execution_plan.extend(quantization_getters(parameters))
    assert streams == [hashlib.sha256(entry.data).hexdigest()
                       for entry in pte.program.backend_delegate_data]
    target.write_bytes(bytes(serialize_pte_binary(pte)))
    return {"source_pte_sha256": hashlib.sha256(original).hexdigest(),
            "pte_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "delegate_stream_sha256": streams, "delegate_streams_unchanged": True,
            "quantize_calls": 1, "dequantize_calls": 1, "delegates": 2,
            "external_inputs_are_memory_planned": False,
            "planned_buffer_bytes": plan.non_const_buffer_sizes[1],
            "constant_quantization_methods": 292,
            "remaining_tensor_allocations_are_disjoint": True}
