# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Export INT8 cache boundaries while preserving the calibrated Zipformer computation."""
from pathlib import Path

import torch


def bypass_state(placeholder: object, decoder: object, output: object, slot: int) -> dict:
    """Check one matched Q/DQ pair and bypass its input quantization."""
    quantize = torch.ops.quantized_decomposed.quantize_per_tensor.default
    dequantize = torch.ops.quantized_decomposed.dequantize_per_tensor.default
    domain = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8)
    users = list(placeholder.users)
    assert len(users) == 1 and users[0].target == quantize, slot
    encoder = users[0]
    assert decoder.target == dequantize and set(decoder.users) == {output}, slot
    assert encoder.args[1:] == decoder.args[1:], slot
    scale, zero_point, minimum, maximum, dtype = encoder.args[1:]
    assert dtype == torch.int8 and (minimum, maximum) == (-128, 127)
    restored = quantize(dequantize(domain, *encoder.args[1:]), *encoder.args[1:])
    assert torch.equal(domain, restored), "State Q(DQ(q)) is not an identity"
    encoder.replace_all_uses_with(placeholder)
    placeholder.graph.erase_node(encoder)
    return {"slot": slot, "shape": list(placeholder.meta["val"].shape),
            "scale": scale, "zero_point": zero_point}


def export_states(source: Path, target: Path) -> list[dict]:
    """Bypass matched state Q/DQ pairs and re-export with fresh input metadata."""
    program = torch.export.load(source)
    module = program.module(check_guards=False)
    graph = module.graph
    inputs = [node for node in graph.nodes if node.op == "placeholder"]
    output = next(node for node in graph.nodes if node.op == "output")
    outputs = list(output.args[0])
    assert len(inputs) == len(outputs) == 75
    example = list(program.example_inputs[0])
    parameters = []
    for slot in range(1, 74):
        row = bypass_state(inputs[slot], outputs[slot], output, slot)
        parameters.append(row)
        outputs[slot] = outputs[slot].args[0]
        example[slot] = torch.full(row["shape"], row["zero_point"], dtype=torch.int8)
    output.args = (tuple(outputs),)
    graph.eliminate_dead_code()
    graph.set_codegen(torch.fx.graph.CodeGen())
    # A fresh GraphModule has no old ExportedProgram input-constraint hooks.
    fresh = torch.fx.GraphModule(module, graph)
    fresh.recompile()
    exported = torch.export.export(fresh, tuple(example), strict=False)
    torch.export.save(exported, target)
    return parameters


def compare_clip(old: object, new: object, features: list, parameters: list[dict]) -> dict:
    """Compare all outputs across independently evolving old/new state trajectories."""
    old_states = tuple(torch.zeros(row["shape"]) for row in parameters) + (
        torch.zeros(1, dtype=torch.int32),)
    new_states = tuple(torch.full(row["shape"], row["zero_point"], dtype=torch.int8)
                       for row in parameters) + (torch.zeros(1, dtype=torch.int32),)
    paths = []
    for index, block in enumerate(features):
        before, after = old(block, *old_states), new(block, *new_states)
        assert torch.equal(before[0], after[0]), f"Logits differ at chunk {index}"
        for row in parameters:
            actual = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
                after[row["slot"]], row["scale"], row["zero_point"], -128, 127, torch.int8)
            assert torch.equal(before[row["slot"]], actual), (index, row["slot"])
        assert torch.equal(before[74], after[74]) and int(after[74].item()) == (index + 1) * 32
        old_states, new_states = tuple(before[1:]), tuple(after[1:])
        paths.append(after[0].argmax(-1).flatten().tolist())
    return {"chunks": len(paths), "all_outputs_bit_exact_after_state_dequantize": True,
            "argmax_paths": paths}
