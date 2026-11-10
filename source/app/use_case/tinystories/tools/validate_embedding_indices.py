# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Check the serialized TinyStories CPU embedding index contract."""

import argparse
from pathlib import Path

from mlek_tools.validation.pte_ops_dump import _load_pte_program


def validate(model_path: Path):
    """Reject a PTE whose CPU embedding receives non-Long indices.

    :param model_path:  Serialized ExecuTorch model to inspect.
    :raises ValueError: If the input or embedding index contract is invalid.
    """
    program = _load_pte_program(model_path.read_bytes())
    if program is None:
        raise ValueError("Unable to inspect TinyStories PTE")
    plan = next(plan for plan in program.execution_plan if plan.name == "forward")
    input_types = [plan.values[index].val.scalar_type.name for index in plan.inputs]
    if input_types != ["LONG", "INT"]:
        raise ValueError(f"Expected Long tokens and Int positions, got {input_types}")
    count = 0
    for chain in plan.chains:
        for instruction in chain.instructions:
            call = instruction.instr_args
            if not hasattr(call, "op_index"):
                continue
            operator = plan.operators[call.op_index]
            if operator.name != "quantized_decomposed::embedding_byte":
                continue
            indices = plan.values[call.args[5]].val
            if indices.scalar_type.name != "LONG":
                raise ValueError(
                    "CPU embedding requires Long indices, got "
                    f"{indices.scalar_type.name} at value {call.args[5]}"
                )
            count += 1
    if count != 1:
        raise ValueError(f"Expected one quantized embedding call, found {count}")
    print("TinyStories PTE contract PASS: Long tokens -> Long CPU embedding indices")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    validate(parser.parse_args().model)
