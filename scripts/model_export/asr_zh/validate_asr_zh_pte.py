# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Validate Zipformer delegate argument order/types against embedded Vela I/O metadata."""
import argparse
import hashlib
from importlib import import_module
import json
import math
from pathlib import Path
import struct

from executorch.exir._serialize import _deserialize_pte_binary


def blocks(blob: bytes) -> dict:
    """Read bounded Vela stream blocks without interpreting command or weight bytes."""
    result = {}
    offset = 0
    while offset + 32 <= len(blob):
        name = blob[offset:offset + 16].split(b"\0", 1)[0].decode("ascii")
        size = struct.unpack_from("<I", blob, offset + 16)[0]
        assert offset + 32 + size <= len(blob), "Truncated Vela stream"
        result[name] = blob[offset + 32:offset + 32 + size]
        offset += 32 + ((size + 15) // 16) * 16
        if name == "vela_end_stream":
            break
    return result


def check_io(values: list, indices: list, data: bytes, output: bool) -> list:
    """Check every delegate tensor's element width and capacity in serialized argument order."""
    count = struct.unpack_from("<i", data)[0]
    assert count == len(indices), "Delegate argument count does not match Vela"
    result = []
    widths = {"BYTE": 1, "CHAR": 1, "SHORT": 2, "INT": 4, "BOOL": 1}
    for slot, index in enumerate(indices):
        tensor = values[index].val
        row = struct.unpack_from("<9i", data, 4 + slot * 36)
        assert widths.get(tensor.scalar_type.name) == row[6], (
            f"{'output' if output else 'input'} {slot}: PTE dtype={tensor.scalar_type.name} "
            f"shape={tensor.sizes}, Vela elem_size={row[6]} shape={row[:6]}")
        size = math.prod(max(1, dimension) for dimension in row[:6])
        expected = math.prod(tensor.sizes)
        assert size >= expected if output else size == expected, "Vela tensor capacity mismatch"
        result.append({"slot": slot, "dtype": tensor.scalar_type.name,
                       "pte_shape": tensor.sizes, "vela_shape": row[:6], "elem_size": row[6]})
    return result


def validate_model(path: Path) -> dict:
    """Return validation evidence, raising on an incompatible or reordered delegate interface."""
    schema = import_module("executorch.exir.schema")
    data = path.read_bytes()
    program = _deserialize_pte_binary(data).program
    plans = [plan for plan in program.execution_plan if plan.name == "forward"]
    assert len(plans) == 1 and program.execution_plan[0].name == "forward"
    plan = plans[0]
    for metadata in program.execution_plan[1:]:
        assert not metadata.inputs and not metadata.delegates and not metadata.operators
        assert len(metadata.outputs) == 1
        assert all(not chain.instructions for chain in metadata.chains)
    assert len(plan.inputs) == len(plan.outputs) == 75
    delegates = []
    for chain in plan.chains:
        for instruction in chain.instructions:
            call = instruction.instr_args
            if not isinstance(call, schema.DelegateCall):
                continue
            delegate = plan.delegates[call.delegate_index]
            assert delegate.id == "EthosUBackend"
            stream = blocks(program.backend_delegate_data[delegate.processed.index].data)
            count = struct.unpack_from("<i", stream["inputs"])[0]
            delegates.append({
                "index": call.delegate_index,
                "inputs": check_io(plan.values, call.args[:count], stream["inputs"], False),
                "outputs": check_io(plan.values, call.args[count:], stream["outputs"], True),
            })
    assert len(delegates) == 2
    return {"pte_sha256": hashlib.sha256(data).hexdigest(),
            "delegate_io_checked": True, "delegates": delegates}


def main():
    """Validate one PTE; failure is an error rather than an advisory warning."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pte", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_model(args.pte)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pte_sha256": result["pte_sha256"], "delegate_io_checked": True}))


if __name__ == "__main__":
    main()
