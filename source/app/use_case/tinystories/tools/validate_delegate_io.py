# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Validate the serialized TinyStories delegate tensor interface."""

import argparse
import math
import struct
from pathlib import Path

from mlek_tools.validation.pte_ops_dump import _load_pte_program


def read_io(payload: bytes) -> dict[str, list[tuple[int, ...]]]:
    """Read inline Vela input and output descriptors.

    :param payload:  Serialized Vela stream.
    :returns:       Input and output descriptor lists.
    :raises ValueError: If the stream is incomplete or uses external I/O blocks.
    """
    result = {}
    offset = 0
    while offset + 32 <= len(payload):
        name = payload[offset:offset + 16].rstrip(b"\0").decode("ascii")
        size = struct.unpack_from("<I", payload, offset + 16)[0]
        start = offset + 32
        if start + size > len(payload):
            raise ValueError(f"Truncated Vela block: {name}")
        if name in ("inputs", "outputs"):
            if payload[offset + 20] or size < 4:
                raise ValueError(f"Unsupported Vela I/O block: {name}")
            count = struct.unpack_from("<i", payload, start)[0]
            if count < 0 or size != 4 + count * 36:
                raise ValueError(f"Invalid Vela descriptor count: {name}")
            result[name] = [
                struct.unpack_from("<9i", payload, start + 4 + index * 36)
                for index in range(count)
            ]
        offset = start + (size + 15) // 16 * 16
        if name == "vela_end_stream":
            break
    if set(result) != {"inputs", "outputs"}:
        raise ValueError("Missing Vela I/O descriptors")
    return result


def validate(model_path: Path):
    """Reject mismatched delegate argument counts, shapes, and element sizes.

    :param model_path:  Serialized TinyStories PTE.
    :raises ValueError: If the delegate interface does not match Vela metadata.
    """
    program = _load_pte_program(model_path.read_bytes())
    if program is None:
        raise ValueError("Unable to deserialize TinyStories PTE")
    widths = {"BOOL": 1, "BYTE": 1, "CHAR": 1, "SHORT": 2,
              "INT": 4, "LONG": 8, "HALF": 2, "FLOAT": 4}
    calls = 0
    for plan in program.execution_plan:
        for chain in plan.chains:
            for instruction in chain.instructions:
                call = instruction.instr_args
                if not hasattr(call, "delegate_index"):
                    continue
                delegate = plan.delegates[call.delegate_index]
                if delegate.id != "EthosUBackend":
                    continue
                payload = bytes(program.backend_delegate_data[delegate.processed.index].data)
                descriptors = read_io(payload)
                ordered = descriptors["inputs"] + descriptors["outputs"]
                if len(call.args) != len(ordered):
                    raise ValueError(
                        f"Delegate has {len(call.args)} tensor arguments, but Vela has "
                        f"{len(descriptors['inputs'])} inputs + "
                        f"{len(descriptors['outputs'])} outputs"
                    )
                for index, (value_index, descriptor) in enumerate(zip(call.args, ordered)):
                    tensor = plan.values[value_index].val
                    shape = tuple(tensor.sizes)
                    vela_shape = tuple(value or 1 for value in descriptor[:6])
                    tensor_bytes = math.prod(shape) * widths[tensor.scalar_type.name]
                    vela_bytes = math.prod(vela_shape) * descriptor[6]
                    if tensor_bytes != vela_bytes or shape != vela_shape[-len(shape):]:
                        raise ValueError(
                            f"Delegate tensor {index}: {shape}/{tensor_bytes} bytes "
                            f"does not match Vela {vela_shape}/{vela_bytes} bytes"
                        )
                calls += 1
                print(f"TinyStories delegate I/O PASS: {len(descriptors['inputs'])} "
                      f"inputs, {len(descriptors['outputs'])} outputs")
    if calls != 1:
        raise ValueError(f"Expected one Ethos-U delegate call, found {calls}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    validate(parser.parse_args().model)
