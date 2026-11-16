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


def count_context(descriptors: dict, context: int) -> tuple[int, int]:
    """Count KV inputs and outputs with the configured context.

    :param descriptors:  Vela input and output descriptors.
    :param context:      Context length the firmware is compiled for.
    :returns:            Number of KV inputs and outputs in this delegate.
    """
    shape = (1, 8, context, 64)
    token_shape = (1, 8, 1, 64)
    inputs = [item for item in descriptors["inputs"] if tuple(item[:6])[-4:] == shape]
    outputs = [item for item in descriptors["outputs"]
               if tuple(item[:6])[-4:] in (shape, token_shape)]
    return len(inputs), len(outputs)


def validate(model_path: Path, context: int | None = None):
    """Reject mismatched delegate argument counts, shapes, and element sizes.

    :param model_path:  Serialized TinyStories PTE.
    :param context:     Optional context length required by the build.
    :raises ValueError: If the delegate interface does not match Vela metadata.
    """
    program = _load_pte_program(model_path.read_bytes())
    if program is None:
        raise ValueError("Unable to deserialize TinyStories PTE")
    widths = {"BOOL": 1, "BYTE": 1, "CHAR": 1, "SHORT": 2,
              "INT": 4, "LONG": 8, "HALF": 2, "FLOAT": 4}
    calls = 0
    kv_inputs = 0
    kv_outputs = 0
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
                if context is not None:
                    inputs, outputs = count_context(descriptors, context)
                    kv_inputs += inputs
                    kv_outputs += outputs
                ordered = descriptors["inputs"] + descriptors["outputs"]
                if len(call.args) < len(ordered):
                    raise ValueError(
                        f"Delegate has {len(call.args)} tensor arguments, but Vela has "
                        f"{len(descriptors['inputs'])} inputs + "
                        f"{len(descriptors['outputs'])} outputs"
                    )
                input_bytes = {
                    math.prod(tuple(value or 1 for value in descriptor[:6])) * descriptor[6]
                    for descriptor in descriptors["inputs"]
                }
                used = [False] * len(ordered)
                aliases = 0
                for value_index in call.args:
                    tensor = plan.values[value_index].val
                    shape = tuple(tensor.sizes)
                    tensor_bytes = math.prod(shape) * widths[tensor.scalar_type.name]
                    match = None
                    for index, descriptor in enumerate(ordered):
                        if used[index]:
                            continue
                        vela_shape = tuple(value or 1 for value in descriptor[:6])
                        vela_bytes = math.prod(vela_shape) * descriptor[6]
                        if tensor_bytes == vela_bytes:
                            match = (index, descriptor, vela_shape)
                            break
                    if match is None:
                        if tensor_bytes not in input_bytes:
                            raise ValueError(
                                f"Delegate tensor {shape}/{tensor_bytes} bytes has no "
                                f"matching Vela descriptor and aliases no input")
                        aliases += 1
                        continue
                    index, descriptor, vela_shape = match
                    used[index] = True
                    if shape != vela_shape[-len(shape):]:
                        print(f"TinyStories delegate tensor {index}: shape-only mismatch "
                              f"{shape} vs {vela_shape}, bytes match")
                if not all(used):
                    raise ValueError("Not all Vela descriptors were matched by ET arguments")
                if aliases:
                    print(f"TinyStories delegate I/O: {aliases} in-place output alias(es)")
                calls += 1
                print(f"TinyStories delegate I/O PASS: {len(descriptors['inputs'])} "
                      f"inputs, {len(descriptors['outputs'])} outputs")
    if calls < 1:
        raise ValueError("No Ethos-U delegate call found")
    if context is not None:
        if calls == 1 and (kv_inputs != 16 or kv_outputs != 16) and not (kv_outputs == 0 and aliases >= 16):
            raise ValueError(
                f"Expected 16 KV inputs and 16 KV outputs shaped {(1, 8, context, 64)}, "
                f"found {kv_inputs} and {kv_outputs}")
        if calls > 1 and (kv_inputs == 0 or kv_outputs == 0):
            raise ValueError(
                f"Expected KV inputs and outputs shaped {(1, 8, context, 64)}, "
                f"found {kv_inputs} and {kv_outputs} across {calls} delegates")
    print(f"TinyStories delegate I/O total: {calls} calls, "
          f"{kv_inputs} KV inputs, {kv_outputs} KV outputs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--context", type=int)
    args = parser.parse_args()
    validate(args.model, args.context)
