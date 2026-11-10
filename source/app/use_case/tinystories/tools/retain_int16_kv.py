# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Retain INT16 KV state around an unchanged Ethos-U delegate in a TinyStories PTE."""

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from executorch.exir._serialize._program import (
    PTEFile, deserialize_pte_binary, serialize_pte_binary,
)
from executorch.exir.schema import (
    DelegateCall, ExecutionPlan, Instruction, KernelCall, Operator, ScalarType, Tensor,
)

CACHE_SHAPE = [1, 8, 64, 64]


def require(condition: bool, message: str):
    """Reject a source graph outside the supported transformation contract.

    :param condition:  Required invariant.
    :param message:    Error description.
    :raises ValueError: If the invariant is false.
    """
    if not condition:
        raise ValueError(message)


def cache_paths(plan: ExecutionPlan) -> list[dict]:
    """Locate each cache's quantize, dequantize and state-copy instructions.

    :param plan:  Original straight-line forward plan.
    :returns:     Sixteen independently validated cache path descriptions.
    """
    require(len(plan.chains) == 1, 'Expected one straight-line chain')
    calls = {'quantize': {}, 'dequantize': {}, 'copy': {}}
    for index, instruction in enumerate(plan.chains[0].instructions):
        call = instruction.instr_args
        require(isinstance(call, (KernelCall, DelegateCall)), 'Unsupported control flow')
        if not isinstance(call, KernelCall):
            continue
        tensor = plan.values[call.args[0]].val
        if not isinstance(tensor, Tensor) or tensor.sizes != CACHE_SHAPE:
            continue
        name = plan.operators[call.op_index].name
        if name == 'quantized_decomposed::quantize_per_tensor':
            calls['quantize'][call.args[0]] = (index, call)
        elif name == 'quantized_decomposed::dequantize_per_tensor':
            calls['dequantize'][call.args[-1]] = (index, call)
        elif name == 'aten::copy_':
            calls['copy'][call.args[0]] = (index, call)
    require(all(len(group) == 16 for group in calls.values()), 'Expected sixteen cache paths')
    result = []
    for state, quantize in calls['quantize'].items():
        state_copy = calls['copy'][state]
        result.append({'state': state, 'quantize': quantize, 'copy': state_copy,
                       'dequantize': calls['dequantize'][state_copy[1].args[1]]})
    return result


def validate_path(plan: ExecutionPlan, path: dict, delegate: tuple):
    """Check a cache path has matching quantization parameters and exclusive uses.

    :param plan:      Original forward plan.
    :param path:      One cache path.
    :param delegate:  Instruction index and sole delegate call.
    """
    quantize = path['quantize'][1]
    dequantize = path['dequantize'][1]
    state = plan.values[path['state']].val
    input_tensor = plan.values[quantize.args[-1]].val
    output_tensor = plan.values[dequantize.args[0]].val
    require(path['quantize'][0] < delegate[0] < path['dequantize'][0] < path['copy'][0],
            'Unexpected cache instruction order')
    require([plan.values[index].val for index in quantize.args[1:6]] ==
            [plan.values[index].val for index in dequantize.args[1:6]],
            'Input and output cache quantization parameters differ')
    require(state.scalar_type == ScalarType.FLOAT and
            input_tensor.scalar_type == output_tensor.scalar_type == ScalarType.SHORT,
            'Unexpected cache dtypes')
    require(input_tensor.sizes == output_tensor.sizes == state.sizes == CACHE_SHAPE and
            input_tensor.dim_order == output_tensor.dim_order == state.dim_order,
            'Unexpected cache layout')
    require(state.allocation_info is not None and state.data_buffer_idx > 0 and
            state.storage_offset == 0, 'Cache must be initialized planned mutable state')
    require(delegate[1].args.count(quantize.args[-1]) ==
            delegate[1].args.count(dequantize.args[0]) == 1, 'Unexpected delegate cache uses')
    excluded = {path[key][0] for key in ('quantize', 'dequantize', 'copy')} | {delegate[0]}
    cache_values = (path['state'], quantize.args[-1], dequantize.args[0],
                    path['copy'][1].args[1])
    for index, instruction in enumerate(plan.chains[0].instructions):
        if index not in excluded:
            require(not any(value in instruction.instr_args.args for value in cache_values),
                    'Cache path has an additional consumer')


def convert_state(pte: PTEFile, plan: ExecutionPlan, path: dict) -> dict:
    """Quantize initial state and prove its later Q/DQ transition equals integer clamp.

    :param pte:   PTE including mutable initialization buffers.
    :param plan:  Original forward plan.
    :param path:  Cache path with matching quantization parameters.
    :returns:     Numeric validation record for this cache.
    """
    parameters = [plan.values[index].val for index in path['quantize'][1].args[1:5]]
    state = plan.values[path['state']].val
    require(parameters[0].double_val > 0 and np.isfinite(parameters[0].double_val) and
            parameters[1].int_val == 0 and parameters[2].int_val == -32767 and
            parameters[3].int_val == 32767, 'Unsupported quantization domain')
    inverse = np.float32(1) / np.float32(parameters[0].double_val)
    domain = np.arange(-32768, 32768, dtype=np.int32)
    restored = domain.astype(np.float32) * np.float32(parameters[0].double_val)
    roundtrip = np.clip(np.rint(restored * inverse), -32767, 32767).astype(np.int32)
    require(np.array_equal(roundtrip, np.clip(domain, -32767, 32767)),
            'Q(DQ(q)) is not equivalent to integer clamp')
    buffer = pte.mutable_data[state.data_buffer_idx]
    require(len(buffer.storage) == 131072, 'Unexpected initial cache byte length')
    initial = np.frombuffer(buffer.storage, dtype='<f4')
    require(np.isfinite(initial).all(), 'Non-finite initial cache data')
    require(sum(isinstance(value.val, Tensor) and value.val.allocation_info is not None and
                value.val.data_buffer_idx == state.data_buffer_idx
                for value in plan.values) == 1, 'Mutable buffer has multiple owners')
    buffer.storage = np.clip(np.rint(initial * inverse), -32767, 32767).astype('<i2').tobytes()
    state.scalar_type = ScalarType.SHORT
    return {'state_value': path['state'], 'buffer_index': state.data_buffer_idx,
            'scale': parameters[0].double_val, 'zero_point': 0, 'min': -32767, 'max': 32767,
            'allocation_offset_unchanged': state.allocation_info.memory_offset,
            'initial_int16_sha256': hashlib.sha256(buffer.storage).hexdigest(),
            'exhaustive_int16_values_checked': len(domain)}


def compact_operator_table(plan: ExecutionPlan):
    """Remove unused operator declarations and remap kernel indices.

    :param plan:  Rewritten forward plan.
    """
    used = sorted({instruction.instr_args.op_index
                   for instruction in plan.chains[0].instructions
                   if isinstance(instruction.instr_args, KernelCall)})
    remap = {old: new for new, old in enumerate(used)}
    plan.operators = [plan.operators[index] for index in used]
    for instruction in plan.chains[0].instructions:
        if isinstance(instruction.instr_args, KernelCall):
            instruction.instr_args.op_index = remap[instruction.instr_args.op_index]


def rewrite_paths(pte: PTEFile, plan: ExecutionPlan) -> dict:
    """Rewire delegate cache slots to persistent INT16 values and insert saturation.

    :param pte:   PTE including mutable initialization buffers.
    :param plan:  Original forward plan.
    :returns:     Cache records and instruction-change counts.
    """
    paths = cache_paths(plan)
    delegates = [(index, instruction.instr_args)
                 for index, instruction in enumerate(plan.chains[0].instructions)
                 if isinstance(instruction.instr_args, DelegateCall)]
    require(len(delegates) == 1, 'Expected one delegate')
    delegate = delegates[0]
    require(plan.delegates[delegate[1].delegate_index].id == 'EthosUBackend', 'Wrong delegate')
    require(pte.mutable_data is not None, 'Missing mutable initial cache data')
    changes = {'removed': set(), 'clamps': [], 'records': []}
    clamp_operator = len(plan.operators)
    plan.operators.append(Operator(name='aten::clamp', overload='out'))
    for path in paths:
        validate_path(plan, path, delegate)
        changes['records'].append(convert_state(pte, plan, path))
        replacements = (path['quantize'][1].args[-1], path['dequantize'][1].args[0])
        delegate[1].args[:] = [path['state'] if index in replacements else index
                              for index in delegate[1].args]
        changes['removed'].update(path[key][0] for key in ('quantize', 'dequantize', 'copy'))
        changes['clamps'].append(Instruction(instr_args=KernelCall(
            op_index=clamp_operator,
            args=[path['state'], *path['quantize'][1].args[3:5],
                  path['state'], path['state']])))
    instructions = []
    for index, instruction in enumerate(plan.chains[0].instructions):
        if index not in changes['removed']:
            instructions.append(instruction)
        if index == delegate[0]:
            instructions.extend(changes['clamps'])
    plan.chains[0].instructions = instructions
    plan.chains[0].stacktrace = None
    compact_operator_table(plan)
    return changes


def kernel_counts(plan: ExecutionPlan) -> dict[str, int]:
    """Count active CPU calls in a serialized forward plan.

    :param plan:  Rewritten forward plan.
    :returns:     Counts keyed by qualified operator name and overload.
    """
    counts = Counter()
    for instruction in plan.chains[0].instructions:
        call = instruction.instr_args
        if isinstance(call, KernelCall):
            operator = plan.operators[call.op_index]
            counts[operator.name + '.' + operator.overload] += 1
    return dict(counts)


def transform(source: Path, destination: Path, report_path: Path):
    """Create an independent INT16-state candidate while preserving weights and payload.

    :param source:       Original PTE to preserve.
    :param destination:  New candidate path; must not already exist.
    :param report_path:  JSON validation and transformation report.
    """
    require(source.resolve() != destination.resolve(), 'Source must be preserved')
    require(report_path.resolve() not in (source.resolve(), destination.resolve()),
            'Report path must differ from both PTE files')
    require(not destination.exists(), 'Candidate already exists; choose a new output path')
    original = source.read_bytes()
    pte = deserialize_pte_binary(original)
    plan = next(plan for plan in pte.program.execution_plan if plan.name == 'forward')
    preserved = {'constants': copy.deepcopy(pte.program.constant_buffer),
                 'payloads': copy.deepcopy(pte.program.backend_delegate_data),
                 'outputs': list(plan.outputs)}
    changes = rewrite_paths(pte, plan)
    require(pte.program.constant_buffer == preserved['constants'], 'Weights/constants changed')
    require(pte.program.backend_delegate_data == preserved['payloads'], 'NPU payload changed')
    require(plan.outputs == preserved['outputs'], 'Public output changed')
    candidate = bytes(serialize_pte_binary(pte))
    require(deserialize_pte_binary(candidate).program.backend_delegate_data ==
            preserved['payloads'], 'Delegate changed during serialization')
    report = {'source': str(source), 'source_bytes': len(original),
              'source_sha256': hashlib.sha256(original).hexdigest(),
              'candidate_bytes': len(candidate),
              'candidate_sha256': hashlib.sha256(candidate).hexdigest(),
              'payload_sha256_unchanged': [hashlib.sha256(blob.data).hexdigest()
                                         for blob in preserved['payloads']],
              'cache_count': len(changes['records']),
              'removed_instructions': len(changes['removed']),
              'added_integer_clamps': len(changes['clamps']), 'cpu_calls': kernel_counts(plan),
              'mutable_cache_bytes_before': 16 * 131072,
              'mutable_cache_bytes_after': 16 * 65536,
              'planned_buffers_preserved': plan.non_const_buffer_sizes,
              'layout_note': 'Preserve old planned allocation offsets and buffer sizes; '
                             'no planned-arena compaction in this candidate',
              'cache_parameters': changes['records']}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(candidate)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items()
                      if key != 'cache_parameters'}, indent=2))


def main():
    """Build a separate candidate and record all transformation invariants."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    transform(args.source, args.candidate, args.report)


if __name__ == '__main__':
    main()
