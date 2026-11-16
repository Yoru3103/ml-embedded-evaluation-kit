# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Check full-range cache transition equivalence and actual PTE buffer execution."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from executorch.exir._serialize._program import deserialize_pte_binary
from executorch.exir.schema import DelegateCall, KernelCall, Tensor

DTYPES = {'FLOAT': '<f4', 'SHORT': '<i2', 'CHAR': 'i1', 'BYTE': 'u1',
          'INT': '<i4', 'LONG': '<i8', 'BOOL': '?'}


class CpuPlan:
    """Execute surrounding PTE CPU operators on their actual planned buffer views."""

    def __init__(self, path: Path):
        """Load constants, mutable state, and byte-exact planned memory views.

        :param path:  Serialized TinyStories PTE.
        """
        self.pte = deserialize_pte_binary(path.read_bytes())
        self.plan = next(p for p in self.pte.program.execution_plan if p.name == 'forward')
        self.memory = [bytearray(size) for size in self.plan.non_const_buffer_sizes]
        self.values = {}
        for index, item in enumerate(self.plan.values):
            value = item.val
            if not isinstance(value, Tensor):
                continue
            if value.scalar_type.name not in DTYPES:
                raise ValueError(f'Unexpected dtype {value.scalar_type}')
            if value.dim_order != list(range(len(value.sizes))):
                raise ValueError(f'Unexpected non-contiguous dim order at value {index}')
            dtype = np.dtype(DTYPES[value.scalar_type.name])
            shape = tuple(value.sizes)
            if value.allocation_info is None:
                if value.data_buffer_idx:
                    storage = self.pte.program.constant_buffer[value.data_buffer_idx].storage
                    array = np.frombuffer(storage, dtype=dtype,
                                          count=math.prod(shape),
                                          offset=value.storage_offset * dtype.itemsize)
                    array = array.reshape(shape)
                else:
                    array = np.zeros(shape, dtype=dtype)
            else:
                allocation = value.allocation_info
                array = np.ndarray(shape, dtype=dtype,
                                   buffer=self.memory[allocation.memory_id],
                                   offset=allocation.memory_offset +
                                   value.storage_offset * dtype.itemsize)
                if value.data_buffer_idx:
                    storage = self.pte.mutable_data[value.data_buffer_idx].storage
                    initial = np.frombuffer(storage, dtype=dtype, count=math.prod(shape))
                    np.copyto(array, initial.reshape(shape))
            self.values[index] = array
        self.instructions = self.plan.chains[0].instructions
        self.cursor = 0

    def value(self, index: int) -> object:
        """Resolve serialized scalars, tensor lists and tensor views.

        :param index:  PTE value index.
        :returns:      NumPy view or Python scalar/list.
        """
        if index in self.values:
            return self.values[index]
        value = self.plan.values[index].val
        for name in ('int_val', 'double_val', 'bool_val'):
            if hasattr(value, name):
                return getattr(value, name)
        if hasattr(value, 'items'):
            if type(value).__name__ in ('TensorList', 'OptionalTensorList'):
                return [None if item < 0 else self.value(item) for item in value.items]
            return value.items
        if type(value).__name__ == 'Null':
            return None
        raise ValueError(f'Unsupported scalar {value}')

    def begin(self, token: int, position: int):
        """Set one token and position for the next model invocation.

        :param token:     Token ID.
        :param position:  Absolute context position.
        """
        self.values[self.plan.inputs[0]][...] = token
        self.values[self.plan.inputs[1]][...] = position
        self.cursor = 0

    def execute_to_delegate(self) -> list[np.ndarray]:
        """Execute CPU instructions preceding the NPU invocation.

        :returns: Snapshots of every delegate input.
        """
        while self.cursor < len(self.instructions):
            call = self.instructions[self.cursor].instr_args
            if isinstance(call, DelegateCall):
                return [self.value(index).copy() for index in call.args[:45]]
            self.execute_kernel(call)
            self.cursor += 1
        raise ValueError('No delegate found')

    def finish(self, outputs: list[np.ndarray]):
        """Copy delegate outputs and execute CPU state update instructions.

        :param outputs:  Shared deterministic delegate results for both graphs.
        """
        call = self.instructions[self.cursor].instr_args
        for index, data in zip(call.args[45:], outputs):
            np.copyto(self.value(index), data)
        self.cursor += 1
        while self.cursor < len(self.instructions):
            self.execute_kernel(self.instructions[self.cursor].instr_args)
            self.cursor += 1

    def execute_kernel(self, call: KernelCall):
        """Apply the supported CPU op with portable-kernel float32 arithmetic.

        :param call:  Serialized kernel call.
        """
        operator = self.plan.operators[call.op_index]
        name = operator.name
        args = [self.value(index) for index in call.args]
        if name == 'quantized_decomposed::quantize_per_tensor':
            inverse = np.float32(1) / np.float32(args[1])
            result = np.clip(np.rint(args[0] * inverse) + args[2], args[3], args[4])
        elif name == 'quantized_decomposed::dequantize_per_tensor':
            result = (args[0].astype(np.float32) - args[2]) * np.float32(args[1])
        elif name == 'quantized_decomposed::embedding_byte':
            require_zero = args[2] is None
            if not require_zero:
                raise ValueError('Unexpected embedding zero points')
            ids = args[5]
            weights = args[0][ids].astype(np.float32)
            scales = args[1][ids]
            result = weights * scales.reshape(*ids.shape, -1)
        elif name == 'aten::index':
            indices = tuple(slice(None) if index is None else index for index in args[1])
            result = args[0][indices]
        elif name == 'aten::copy_':
            result = args[1]
        elif name == 'aten::clamp':
            result = np.clip(args[0], args[1], args[2])
        else:
            raise ValueError(f'Unexpected kernel {name}')
        np.copyto(args[-1], np.asarray(result).astype(args[-1].dtype))


def check_memory_aliases(candidate: CpuPlan):
    """Reject unintended write aliases into the sixteen persistent cache tensors.

    :param candidate:  Candidate CPU execution plan and memory views.
    """
    cache_ids = [index for index, value in enumerate(candidate.plan.values)
                 if isinstance(value.val, Tensor) and value.val.allocation_info is not None
                 and value.val.data_buffer_idx > 0]
    assert len(cache_ids) == 16
    caches = [candidate.value(index) for index in cache_ids]
    for instruction in candidate.instructions:
        call = instruction.instr_args
        if isinstance(call, KernelCall):
            if candidate.plan.operators[call.op_index].name == 'aten::clamp':
                continue
            target = candidate.value(call.args[-1])
            assert not any(np.shares_memory(target, cache) for cache in caches)


def compare_inputs(reference: CpuPlan, candidate: CpuPlan, token: int,
                   position: int) -> list[np.ndarray]:
    """Compare all NPU inputs after actual CPU instruction execution.

    :param reference:   Original FP32-state CPU plan.
    :param candidate:   INT16-state CPU plan.
    :param token:       Current token ID.
    :param position:    Absolute context position.
    :returns:           Original delegate input snapshots.
    """
    reference.begin(token, position)
    candidate.begin(token, position)
    original_inputs = reference.execute_to_delegate()
    new_inputs = candidate.execute_to_delegate()
    assert len(original_inputs) == len(new_inputs) == 45
    for index, (original, new) in enumerate(zip(original_inputs, new_inputs)):
        assert original.dtype == new.dtype and np.array_equal(original, new), (position, index)
    return original_inputs


def simulated_outputs(inputs: list[np.ndarray], position: int, rng: np.random.Generator,
                      cache_shape: tuple[int, ...]) -> list[np.ndarray]:
    """Update cache rows with arbitrary INT16 results, including signed-min values.

    :param inputs:       Current delegate input snapshots.
    :param position:     Cache position to update.
    :param rng:          Deterministic random source.
    :param cache_shape:  Exported KV tensor shape.
    :returns:            Sixteen cache outputs and one logits output.
    """
    states = [value for value in inputs if tuple(value.shape) == cache_shape]
    assert len(states) == 16
    outputs = []
    for state in states:
        result = state.copy()
        row = (1, cache_shape[1], cache_shape[3])
        result[:, :, position, :] = rng.integers(-32768, 32768, row, dtype=np.int16)
        result[:, :, position, 0] = -32768
        outputs.append(result)
    outputs.append(rng.integers(-32768, 32768, (1, 32000), dtype=np.int16))
    return outputs


def check_negative_control(baseline_path: Path, candidate_path: Path,
                           outputs: list[np.ndarray]):
    """Require the checker to detect a candidate missing its integer saturation.

    :param baseline_path:   Original PTE.
    :param candidate_path:  Candidate PTE.
    :param outputs:         Shared simulated NPU outputs containing signed-min values.
    """
    reference, unchecked = CpuPlan(baseline_path), CpuPlan(candidate_path)
    unchecked.instructions = [instruction for instruction in unchecked.instructions
        if not (isinstance(instruction.instr_args, KernelCall) and
                unchecked.plan.operators[instruction.instr_args.op_index].name == 'aten::clamp')]
    for model in (reference, unchecked):
        model.begin(1, 0)
        model.execute_to_delegate()
        model.finish(outputs)
        model.begin(727, 1)
    original_inputs = reference.execute_to_delegate()
    unchecked_inputs = unchecked.execute_to_delegate()
    assert any(not np.array_equal(original, new)
               for original, new in zip(original_inputs, unchecked_inputs))


def main():
    """Compare every NPU input across 64 stateful calls with adversarial outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--cache-shape', default='1,8,64,64')
    args = parser.parse_args()
    cache_shape = tuple(int(value) for value in args.cache_shape.split(','))
    baseline, candidate = CpuPlan(args.baseline), CpuPlan(args.candidate)
    check_memory_aliases(candidate)
    rng = np.random.default_rng(42)
    input_checks = 0
    context = cache_shape[2]
    for position in range(context):
        token = 1 if position == 0 else int(rng.integers(3, 32000))
        inputs = compare_inputs(baseline, candidate, token, position)
        input_checks += len(inputs)
        outputs = simulated_outputs(inputs, position, rng, cache_shape)
        baseline.finish(outputs)
        candidate.finish(outputs)
        assert np.array_equal(baseline.value(baseline.plan.outputs[0]),
                              candidate.value(candidate.plan.outputs[0]))
    check_negative_control(args.baseline, args.candidate, outputs)
    print(json.dumps({'delegate_input_comparisons': input_checks,
                      'context_positions': context, 'mismatches': 0,
                      'initial_state_preserved': True,
                      'planned_memory_alias_check': 'PASS',
                      'signed_min_boundary_exercised': True,
                      'omitted_clamp_negative_control': 'PASS',
                      'note': 'CPU PTE execution and identical simulated delegate outputs; '
                              'not an NPU generation or speed measurement'}, indent=2))


if __name__ == '__main__':
    main()
