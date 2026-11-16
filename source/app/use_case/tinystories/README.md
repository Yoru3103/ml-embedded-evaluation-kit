# TinyStories FPGA generation and timing

For the floor 1e-3 ctx128 milestone build and acceptance procedure, see the
[UART demo guide](DEMO.md#build-and-run-the-ctx128-milestone).

The PTE uses INT8 embedding weights, a W8A16 delegated body, an INT16 KV state
and context 128. `MainLoop.cc` consumes ten prompt tokens, compares every top1
with the matching host quantized PT2 reference, and generates at most sixteen
new tokens using greedy argmax. The first generated token is produced by the
last prompt call; a full run therefore has ten prompt calls and fifteen decode
calls. EOS is token 2. After generation, the Cortex-M decodes the prompt and
actual generated token IDs using the complete exported 32000-piece vocabulary
and prints TINYSTORIES_TEXT.

The DEMO build defaults to top-k / top-p sampling (`temperature=0.8`,
`top_p=0.9`, `top_k=64`) and keeps greedy available at run time. `!greedy` and
`!sample` switch the mode, `!seed N` selects the request seed base, and `!temp`,
`!topp`, `!topk` tune the distribution. The same seed reproduces the same story;
the seed is printed in DONE. TEST remains greedy so the fixed reference tokens
stay a bit-exact regression signal.

The existing board smoke log matches 7/10 host prompt top1 predictions (positions
5, 6 and 7 differ). The final prompt prediction matches token 727. The reference
is host PT2, not bit-exact Vela/NPU output. Prompt and generation matches are
reported separately so those differences remain visible. The current 128-context
reference is generated from
`npu_full_16a8w_embedding8_ctx128/quantized.pt2`.

Each `TINYSTORIES_TIME` record reports:

- `npu_total_cycles`: hardware NPU TOTAL, including NPU stalls, enabled around
  driver execution by the existing HAL hooks. This is not NPU ACTIVE or CPU
  driver submission time.
- `invoke_cycles`: CPU-clock elapsed time around `RunInference()`, including
  CPU kernels, delegate I/O, driver calls and waiting for the NPU.
- `e2e_cycles`: CPU-clock elapsed time for setting token/position, inference,
  checking finite logits and greedy argmax. Model initialization, PMU setup,
  UART logging and separately measured board-side text decoding are excluded.

All counters are reset per call. `prompt_sum` and `decode_sum` sum measured
steps, excluding time spent printing between steps. Prompt sum is the measured
compute time to the first generated token after model initialization. The
clock record prints both conversion frequencies. NPU milliseconds assume the
configured `tinystories_NPU_CLOCK_HZ` (default 50000000 Hz for this FPGA build);
CPU milliseconds use `get_mps4_core_clock()`, including its logged fallback.
Raw cycle counts remain available if either frequency needs correction.

Build with the current corrected PTE, `CPU_PROFILE_ENABLED=ON`,
`ETHOS_U_NPU_CACHE_SIZE=1258304`, display OFF, and the existing safe DDR layout.
The default model path in `usecase.cmake` points to the INT16-context-128 PTE.
The method pool is 7 MiB, the temporary pool is 2.25 MiB at `0x73DC0000`, and
the static DDR total is 66,831,827 / 67,108,864 bytes. After changing the PTE or
the context, configure a fresh build directory (or pass an explicit
`-Dtinystories_MODEL_PATH` and `-Dtinystories_CONTEXT`); CMake keeps the old
model path in an existing cache.

Build with `-Dtinystories_LOGIT_DUMP=ON` to print
`TINYSTORIES_LOGITS position=... id0..id4` for each inference. The matching host
dump is `tools/host_logits.py`; both use the fixed TEST prompt so the first
positions can be compared directly. The build helpers expose it as `-LogitDump`:

```powershell
# DEMO (default)
.\source\app\use_case\tinystories\tools\build_fpga_demo.ps1
.\source\app\use_case\tinystories\tools\build_fvp_demo.ps1

# TEST (fixed 10+16 regression; optional logits dump)
.\source\app\use_case\tinystories\tools\build_test.ps1 -Target FVP -LogitDump
.\source\app\use_case\tinystories\tools\build_test.ps1 -Target FPGA
```

Both helpers pin the 128-context PTE and accept `-ModelPath`, `-BuildDirectory`
and `-Jobs`; the default TEST directories are `build/tinystories42m/test/fpga`
and `.../test/fvp`. Pinning the path avoids a stale CMake cache after a model
change.

Generate host reference data in the WSL export environment:

```sh
python source/app/use_case/tinystories/tools/prepare_reference.py \
  resources_downloaded/tinystories42m/npu_full_16a8w_embedding8_dedicated_protected/quantized.pt2 \
  resources_downloaded/tinystories42m/tokenizer.model \
  resources_downloaded/tinystories42m/board_reference.json \
  --header source/app/use_case/tinystories/src/TinyStoriesReference.hpp
```

Save one complete UART run to a UTF-8 log. Decode the actual board token IDs
and compare against the host reference in the same WSL environment:

```sh
python source/app/use_case/tinystories/tools/decode_board_log.py \
  resources_downloaded/tinystories42m/board_generation.log \
  resources_downloaded/tinystories42m/board_reference.json \
  resources_downloaded/tinystories42m/tokenizer.model \
  --output resources_downloaded/tinystories42m/board_generation_report.json
```

The decoder rejects missing, duplicate or unordered prompt/generation records.
An early EOS is accepted as a complete shorter generation and is still compared
against the reference. It prints the decoded board text and timing records.

## Board text decoding and detailed profiling

`TokenDecoder.cc` implements SentencePiece surface decoding, the initial dummy
space rule, control-token removal, byte fallback, and UTF-8 replacement. It
uses the actual generated IDs, including tokens that differ from the host
reference. It does not run the neural model or encode arbitrary input text.
`TINYSTORIES_TEXT_TIME` measures only text decoding on the CPU; UART printing
is excluded. The output buffer is 4096 bytes in SRAM. Invalid IDs, invalid table
bounds and insufficient output capacity return failure.

Export the complete vocabulary once in the WSL model environment:

```sh
python source/app/use_case/tinystories/tools/export_tokenizer.py \
  resources_downloaded/tinystories42m/tokenizer.model \
  resources_downloaded/tinystories42m/tokenizer_surfaces.bin
```

The table is 336586 bytes and lives in DDR after the method pool. CMake includes
it in the static budget and the linker asserts it ends before the temporary
pool. The current map is `0x73D95780..0x73DE7A4A`, below `0x73E00000`.
The PTE and method addresses remain unchanged. The table and test artifacts
live in ignored resource/build directories.

`TinyStoriesProfiler` collects native operator callbacks without ETDump or
per-event allocation. It uses the existing backend execution hooks and linker
wrappers around the synchronous driver and dedicated backend I/O-copy function.
All stage times use CPU elapsed cycles. Hardware NPU TOTAL remains separate.
The shared CMake script adds one tracing condition for the TinyStories use case;
no runtime library or ExecuTorch submodule source changes are needed.

Every inference prints `TINYSTORIES_STAGE` records for:

- `CPU.Embedding`, `CPU.Quantize`, `CPU.Dequantize`, `CPU.Index`,
  `CPU.StateCopy` (native copy_ calls), and `CPU.OtherKernel`.
- `Delegate.Total`: the complete backend execution scope.
- `Delegate.InputCopy`, `Delegate.OutputCopy`: dedicated I/O copies, with
  actual call counts and bytes. Bytes on other stages are not collected.
- `NPU.Driver`: CPU elapsed time inside ethosu_invoke_v3, including submission,
  cache maintenance and synchronous waiting; it is not hardware NPU cycles.
- `Delegate.Other`: delegate total minus the two I/O-copy groups and driver.
  It includes scratch allocation, tensor/layout validation and bookkeeping.
  It is a residual, not an isolated layout-conversion measurement. The current
  PTE has matching input/output shapes and element sizes, so the padded or
  packed-output layout-adjustment path is not needed.
- `Runtime.Other`: invoke minus native kernels and complete delegate execution.

The additive relationships are:

```text
Invoke = CPU kernels + Delegate.Total + Runtime.Other
Delegate.Total = InputCopy + NPU.Driver + OutputCopy + Delegate.Other
```

Do not add child stages to their parent again. Each invocation checks these
relationships and requires one delegate and one driver call. Detailed tracing
adds timestamp/callback overhead; these are profiling-build measurements.
Stage logs are printed after the measured inference, outside its E2E interval.

Validation before FPGA deployment: the actual C++ decoder was compiled on the
host and compared with SentencePiece on 33013 sequences, including every single
vocabulary item, encoded mixed text, valid/invalid UTF-8 fallback sequences and
1000 random byte sequences: zero mismatches. Bounds failures were also checked.
The actual profiler was compiled with mock cycle readers/driver functions to
check native event classification, copy direction, nested accounting and
rejection of inconsistent elapsed totals. Actual board times still require
running the rebuilt firmware.

## CPU tracing callback fix

The first detailed-profile firmware reported zero native calls because the
use-case-specific selected operator library lacked ET_EVENT_TRACER_ENABLED.
Other runtime libraries enabled that macro. The linker selected the selected
operator library's no-op inline EventTracerProfileOpScope implementation,
which disabled native callbacks across the final image. The fixed build sets
the same macro on tinystories_portable_ops_lib. Final AXF disassembly confirms
that the selected scope constructor now checks the profiling level and calls
the tracer through its virtual interface.

At startup, TINYSTORIES_TRACER_SELFTEST must report success=1. Each inference
then checks exact call counts derived from the embedded PTE by
`tools/profile_contract.py`. Current native counts are:

| Stage | Calls per inference |
| --- | ---: |
| CPU.Embedding | 1 |
| CPU.Quantize | 36 |
| CPU.Dequantize | 19 |
| CPU.Index | 10 |
| CPU.StateCopy | 16 |
| CPU.OtherKernel | 0 |

These total 82 native calls. Delegate/driver counts must both be 1; input/output
copy counts must be 45/17. Required stages must also have nonzero measured cycles.
Any mismatch prints an explicit TINYSTORIES_STAGE invalid error and stops the
run; zero CPU counters can no longer be accepted as valid Runtime.Other data.
Successful calls print TINYSTORIES_STAGE_VALID position=... native_calls=82.
The expectations are regenerated when the selected PTE changes.

Regression checks now compile the real EventTracerProfileOpScope in a separate
registration translation unit. They reproduce the old mixed-macro link failure,
verify startup detection, and verify all 82 callbacks plus rejection of zero
callbacks, one missing quantize call, and inconsistent elapsed totals. These
checks cover the runtime hook boundary rather than only directly calling the
profiler's start/end methods. Board measurements still require running this
rebuilt firmware.

## INT16 KV persistence candidate

The separate candidate retains all sixteen KV cache tensors as INT16 between
calls. Its Ethos-U payload and all constant weight buffers are byte-identical
to the corrected FP32-cache baseline. The public token/position/logits interface
and 45-input/17-output delegate descriptors are preserved.

The baseline's saved mutable cache initializers are nonzero. They are converted
using each original cache's scale, ties-to-even float32 quantization and bounds;
they are not silently replaced with zero. Each delegate cache output is routed
back to the same persistent value used by the next call's cache input.

The original state transition is Q(DQ(q)). Across all 65536 INT16 values for each
of the sixteen parameter sets, this equals integer clamp to [-32767,32767].
The exceptional value -32768 must become -32767. The candidate therefore keeps
sixteen standard aten::clamp.out calls after the delegate, rather than blindly
retaining every raw output. These clamps use in-place INT16 state and no FP32
cache conversion. The new CPU.CacheClamp stage reports their cost separately.

| CPU stage | FP32-state baseline | INT16-state candidate |
| --- | ---: | ---: |
| Quantize | 36 | 20 |
| Dequantize | 19 | 3 |
| Embedding | 1 | 1 |
| Index | 10 | 10 |
| StateCopy | 16 | 0 |
| CacheClamp | 0 | 16 |
| Total native calls | 82 | 50 |

Reproduce in the WSL export environment, selecting a new output path (the
converter refuses to overwrite an existing candidate):

```sh
python source/app/use_case/tinystories/tools/retain_int16_kv.py \
  resources_downloaded/tinystories42m/npu_full_16a8w_embedding8_dedicated_terminalkv_1200/tinystories42m_u85_1024_ctx64_kv_full_w8a16_embedding8_dedicated_terminalkv_1200.pte \
  resources_downloaded/tinystories42m/npu_int16_kv_1200/tinystories42m_u85_1024_ctx64_int16_kv_1200.pte \
  resources_downloaded/tinystories42m/npu_int16_kv_1200/transform_report.json

python source/app/use_case/tinystories/tools/verify_int16_kv.py \
  resources_downloaded/tinystories42m/npu_full_16a8w_embedding8_dedicated_terminalkv_1200/tinystories42m_u85_1024_ctx64_kv_full_w8a16_embedding8_dedicated_terminalkv_1200.pte \
  resources_downloaded/tinystories42m/npu_int16_kv_1200/tinystories42m_u85_1024_ctx64_int16_kv_1200.pte
```

The candidate PTE is 55136256 bytes (baseline 56186752). Mutable cache initializer
payload decreases from 2 MiB to 1 MiB. This first candidate deliberately preserves
all planned memory offsets and the 4386304-byte arena; it does not claim a 1 MiB
runtime arena reduction. Removing unused values and compacting planned memory
is separate work. The PTE reduction does increase static DDR headroom.

Build directory: `build/tinystories42m/fpga/int16-kv-1p2m`. It selects the candidate
with tinystories_MODEL_PATH and retains the existing 8 MiB method pool, 2 MiB
scratch pool at 0x73E00000 and 1258304-byte fast cache. The candidate build sets
`tinystories_CACHE_CLAMP_OPTIMIZED=ON`, applying -O2 only to the standard integer
clamp source; all other operators retain the Debug build settings. The option
is OFF by default for other builds. The original baseline build is preserved.

Host verification executes the actual serialized CPU instructions on views of
the PTE's planned byte buffers, checking initialization and aliases. Across all
64 context positions, every one of the 45 delegate inputs matches exactly:
2880 input-array comparisons, zero mismatches. Simulated delegate outputs force
-32768 boundaries on every step. A negative control removing clamp is detected
on the next input. These checks do not execute the NPU or measure real speed.

On the board, require TRACER_SELFTEST success=1 and STAGE_VALID native_calls=50.
Compare prompt top1 and the full generated sequence with the previous NPU run
(which produces '... outside and explore'), independently of the existing host
PT2 token reference. Inspect remaining Quantize/Dequantize and CacheClamp time,
NPU TOTAL, E2E and memory peaks. The first board run matched all ten prompt top1
values and all sixteen generated tokens from the previous NPU firmware. Average
E2E decreased from 3604.78 ms to 1902.28 ms over 25 invocations, but the generic
integer clamp still cost 589.59 ms per invocation.

The next comparison build, `build/tinystories42m/fpga/int16-kv-fastclamp-1p2m`,
enables `tinystories_CACHE_CLAMP_SPECIALIZED=ON` (default OFF). CMake generates
a local copy of the portable clamp source with an early path for exactly the
in-place INT16 `[1,8,64,64]` cache and integral bounds -32767 / 32767. This path
scans storage once, replacing only -32768. Other inputs use the original kernel.
The ExecuTorch submodule, PTE, NPU payload, planned buffers and profiling call
counts remain unchanged. Only this clamp source is compiled at -O2.

`tests/CacheClampTest.cc` exhaustively checks all 65536 values against scalar
clamp, guards around the range, idempotence and an empty range. For example:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -Isource/app/use_case/tinystories/src \
  source/app/use_case/tinystories/tests/CacheClampTest.cc -o /tmp/test_cache_clamp
/tmp/test_cache_clamp
```

The new firmware prints `specialized in-place INT16 KV clamp enabled`. Require
TRACER_SELFTEST success=1, STAGE_VALID native_calls=50 and the same generated
tokens as the previous run. Measure CPU.CacheClamp and E2E on hardware; host
correctness checks do not establish an acceleration figure.
