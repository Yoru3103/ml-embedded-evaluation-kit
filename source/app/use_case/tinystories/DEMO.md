<!-- SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
<open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0 -->

# TinyStories UART demo

This independent build selects `tinystories_MODE=DEMO`; the default `TEST` mode
retains fixed-prompt smoke checks and immediate profiling output. The first demo
targets MPS4 SSE-320 / U85 Z1024, Dedicated SRAM 1258304 bytes. Display is off.
An independent FVP build uses the same model and demo flow, with the platform
flag OFF and the FVP UART address 0x58203000 (FPGA uses 0x59303000).

## Build and load the original FPGA baseline

The INT16 KV PTE, decoder table and encoder table must be present under
`resources_downloaded/tinystories42m`. Export the encoder in the existing WSL
Python environment if needed:

```sh
python source/app/use_case/tinystories/tools/export_encoder.py \
  resources_downloaded/tinystories42m/tokenizer.model \
  resources_downloaded/tinystories42m/tokenizer_encoder.bin
```

From the repository root in PowerShell:

```powershell
./source/app/use_case/tinystories/tools/build_fpga_demo.ps1
```

The compiler directory can be supplied with `-CompilerBin`. Output is separate
from the test firmware:

```text
build/tinystories42m/demo/fpga/bin/mlek_tinystories.axf
build/tinystories42m/demo/fpga/bin/sectors/images.txt
build/tinystories42m/demo/fpga/bin/sectors/tinystories/boot.bin
build/tinystories42m/demo/fpga/bin/sectors/tinystories/bram.bin
build/tinystories42m/demo/fpga/bin/sectors/tinystories/ddr.bin
```

Use the usual MPS4 image-loading procedure. Copy all three images, including
the new DDR image containing the encoder table. The FPGA configuration loads
boot at 0x11000000, BRAM at 0x12000000 and DDR at 0x70000000.

## Build and run the ctx128 milestone

For the milestone demo, use the single-delegate floor 1e-3 INT16 KV model with
context 128. The regular build commands above retain the original PTE default.
On a fresh Windows setup, provision the tool and Python environments using the
[Windows ExecuTorch setup](../../../../docs/sections/windows_executorch.md#安装配置与构建).
Install the Arm compiler and the Corstone SSE-320 FVP before running the demo.
From the repository root, pass the milestone PTE explicitly and use a separate
build directory. Build and start the functional FVP from PowerShell:

```powershell
$tools = 'source/app/use_case/tinystories/tools'
$model = (Resolve-Path 'resources_downloaded/tinystories42m/npu_relu1e3_ctx128/tinystories42m_u85_1024_ctx128_relu1e3_terminal_int16.pte').Path
& "$tools/build_fvp_demo.ps1" -ModelPath $model -Context 128 `
  -BuildDirectory build/tinystories42m/acceptance/fvp
& "$tools/run_fvp_demo.ps1" -BuildDirectory build/tinystories42m/acceptance/fvp
```

The FVP launch window displays output only: typing at its `Prompt>` does not
send UART input. After it reaches READY, open a **second PowerShell** in the
repository root and run the built-in Raw TCP console:

```powershell
./source/app/use_case/tinystories/tools/connect_fvp_demo.ps1

# Stable, deterministic demo: switch the firmware to greedy before the prompt
./source/app/use_case/tinystories/tools/connect_fvp_demo.ps1 -Greedy
```

Enter story openings at `PC prompt>` in the second window. Responses stream
there, followed by timing logs; `/quit` disconnects. Alternatively send one
prompt directly:

```powershell
./source/app/use_case/tinystories/tools/connect_fvp_demo.ps1 -Prompt 'Once upon a time in a small village,'
```

Use `-TcpPort` with both scripts if changing the default 5000. This client does
not restart FVP or reset the firmware. It sends LF and waits for the next READY
before accepting another prompt; allow several minutes for FVP generation.
The built-in FVP console displays the one DONE summary with host wall-clock
`total_ms` and `time_source=host_wall`, from sending the prompt through receiving
DONE. Raw TCP firmware output instead contains simulated CPU-clock time. The
console replaces that value rather than adding another timing line.

The milestone FVP image is `build/tinystories42m/acceptance/fvp/bin/mlek_tinystories.axf`.
Connect PuTTY or another **Raw TCP** terminal to `127.0.0.1:5000`, UTF-8,
local echo off. Use the actual telnetterminal0 port printed by FVP if the port
is already in use; `-TcpPort` changes the requested starting port. Plain Telnet
negotiation is not used. Input and output follow the same READY/Prompt/BUSY/DONE
flow below. Stop the simulator with Ctrl+C after closing the terminal.

The runner selects U85 1024 MACs and `--fast`, disables timing adapters and
visualisation, and uses an FVP-specific firmware build. The temp pool starts
at 0x73DC0000 with DDR overlap checks. Functional fast-mode cycle
counts are not FPGA performance measurements. A zero NPU PMU count is permitted
only in the FVP build; inference output and CPU tracing must still validate.

For FPGA:

Build the same milestone PTE for the board using the `$tools` and `$model`
variables above. The FPGA sectors are under the resulting `bin/sectors` directory.

```powershell
& "$tools/build_fpga_demo.ps1" -ModelPath $model -Context 128 -TimingAdapter `
  -BuildDirectory build/tinystories42m/acceptance/fpga
```

1. Open COM5 (or the board's UART port) at **115200, 8N1**, no flow control,
   UTF-8, local echo off. Reset the board.
2. Wait for `TINYSTORIES_DEMO READY` and `Prompt> `.
3. Type or paste an English story opening and send CR, LF or CRLF, for example:
   `Once upon a time in a small village,`.
4. Wait while BUSY. `Story:` contains the decoded prompt and continuation.
   Text is sent when `.`, `!`, `?` or newline becomes available. The remaining
   fragment is flushed at EOS or the generation limit.
5. After the story, one `DONE` line reports total time, generated token count
   and stop reason. Wait for the next READY before entering another prompt.

Encoding and decoding both run on the board; the PC terminal only transfers
and displays text. The model is an English story continuation model. Although
the tokenizer accepts UTF-8, this does not establish Chinese generation quality.

Input is limited to 256 UTF-8 bytes and 127 tokens including BOS. Overlong input
is rejected, never silently truncated. Backspace edits the input; terminal
display width for non-ASCII characters depends on the terminal. Empty lines and
malformed UTF-8 are rejected. Prompts should contain printable text. Do not send
the next request while BUSY: hardware UART buffering is limited.

Generation uses all remaining space in the 128-token context. A ten-token prompt
including BOS permits up to 118 new tokens. The default decoder samples with
`temperature=0.8`, `top_p=0.9` and `top_k=64`. Lines beginning with `!` are
commands, not prompts: `!greedy`, `!sample`, `!seed N`, `!temp F`, `!topp F`,
`!topk N`. The seed is derived from a fixed base plus the request index and is
printed in DONE, so a run can be reproduced with `!seed`. EOS ends generation
early. `reason=context_limit` means the exported context is full; `reason=eos`
means the model ended the story itself. A story may end mid-sentence at the
context limit. More context requires a new PTE export and memory/compilation
validation. There is no cancellation or conversation history.

The opening is free text; `Once upon a time` is optional. Other example prompts:

```text
A little cat found a magic key.
Lily wanted to fly like a bird.
One rainy morning, Tom heard a strange sound.
In a small village, a lonely robot made a new friend.
```

These are example story openings, not guarantees of a particular plot. Shorter
prompts leave more of the fixed context for the continuation. Tokens are BPE
pieces, not a fixed number of words or characters.

## Isolation and timing

Every request zeros exactly the sixteen persistent INT16 KV regions identified
from the configured PTE, then starts position at zero. Model weights, other
planned buffers and allocator bookkeeping are preserved. The demo supplies
caller-owned planned-buffer Span storage with static lifetime, so its reset
allocator remains valid after model initialization. The model is loaded once.

Per-call inference and CPU/delegate stage records remain in SRAM and are
validated silently. DEMO does not dump them, token IDs or memory reports after
the story. TEST keeps its detailed diagnostic output.

`total_ms` measures from the start of prompt encoding through the last story
output call. It includes encoding, KV reset, inference, argmax, decoding, PMU
setup and sentence UART writes. It excludes user typing, model initialization
and the final summary/READY output. DEMO resets CPU SysTick once per request,
then resets only the NPU PMU between tokens, preserving the full interval.
The CPU clock is used for conversion; on functional FVP this is simulated time,
not host waiting time or FPGA performance. Text is decoded from the complete
token prefix on each step to preserve UTF-8 sentence output.

The milestone PTE is 56201856 bytes. The encoder adds 594923 bytes. Static DDR
usage is 66832723 / 67108864 bytes, including the 7 MiB method pool, 2.25 MiB temp pool
at 0x73DC0000 and both token tables. The linker also checks tokenizer data stays
below the temporary pool. Demo timing/text buffers use SRAM.

## Verification

For a handoff, record SHA-256 for the PTE, both tokenizer tables, the AXF and
any FPGA sector images. Include the compiler and FVP versions, the configured
context and TA setting, and the DDR budget printed by CMake. The
[precision report](docs/TinyStories_U85_Precision_Master_Report.md)
describes the export and INT16 KV compaction. The PTE and tokenizer tables
must be supplied under `resources_downloaded/tinystories42m` on a fresh setup.
If you received `tinystories-ctx128-build-inputs.zip`, extract it at the
repository root to populate those paths. Verify the files against the included
`input-sha256.txt` before building.

For functional acceptance, use FVP in greedy mode and send A, B, then A again
without restarting the firmware. Confirm that the two A stories match and all three
reports show `cache_reset=1` and `DONE success=1` in the raw UART log. Check
that an overlong prompt is rejected and the next valid prompt still runs.
Use a short prompt to check `reason=context_limit`. For FPGA performance, record
time from prompt submission to first generated text and the elapsed time per
decode token from a TA ON TEST trace, together with the clock settings.

The floor 1e-3 ctx128 FVP acceptance run rejected a 257-byte input and then
accepted the greedy sequence `Once upon a time,`, `A little cat found a magic key`,
and `Once upon a time,`. The requests generated 122, 120 and 122 tokens. All
three reset KV state and returned `DONE success=1` with `reason=context_limit`.
Both A responses matched byte for byte.

The actual C++ encoder/decoder match SentencePiece for 1512 deterministic test
prompts, including whitespace, UTF-8 and byte fallback; five malformed/oversized
inputs are rejected. Build the host harness with:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror -DTINYSTORIES_DEMO \
  -Isource/app/use_case/tinystories/src \
  source/app/use_case/tinystories/tests/TokenizerCli.cc \
  source/app/use_case/tinystories/src/TokenEncoder.cc \
  source/app/use_case/tinystories/src/TokenDecoder.cc -o /tmp/demo_tokenizer_test
python source/app/use_case/tinystories/tools/verify_demo_tokenizer.py \
  /tmp/demo_tokenizer_test resources_downloaded/tinystories42m/tokenizer.model \
  resources_downloaded/tinystories42m/tokenizer_encoder.bin \
  resources_downloaded/tinystories42m/tokenizer_surfaces.bin
```

On hardware, run the same prompt twice, then a different prompt and the first
prompt again. Compare story text for repeated prompts and require
`cache_reset=1`, `DONE success=1` and tracer self-test success. DONE succeeds only
when all saved stage contracts validate; their normal diagnostics are hidden.
Check the story is followed by only one total-time summary, with no token,
per-call timing or memory dumps.
Also try a prompt without punctuation (final fragment must flush), oversized
input, CRLF input and a prompt near the token limit. Hardware execution and
multi-request isolation still require this board verification.

The earlier verbose FVP UART regression completed A/B/A with 32 generated
tokens each. Both A outputs matched exactly, all 41/40/41 invocation stage
contracts passed, and timing output followed DONE in each response. The first
16 A tokens also matched the FPGA test log. The report and full UART responses
are in `build/tinystories42m/demo/fvp/uart-regression/`. This functional FVP check
does not replace FPGA hardware validation or establish FPGA throughput.

The 128-context FVP regression generated 122 tokens for the six-token
`Once upon a time,` prompt and ended with `reason=context_limit`. Only one time
summary appeared; per-call, stage, token and memory dumps were absent. The
sampled default produced a coherent multi-sentence story and reported
`DONE success=1`. Functional FVP intervals are simulated time, not FPGA
performance; use the built-in console for actual waiting time.
