# Chinese ASR with streaming Zipformer CTC

Current status (2026-11-02): INT8-state Dedicated 768 KiB and corrected standard
Shared_Sram firmware both complete zh1.wav on FPGA. A seven-clip AISHELL boundary
suite completes 38 chunks in FVP; silence, short tails and clip reset pass, but four
clip texts differ from PT2 (including repeated/concatenated variants of the same source).
This is a functional use case with an open backend numerical-accuracy issue.
See the local records under `docs/reports/asr_zh_devrecord/` for the small regression,
full-audio and fixed-feature experiments. This directory is intentionally ignored by Git.

This independent `asr_zh` use case targets ExecuTorch and Ethos-U85-1024.
It processes PCM16 audio incrementally. Static audio, UART uploads and VSI microphone/WAV
input are available; see [the demo guide](../sections/demos/asr_zh.md) for current recording
options. Chinese text is shown in the terminal; the LCD shows English status messages.

The user-audio/FPGA build with ordered hook timing is documented in
`docs/reports/asr_zh_devrecord/asr_zh_zh1_fpga_20261029.md`. For custom audio use `-NoValidation`;
`-NoTensorDump` also disables semihosted tensor dumps in FVP builds. Stage logs now
include Quantize, NPU.Driver, Dequantize and Invoke.Other within each Invoke.Total,
plus preprocessing, decoding and state feedback. The summary includes clip setup
and final text decoding; UART output is outside the measured intervals.

## Model and data contract

The model has 75 inputs and 75 outputs: features at slot 0, 72 layer-cache tensors,
an embedding cache at slot 73, and an INT32 processed-frame counter at slot 74.
Each state output feeds the same input slot on the next chunk. Every new clip resets
the complete state bank and CTC repeat state.

- Features: normalized PCM16 / 32768, 16 kHz, 80-bin Kaldi-compatible log-Fbank.
- Framing: 400 samples / 160-sample shift; Povey window, DC removal, 0.97 preemphasis,
  dither=0, 20–7600 Hz, snip_edges=False with reflected boundaries.
- Model input: FP32 [1,77,80]; advance 64 feature frames (640 ms).
- Output: FP32 [1,16,1000] CTC log probabilities; blank=0.
- Decoder: greedy CTC across chunks, matching byte-BPE lookup and maximal-valid UTF-8
  recovery. The current Chinese display removes ASCII spaces, matching the reference demo.

The current frontend borrows the full PCM clip while computing each real feature frame once.
It retains the 13 overlapping feature frames. It does not yet expose an online PCM
push API. The padded final model chunk follows the existing Python evaluation policy.

## Build

Prepare model resources in WSL as described in `resources/asr_zh/README.md`.
Use the existing Windows toolchain wrapper:

```powershell
# Three fixed reference feature chunks: isolate the NPU/backend from Fbank.
./scripts/windows/build-asr-zh.ps1 -Target FVP -FixedFeatures

# Complete static-audio frontend, streaming inference and decoding.
./scripts/windows/build-asr-zh.ps1 -Target FVP

# Recommended INT8 state interface; Dedicated 768 KiB.
./scripts/windows/build-asr-zh.ps1 -Target FPGA -Int8State `
  -InputPath resources_downloaded/asr_zh/zh1/zh1.wav -NoValidation -NoTensorDump

# Standard shared SRAM: 3 MiB SRAM temp pool, method/planned pool in DRAM.
./scripts/windows/build-asr-zh.ps1 -Target FPGA -MemoryMode Shared_Sram `
  -InputPath resources_downloaded/asr_zh/zh1/zh1.wav -NoValidation -NoTensorDump
```

The static wrapper accepts `-TimingOutput AUTO|OFF|SUMMARY|DETAILED` (default `DETAILED`).
The existing demo commands use `AUTO`, selecting compact UART timing and unprofiled VSI
input without changing their arguments. `OFF` disables ASR profiling; explicit `SUMMARY`
or `DETAILED` requires CPU profiling. Stage totals exclude printing and display.

The wrapper selects Release, Dedicated 768 KiB, an 8 MiB DRAM method pool,
and 1.5 MiB of DRAM temporary memory. FVP uses semihosting; FPGA selects UART retarget.
Shared_Sram instead uses a 3 MiB temp pool at 0x31100000. The NPU scratch is allocated
from that pool, not from an additional SRAM partition. Standard Shared_Sram has
arena_mem_area=cache_mem_area=Axi0 and requires no separate fast scratch. The former
custom DRAM-arena/shared configuration is invalid for this runtime and is obsolete.
The first build compiles the runtime and model array and takes substantially longer
than an incremental build.

By default validation is specific to demo.wav. For another WAV, pass `-InputPath`
and `-NoValidation`. Fixed-feature mode intentionally processes only the three fixture
chunks regardless of the supplied WAV. It must not be used as an audio recognition benchmark.
Use `-AllInputs` with a directory to process multiple WAV files and reset state per clip.

The `all` build skips this model when its target configuration or local PTE is unavailable.
CPU-only and TFLM execution of this PTE are not supported.

## FVP validation

```bash
# Linux/WSL; FVP_PATH can override the default installed model.
timeout -k 5 240 bash scripts/run_asr_zh_fvp.sh > build/asr-zh-fvp-fixed.log 2>&1
/home/mokuroo/.venvs/mlek-asr-et151/bin/python scripts/model_export/asr_zh/check_asr_zh_fvp.py \
  build/asr-zh-fvp-fixed.log build/asr_zh/fvp-768k-fixed --fixed

timeout -k 5 300 bash scripts/run_asr_zh_fvp.sh build/asr_zh/fvp-768k \
  > build/asr-zh-fvp-audio.log 2>&1
/home/mokuroo/.venvs/mlek-asr-et151/bin/python scripts/model_export/asr_zh/check_asr_zh_fvp.py \
  build/asr-zh-fvp-audio.log build/asr_zh/fvp-768k
```

Semihosting must be enabled on the model; the run script supplies the parameter.
An FVP process exit code of zero alone is insufficient: an application can report a
delegate failure and still exit normally. The checker requires successful completion,
checks 74 quantize / 2 NPU / 74 dequantize calls per chunk, and verifies the transcript
in complete-audio mode. It compares all 75 outputs of the first three chunks with the
PT2 reference and writes per-output errors to a JSON report next to the log.

These are functional fast-mode checks, not FPGA timing measurements.
The generated dump files use the FVP build directory as semihosting cwd.
The static runner validates the actual CMake memory mode: Dedicated requires 768 KiB
cache; Shared requires zero separate cache. Its historical default directory
`build/asr_zh/fvp-768k-fixed` currently contains a Shared build; pass an explicit build
directory for a specific baseline. VSI/UART demos use their separate launchers.
Current dumps are named `asr_zh_<clip>_<chunk>_inputs.bin` and `_outputs.bin`, using
three-digit zero-based indices. The first three chunks per clip are dumped; all chunks
emit `ASR_ZH_REGRESSION` frame paths when dumping is enabled. The legacy checker is
for the FP32-state demo; use `scripts/model_export/asr_zh/asr_zh_regression.py` for the
INT8-state boundary suite.
Dumping, validation prints and instrumentation are outside the reported processing
cycle intervals, but may perturb caches; disable validation/dumps for performance work.
On invoke failure, `ASR_ZH_FAILED` identifies the clip/chunk and separate `Failed.*`
stages report the failed attempt. The normal summary contains completed chunks only;
do not read a zero completed-chunk summary as zero work performed before failure.

Model generation uses the same common `validate_pte_model` checks as other PTE use
cases: NPU delegation, accelerator/MAC configuration and memory mode must match the
firmware configuration. No additional model hash gate is introduced. These checks do
not establish numerical equivalence or validate arbitrary custom Vela INI semantics.

## Native tests

```bash
cmake -S tests/processing/asr_zh -B /tmp/mlek-asr-zh-native -G Ninja
cmake --build /tmp/mlek-asr-zh-native -j4
ctest --test-dir /tmp/mlek-asr-zh-native --output-on-failure
```

Address/undefined-behavior sanitizers are enabled on supported native GCC/Clang builds.
Tests cover independent Fbank fixtures including reflected boundaries, exact feature
overlap reuse, tail padding, silence, short clips, CTC blank/repeat behavior across
calls, incomplete/invalid UTF-8 recovery, bounds and all state-buffer feedback.

## Memory placement

Method/planned memory and the separate input buffers live in DRAM. With INT8 states,
the planned+input subtotal is 1,355,696 bytes (706,176 + 649,520); measured method-pool
use including metadata is 1,370,792 bytes. The reserved 8 MiB pool is not the minimum.
The old FP32-state interface used 5,103,796 bytes including metadata.

Dedicated mode needs 1,257,104 bytes of delegate scratch in DRAM and a separate
768 KiB fast SRAM region. Standard shared mode needs 1,769,200 bytes of SRAM scratch
from the 3 MiB temp pool, with no separate fast cache. Validate these against the selected
PTE and its delegate I/O report. Model data is approximately 22 MiB in the read-only
DRAM area. Check the ELF map, temp region and any display/heap/stack allocations before
deployment. The build does not redefine platform-independent library code to call HAL.

## Accuracy and backend findings

The unchanged INT8 PT2 was independently evaluated on a fixed AISHELL-1 test sample:
200 utterances, 20 speakers, 16.57 minutes; FP32 CER 7.9514%, INT8 CER 8.2639%.
This is a sample, not the full test set, and does not establish NPU accuracy.

FVP exposed a defect in the earlier export shortcut: cached TOSA placed the INT32
processed counter first while the newly partitioned delegate passed the INT8 feature
tensor first. Graph/file hashes alone did not capture preprocessing mutations.
The direct-TOSA-cache shortcut is disabled in the export project. A corrected PTE
must pass `scripts/model_export/asr_zh/validate_asr_zh_pte.py`, which compares every runtime delegate
argument with the embedded Vela element width and tensor capacity.

Compilation and validation results for the corrected artifact are recorded alongside
the build logs. Do not deploy the previous PTE with SHA256
9504ceb63703414683732eb5c3355582f4eb9fee344b6e15f23a77835f3be9b8.

## Implementation boundaries

Platform-independent Fbank, state bank and decoder live in `source/lib/mlek/use_case/asr_zh`.
HAL audio, PMU access and the driver profiling wrapper live in the app use case.
The existing English asr handlers, windowing and hard-coded profiler counts are not reused.
The general ExecuTorch build receives operator tracing for asr_zh when CPU profiling is enabled.


Historical checkpoints and backend investigations are archived locally under
`docs/reports/asr_zh_devrecord/`. They describe the configuration at the time of each
experiment; use the current CMake cache to identify an existing build directory.
