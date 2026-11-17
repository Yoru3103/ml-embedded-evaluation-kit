<!--
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
-->

# FVP audio demo on Windows

The `kws_zh` and English Conformer `asr` demos accept WAV or microphone audio through a
custom minimal VSI0 pull-DMA adapter in the application layer. The host supplies mono
16 kHz PCM16 blocks on request. The adapter uses the FVP VSI Python callbacks and DMA
registers directly; this is not a CMSIS vStream audio-driver integration. HAL and
dependency sources do not need changes.

KWS uses a 1.995-second context with a default 1000-millisecond inference step. The
waveform is rasterized in an application-owned RGB buffer and submitted in one image
update; the application avoids clearing the full LCD for each waveform refresh.
The `queue_ms` log field and LCD queue value describe buffered audio duration, not measured
end-to-end latency. Short WAVs can trigger a final padded inference at EOS.
Real-time factor has not been benchmarked: the inference step is not a latency guarantee.
LCD labels use the canonical Latin names `Hi_Xiaowen` (嗨，小问) and
`Nihao_Wenwen` (你好，问问). A successful detection also shows the ASCII greeting
`NI HAO! ^_^` and prints it in the terminal, using the built-in LCD font.
The result panel is updated only by a new detection, including across input gaps.

On live queue overrun, ASR and KWS reset their context without printing a message for
each discarded block. Repeated gaps form one recovery episode: the status changes once,
identical status redraws are suppressed, and KWS waveform refreshes pause until a full
continuous window is available. Successful results and actual input/inference errors still
print normally. This prevents UART/LCD work from sustaining the queue-overrun loop.
The Windows runner defaults to a 30-second ASR host queue and a 5-second KWS host queue;
`-QueueMs` overrides either. Buffering absorbs temporary inference delays, but cannot make
a persistently slower-than-live inference path lossless. Audio evicted at the queue limit
cannot be recovered, and a new continuous segment is required after a gap.

Both firmware configurations target SSE-320 FVP, U85 Z1024, Dedicated SRAM with a
256 KiB NPU cache, LCD enabled, semihosting disabled, and timing adapters disabled.
The runner uses functional NPU fast mode; its timings are not FPGA performance results.

## Setup and models

Follow the [native Windows prerequisites](../windows.md) and run the
following from the repository root to prepare the ExecuTorch build environment:

```powershell
.\scripts\windows\build.ps1 -Framework pte -Action Setup
```

Supply an external, compiled U85-1024 Dedicated SRAM 256 KiB PTE that matches the
selected use case. Use `-ModelPath` to pass its location or put it in the profile's
default location:

| Use case | Default model |
| --- | --- |
| `kws_zh` | `resources_downloaded/kws_zh/pte_Dedicated_Sram_256k/wekws_dstcn_u85.pte` |
| `asr` | `resources_downloaded/asr/et151-u85-1024-dedicated-256k/conformer_fp32_cln_wer_6_47_arm_delegate_ethos-u85-1024_Dedicated_Sram.pte` |

The required input and output layouts are documented in the
[Chinese KWS](../../use_cases/kws_zh.md) and [English ASR](../../use_cases/asr.md) use cases.
For a fresh setup of the Chinese Zipformer demo, see [Chinese ASR](asr_zh.md).

| Component | Location |
| --- | --- |
| CMake and Ninja | `resources_downloaded/windows-tools/env/Scripts` |
| ExecuTorch Python | `resources_downloaded/windows-executorch/env/Scripts/python.exe` |
| ATFE | Discovered by `windows-venv.ps1`; override with `-CompilerBin` |
| Default FVP | `C:/Program Files/Arm/FVP_Corstone_SSE-320/models/Win64_VC2019/FVP_Corstone_SSE-320.exe` |

The runner gives the host its own Python environment, then configures the FVP with its
bundled Python home (`<FVP installation>/python`). That distribution can lack `_socket.pyd`.
Supply matching **x64 CPython 3.9** extension modules with `-FvpPythonExtensions`; the runner
automatically checks `%LOCALAPPDATA%/Programs/Python/Python39/DLLs` and then
`resources_downloaded/fvp-python39`. Keep any supporting DLLs beside the extension modules.
Do not use Python 3.12 or 3.14 extensions in the embedded Python 3.9 interpreter.
`-FvpPythonHome` overrides the bundled home. Process environment changes are restored on exit.

## Build

```powershell
./scripts/windows/build.ps1 -Profile AudioDemo -UseCase kws_zh -Jobs 4
./scripts/windows/build.ps1 -Profile AudioDemo -UseCase asr -Jobs 4
```

The separate output directories are `build/fvp-audio/kws_zh` and `build/fvp-audio/asr`.
Firmware is `bin/mlek_kws_zh.axf` or `bin/mlek_asr.axf` inside the corresponding directory.
Use `-Action Configure` for configuration only, `-ModelPath` for another compatible PTE,
and `-BuildDirectory` for a separate output directory. Existing unrelated builds are not reused.
`-InputMode STATIC` selects the existing embedded sample path in a separate `-static` directory.
The default input mode is `VSI`; it does not enable the camera-oriented `FVP_VSI_ENABLED` option.

ASR microphone sentence detection can be tuned with `-VadRms`, `-PreRollMs` and
`-SilenceMs`. The default build uses `-VadRms 800 -PreRollMs 600 -SilenceMs 1000`.
These ASR-only options set `asr_VSI_VAD_RMS` (1–32767 PCM RMS units) and
`asr_VSI_SILENCE_MS` (20–5000 milliseconds). Raise the RMS threshold when ambient noise
prevents sentence termination. The host does not perform sentence detection.
ASR uses energy VAD with 600 milliseconds of pre-roll and a default 1000-millisecond
silence endpoint. It requires 120 milliseconds of consecutive speech to start, uses
half the starting RMS threshold to maintain an active capture, and ignores utterances
with less than 250 milliseconds of voiced samples. Status changes and empty results
do not clear the last successful transcript. It decodes complete utterances rather than streaming incremental
tokens, and splits at the 30-second maximum utterance length.

## Run

```powershell
./scripts/windows/run-fvp.ps1 -Profile AudioDemo -UseCase kws_zh `
    -Wav resources/kws_zh/samples/01_hi_xiaowen.wav
./scripts/windows/run-fvp.ps1 -Profile AudioDemo -UseCase asr `
    -Wav resources/asr/samples/another_door.wav
./scripts/windows/run-fvp.ps1 -Profile AudioDemo -UseCase asr -Microphone -Device 1 `
    -QueueMs 30000 -Duration 30
```

Install `sounddevice` in the selected host Python environment for microphone input:

```powershell
.\resources_downloaded\windows-executorch\env\Scripts\python.exe -m pip install sounddevice
```

Enumerate devices before selecting an index:

```powershell
./resources_downloaded/windows-executorch/env/Scripts/python.exe `
    scripts/py/fvp_audio_demo.py --list-devices
```

`-Realtime` optionally paces WAV playback; it applies only to WAV mode. `-QueueMs` sets
the bounded source queue (minimum 20 milliseconds). The launcher defaults to 5000 ms
for KWS and 30000 ms for ASR. The ASR queue can retain speech captured while the previous
sentence is being inferred; the KWS queue and longer hop reduce repeated context resets
under moderate overload. `-Duration` limits source acquisition.
The host listens on localhost port 6010 by default; `-Port` changes both the host CLI and
the bridge's inherited `FVP_AUDIO_DEMO_PORT`. Capture starts on the firmware's first pull,
after model initialization. The runner waits for the host's readiness log without consuming
its single client connection.

WAV playback applies backpressure and does not drop source samples. Microphone capture
evicts the oldest blocks when its bounded queue fills. A resulting discontinuity resets
the KWS context or aborts the current ASR partial utterance. Input must already be
16 kHz mono PCM16; this path performs no resampling or denoising.

The FVP receives `mps4_board.v_path=<absolute repository>/scripts/py/fvp_audio_demo_vsi`,
`mps4_board.subsystem.ethosu.num_macs=1024`, and
`mps4_board.subsystem.cpu0.semihosting-enable=0`. The board exterior is hidden and the LCD
window is enabled. Ctrl+C or Q in the launching terminal stops the demo. Both modes
launch FVP as an explicitly owned process and forward its log to the terminal from a
short polling loop. Cleanup terminates both owned processes and waits for their exit;
there is no blocking foreground FVP invocation. Logs are kept even in interactive mode.

For bounded smoke tests, suppress the LCD window while retaining firmware display support:

```powershell
./scripts/windows/run-fvp.ps1 -Profile AudioDemo -UseCase kws_zh `
    -Wav resources/kws_zh/samples/01_hi_xiaowen.wav -Headless -SmokeTimeoutSeconds 120
```

Normal WAV playback keeps FVP and the final LCD result alive after EOS. Press Q or Ctrl+C
in the launching terminal to close the demo. The audio host may exit successfully at EOS;
this does not close the LCD. Rebuild both audio demo firmwares after updating this behavior.

Smoke mode waits for `AUDIO_EOS` (KWS) or `ASR_VSI_EOS ... success=1` (ASR), then terminates
the FVP and host processes it started. A timeout or premature FVP exit fails the command.
Host logs are `audio-host.log` and `audio-host.err.log`; smoke FVP logs are `audio-fvp.log`
and `audio-fvp.err.log` in the selected build directory. Each run replaces those logs.

## Earlier end-to-end validation and troubleshooting

The overrun-recovery fix was checked by rebuilding both firmware images, native ASR
handler tests with 1500 consecutive discontinuities followed by a second recognized
phrase, and an audio-window test crossing 100 seconds of source time before a prolonged
gap episode and recovery. These are software regressions, not a new long live-microphone
test. The ASR mock also verifies that LCD writes stay bounded during repeated gaps.

The recorded FVP results below predate the LCD retention, VAD and cancellation fixes.
The fixes were checked with native regression tests, PowerShell parsing, owned dummy
process cleanup and firmware builds; no new live microphone or extensive FVP runs were
performed. `tests/processing/fvp_audio/Test-ProcessCleanup.ps1` runs the simple exit check.

Both use cases built successfully with ATFE 22.1.0 on Windows. The short KWS WAV smoke
reached `AUDIO_EOS sample=29968` and detected `Hi_Xiaowen` with score `0.874083` at `1.145s`.
This also exercised the EOS-padded inference path for a WAV shorter than the model window.
The ASR WAV smoke reached `ASR_VSI_EOS 62480 success=1` and decoded:
"and he walked immediately out of the apartment by another door".
Microphone device enumeration and the Realtek device 1 input-format check for 16 kHz
mono PCM16 passed locally. Actual microphone capture has not been tested.
The concatenated KWS stream detected `Hi_Xiaowen` at `2.145s` and `Nihao_Wenwen` at
`5.065s`, then reached EOS at sample 106944. Background-only audio reached EOS at sample
49760 without any wake events. Local evidence is preserved in `smoke-short.log`,
`smoke-stream.log`, and `smoke-background.log` under `build/fvp-audio/kws_zh`.
The two-sentence ASR WAV produced two `SILENCE` utterance boundaries and two matching
transcriptions, then `ASR_VSI_EOS 180960 success=1`. Its evidence is
`build/fvp-audio/asr/smoke-sentences.log`; the single-WAV EOS test is `smoke-eos.log`.
The paced (`-Realtime`) KWS stream also produced exactly two wake events and EOS;
`smoke-realtime.log` records queue durations of 820 and 980 milliseconds at those events.
This run exercised source pacing but did not establish that the full FVP path keeps up
with live input. Queue depth was not near zero.
A timed repeat on this machine took **35.061 seconds wall time for 6.684 seconds of
source audio**, with `queue_ms=980.0` at both wake events. Wall time covers host startup,
FVP/model initialization, audio processing, EOS detection, and cleanup; it is not a
steady-state inference real-time-factor measurement. Evidence is saved as
`smoke-realtime-timed.log` and `smoke-realtime-timing.json` in the KWS build directory.
These runs do **not** demonstrate real-time operation. WAV backpressure preserves samples
while allowing playback to fall behind. Under sustained microphone overload, bounded
eviction can instead drop audio and reset the KWS window or abort an ASR partial utterance.

After the simple batched-waveform change, the paced two-keyword regression passed again
at the same sample timestamps. That run took 22.986 seconds including startup and cleanup,
with queue values of 800 and 980 milliseconds; see `smoke-batched.log`. This single-run
comparison is not a controlled latency benchmark.

The installed FVP's `--list-params` confirms `mps4_board.v_path`, VSI0's `arm_vsi` basename,
the semihosting-enable parameter, MAC configuration, and LCD visualization controls.
Evidence is saved locally in `build/fvp-audio/fvp-verified-params.txt`.

In the managed agent sandbox, the Windows FVP launcher failed with an invalid system DLL
relocation error and WSL returned access denied. Both launched outside that sandbox with
reviewed execution. An `_socket` import failure instead indicates missing Python 3.9
extension modules; inspect `audio-fvp.err.log` and set `-FvpPythonExtensions` as above.

## Regression tests

The Python host/bridge suite contains 20 passing regression tests:

```powershell
./resources_downloaded/windows-executorch/env/Scripts/python.exe -m unittest discover `
    -s tests/py -p 'test_fvp_audio_demo*.py'
```

Standalone native tests can run in WSL with a C++17 compiler. The window and ASR
VAD/handler suites use address and undefined-behavior sanitizers with GNU/Clang:

```sh
cmake -S tests/processing/fvp_audio -B build/fvp-audio-window-tests
cmake --build build/fvp-audio-window-tests -j2
ctest --test-dir build/fvp-audio-window-tests --output-on-failure
cmake -S tests/processing/asr_vsi -B build/asr-vsi-tests -DCMAKE_BUILD_TYPE=Debug
cmake --build build/asr-vsi-tests -j2
ctest --test-dir build/asr-vsi-tests --output-on-failure
```

See the [ASR native test guide](../../../tests/processing/asr_vsi/README.md) for coverage
and sanitizer options. These native suites validate segmentation/control flow rather
than model accuracy or the FVP hardware interface.
