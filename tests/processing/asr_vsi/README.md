<!-- SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
     <open-source-office@arm.com>
     SPDX-License-Identifier: Apache-2.0 -->

# Native ASR VSI regression tests

Run from the repository root with a native C++17 compiler and CMake 3.21 or newer:

```sh
cmake -S tests/processing/asr_vsi -B build/asr-vsi-tests -DCMAKE_BUILD_TYPE=Debug
cmake --build build/asr-vsi-tests -j2
ctest --test-dir build/asr-vsi-tests --output-on-failure
```

The suite contains three CTest tests:

- `asr-vsi-vad`: click rejection, confirmed speech start, pre-roll retention and absolute
  positions, silence endpoint, reset, 30-second capacity, and hysteresis/RMS boundaries.
- `asr-vsi-custom-vad`: configured RMS 400 and silence duration 1000 ms.
- `asr-vsi-handler`: compiles the actual `ConformerVsiHandler.cc` and exercises short
  noise rejection, qualified EOS with/without PCM, silence-only EOS, explicit discontinuity,
  implicit position gaps, pre-roll discard, ERROR and failed-read aborts, malformed
  empty input, sentence-end/EOS deduplication, closing the input, and retaining the
  last decoded LCD text at EOS and after a subsequent input gap.

The handler test uses the real audio block interface, VAD and sliding-window headers.
Minimal mocks replace the model, preprocessing/postprocessing, transport, logging and
display. No HAL, core-platform, ExecuTorch, firmware libraries or submodule builds are
needed. These tests check control flow and PCM segmentation; they do not measure model
accuracy or exercise the VSI bridge hardware.

Address and undefined-behavior sanitizers are enabled by default for GNU/Clang.
Use `-DASR_VSI_TEST_SANITIZERS=OFF` if the native toolchain lacks sanitizer support.
Assertions remain enabled in Release builds. For multi-configuration generators, add
`--config Debug` when building and `-C Debug` when running CTest.
