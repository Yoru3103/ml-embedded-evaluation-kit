# Noise Reduction Code Sample

- [Noise Reduction Code Sample](./noise_reduction.md#noise-reduction-code-sample)
  - [Introduction](./noise_reduction.md#introduction)
  - [How the default neural network model works](./noise_reduction.md#how-the-default-neural-network-model-works)
  - [Post-processing](./noise_reduction.md#post_processing)
    - [Windows FVP demo and listening report](#windows-fvp-demo-and-listening-report)
    - [Saving WAV files directly from an FVP](#saving-wav-files-directly-from-an-fvp)
    - [Processing longer inputs](#processing-longer-inputs)
    - [Extracting WAV files from an MPS3 memory dump](#extracting-wav-files-from-an-mps3-memory-dump)
    - [Dumping of memory contents from the Fixed Virtual Platform](./noise_reduction.md#dumping-of-memory-contents-from-the-fixed-virtual-platform)
    - [Dumping post processed results for all inferences](./noise_reduction.md#dumping-post_processed-results-for-all-inferences)
  - [Prerequisites](./noise_reduction.md#prerequisites)
  - [Building the code sample application from sources](./noise_reduction.md#building-the-code-sample-application-from-sources)
    - [Build options](./noise_reduction.md#build-options)
    - [Build process](./noise_reduction.md#build-process)
    - [Add custom input](./noise_reduction.md#add-custom-input)
    - [Add custom model](./noise_reduction.md#add-custom-model)
  - [Setting up and running Ethos-U NPU code sample](./noise_reduction.md#setting-up-and-running-ethos_u-npu-code-sample)
    - [Setting up the Ethos-U NPU Fast Model](./noise_reduction.md#setting-up-the-ethos_u-npu-fast-model)
    - [Starting Fast Model simulation](./noise_reduction.md#starting-fast-model-simulation)
    - [Running Noise Reduction](./noise_reduction.md#running-noise-reduction)

## Introduction

This document describes the process of setting up and running the Arm® Ethos™-U NPU Noise Reduction
example.

Use case code is stored in the following directory: [source/app/use_case/noise_reduction](../../source/app/use_case/noise_reduction).

> **NOTE**: This use case only supports `TensorFlow Lite Micro`.

## How the default neural network model works

Instead of replicating a "noisy audio in" and "clean audio out" problem, a simpler version is
defined. We use different frequency bands for the audio (22 in the original paper
[RNNoise: Learning Noise Suppression](https://jmvalin.ca/demo/rnnoise/)). It is based on a scale like the "Mel scale"
or "Bark scale" and calculates the energies for each band. Using this type of scale, the bands get
divided up and the result is based on what is important to the human ear.

When we have a noisy audio clip, the model takes the energy levels of these different bands as
input. The model then tries to predict a value (called a gain), to apply to each frequency band. It
is expected that applying this gain to each band brings the audio back to what a "clean" audio
sample would have been like. It is like a 22-band equalizer, where we quickly adjust the level of
each band so that the noise is removed. However, the signal, or speech, still passes through.

In addition to the 22 band values calculated, the input features also include:

- First and second derivatives of the first 6 coefficients,
- The pitch period (1/frequency),
- The pitch gain for six bands,
- A value used to detect if speech is occurring.

This provides 42 feature inputs, `22 + 6 + 6 + 1 + 6 + 1 = 42`, and the model produces `22` (gain
values) outputs.

> **Note:** The model also has a second output that predicts if speech is occurring in the given
> sample.

The pre-processing works in a windowed fashion, on 20ms of the audio clip at a time, and the stride
is 10ms. So, for example, if we provide one second of audio this gives us `1000ms/10ms = 100` windows of
features and, therefore, an input shape of `100x42` to the model. The output shape of the model is
then `100x22`, representing the gain values to apply to each of the 100 windows.

These output gain values can then be applied to each corresponding window of the noisy audio clip,
producing a cleaner output.

For more information please refer to the original paper:
[A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement](https://arxiv.org/pdf/1709.08243.pdf)

## Post-processing

After each inference the output of the model is passed to post-processing code which uses the gain values the model
produced to generate audio with the noise removed from it.

### Windows FVP demo and listening report

Run these commands from the repository root. The build wrapper uses the existing
Windows TFLM toolchain and the SSE-320 U85-1024 Dedicated SRAM configuration.
Use `-Action Setup` once if the Windows build tools and default model are missing.
If you already have a matching Vela model, pass it as `-ModelPath`.

```powershell
$inputWav = 'resources/noise_reduction/samples/p232_113.wav'
.\scripts\windows\build.ps1 -Profile NoiseReduction -Action Setup
.\scripts\windows\build.ps1 -Profile NoiseReduction -InputPath $inputWav
.\scripts\windows\run-fvp.ps1 -Profile NoiseReduction -Headless -OutputDirectory build/rnnoise-listening
py -3.12 scripts/py/noise_reduction_report.py --noisy $inputWav `
    --denoised build/rnnoise-listening/denoised_loop0.wav `
    --output-dir build/rnnoise-listening
Invoke-Item build/rnnoise-listening/report.html
```

The run script creates a fresh timestamped directory by default and saves the FVP
log beside its WAV files. An explicit `-OutputDirectory` must be new. Use the same
`-BuildDirectory` for both scripts if you override it.

The Python program uses the standard library and requires 48 kHz mono PCM16
inputs. The firmware build can resample other inputs. For the report, provide
the corresponding 48 kHz mono version of the same input. The report embeds the
aligned audio for same-position A/B switching and shows the waveform, short-time
RMS, overall RMS, peak and PCM rail hits. With `--clean clean.wav` it also reports
SI-SDR before and after and the improvement in `metrics.json`. The clean file
must already be aligned to the noisy input.

The default delay correction is 512 samples. RNNoise analysis places the previous
block in the first half of its window, and synthesis emits that half first.
Use `--denoised-delay-samples 0` for files that have already been aligned, or pass
a measured delay for another implementation. Measurements and playback use the
common interval. For long recordings the self-contained HTML grows with audio length.

### Evaluating a VoiceBank-DEMAND test directory

The dataset evaluator pairs files by sorted filename. Build the noisy test
directory as one firmware input set so each utterance starts with a fresh
recurrent state and exports as `denoised_loopN.wav` in the same sorted order.
For a full test set, `run-fvp.ps1 -Profile NoiseReductionDataset` builds and runs
batches, then generates the report. Supply `-DatasetDirectory` and a compatible
`-ModelPath`. Use `-Profile NoiseReductionDataset -ProfileHelp` for options.
The following commands show how to run and score one set manually. Build and
FVP output together can contain hundreds of megabytes of generated audio and artifacts.

```powershell
$dataset = 'resources_downloaded/noise_reduction'
$build = 'build/fvp-noise-reduction-voicebank'
$out = 'build/fvp-noise-reduction-voicebank/results'
.\scripts\windows\build.ps1 -Profile NoiseReduction `
    -InputPath "$dataset/noisy_testset_wav" `
    -ModelPath 'resources_downloaded/linux-batch/dedicated/noise_reduction/rnnoise_INT8_vela_Z1024_Dedicated_Sram.tflite' `
    -BuildDirectory $build
.\scripts\windows\run-fvp.ps1 -Profile NoiseReduction -BuildDirectory $build -Headless -Quiet -OutputDirectory $out
py -3.12 scripts/py/noise_reduction_dataset_report.py `
    --noisy-dir "$dataset/noisy_testset_wav" `
    --clean-dir "$dataset/clean_testset_wav" `
    --denoised-dir $out --output-dir $out/report
Invoke-Item "$out/report/dataset_report.html"
```

The evaluator verifies that all three sets have matching counts and that noisy
and clean filenames match. It reports per-utterance CSV scores and a summary
JSON and HTML histogram. The current implementation averages SI-SDR and
improvement per utterance rather than weighting longer clips more heavily.

### Saving WAV files directly from an FVP

Add these options to your MPS3 or MPS4 build configuration.

```text
-DUSE_CASE_BUILD=noise_reduction
-DML_FRAMEWORK=tflm
-DSEMIHOSTING_ENABLED=ON
-Dnoise_reduction_SAVE_WAV=ON
-Dnoise_reduction_FILE_PATH=/path/to/noisy.wav
-Dnoise_reduction_AUDIO_DURATION=0
-DINTERACTIVE_MODE=OFF
-DHAL_AUDIO_LOOP=OFF
```

Launch the rebuilt AXF from the desired host output directory. Pass
`-C cpu0.semihosting-enable=1` for MPS3 or
`-C mps4_board.subsystem.cpu0.semihosting-enable=1` for MPS4.
See [Semihosting](../sections/deployment.md#semihosting) for stack parameters.

The application writes `denoised_loop0.wav`, then increments the index for
each subsequent clip in the order shown by the `Using sample audio` logs.
It logs success after closing each file. The output is 48 kHz mono PCM16.
Files with the same names are overwritten on subsequent runs.

### Processing longer inputs

With `noise_reduction_AUDIO_DURATION=0`, the build embeds the whole WAV
after applying the configured offset and resampling. Keep the 48 kHz and
mono settings. The recurrent state persists within a clip and resets between
clips. Each inference consumes 512 samples (about 10.67 ms). The final
partial frame of up to 511 samples is discarded.

Direct WAV export writes one frame at a time. Longer clips still require
about 96 kB per second for the embedded input and take longer to simulate.
For listening tests, compare at matched playback levels. With a clean
reference, convert it to the same sample rate and align it with the output
before measuring the common interval.

### Extracting WAV files from an MPS3 memory dump

With `noise_reduction_SAVE_WAV=OFF`, MPS3 writes output to memory. After the
FVP exits and creates the dump file, extract it with Python.

```commandline
python scripts/py/rnnoise_dump_extractor.py --dump_file output.bin --output_dir denoised
```

The application for this use case has been written to dump the post-processed output to the address pointed to by
the CMake parameter `noise_reduction_MEM_DUMP_BASE_ADDR`. The default value is set to `0x80000000`.

### Dumping of memory contents from the Fixed Virtual Platform

The fixed virtual platform supports dumping of memory contents to a file. This can be done by
specifying command-line arguments when starting the FVP executable. For example, the argument:

```commandline
$ FVP_Corstone_SSE-300_Ethos-U55 -a ./bin/mlek_noise_reduction.axf \
    --dump cpu0=output.bin@Memory:0x80000000,0x100000
```

Dumps 1 MiB worth of data from address `0x80000000` to the file `output.bin`.

### Dumping post-processed results for all inferences

The Noise Reduction application uses the memory address specified by
`noise_reduction_MEM_DUMP_BASE_ADDR` as a buffer to store post-processed results from all inferences.
The maximum size of this buffer is set by the parameter
`noise_reduction_MEM_DUMP_LEN` which defaults to 1 MiB.

Each frame uses 1024 PCM bytes. The default 1 MiB buffer holds just under
10.93 seconds of output across all clips, including headers and the end marker.
To capture more, increase both `noise_reduction_MEM_DUMP_LEN` and the length
in the FVP `--dump` argument. Ensure that the entire range is mapped and reserved.

## Prerequisites

See [Prerequisites](../documentation.md#prerequisites)

## Building the code sample application from sources

### Build options

In addition to the already specified build option in the main documentation, keyword spotting use
case adds:

- `noise_reduction_MODEL_PATH` - The path to the NN model file in *TFLite* format. The model
  is processed and is included in the application axf file. The default value points to one of the
  delivered set of models. Note that the parameter
  `ETHOS_U_NPU_ENABLED` must be aligned with the chosen model. Therefore:
  - if `ETHOS_U_NPU_ENABLED` is set to `On` or `1`, we assume that the NN model is optimized. The
    model naturally falls back to the Arm® Cortex®-M CPU if an unoptimized model is supplied.
  - if `ETHOS_U_NPU_ENABLED` is set to `Off` or `0`, then we assume that the NN model is unoptimized.
    In this case, supplying an optimized model results in a runtime error.

- `noise_reduction_FILE_PATH`: The path to the directory containing WAV files, or a path to single
  WAV file, to be used in the application. The default value points to the
  `resources/noise_reduction/samples` folder containing the delivered set of audio clips.

- `noise_reduction_AUDIO_RATE`: The input data sampling rate. Each audio file from `noise_reduction_FILE_PATH` is
  preprocessed during the build to match the NN model input requirements. The default value is `48000`.

- `noise_reduction_AUDIO_MONO`: If set to `ON`, then the audio data is converted to mono. The default value is `ON`.

- `noise_reduction_AUDIO_OFFSET`: Begins loading audio data and starts from this specified offset, defined in seconds.
  The default value is set to `0`.

- `noise_reduction_AUDIO_DURATION`: The length of the audio data to be used in the application in seconds.
  The default is `0`, meaning that the whole audio file is used.

- `noise_reduction_AUDIO_MIN_SAMPLES`: Minimum number of samples required by the network model. If the audio clip is shorter than
  this number, then it is padded with zeros. The default value is `480`.

- `noise_reduction_ACTIVATION_BUF_SZ`: The intermediate, or activation, buffer size reserved for the
  neural network model. By default, it is set to 2MiB.

To **ONLY** build a `noise_reduction` example application, add `-DUSE_CASE_BUILD=noise_reduction`
  (as specified in [Building](../documentation.md#Building) to the `cmake` command line).

### Build process

> **Note:** This section describes the process for configuring the build for `MPS3: SSE-300`. To
> configure a different target platform, please see the [Building](../documentation.md#Building)
> section.

To **only** build the `noise_reduction` example, create a build directory, and then navigate inside.
For example:

```commandline
mkdir build_noise_reduction && cd build_noise_reduction
```

On Linux, when providing only the mandatory arguments for CMake configuration, use the following
command to build the Noise Reduction application to run on the *Ethos-U55* Fast Model:

```commandline
cmake ../ -DUSE_CASE_BUILD=noise_reduction
```

To configure a build that can be debugged using Arm DS, we specify the build type as `Debug` and use
the `Arm Compiler` toolchain file:

```commandline
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=scripts/cmake/toolchains/bare-metal-armclang.cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DUSE_CASE_BUILD=noise_reduction
```

For more notes, please refer to:

- [Configuring with custom TPIP dependencies](../sections/building.md#configuring-with-custom-tpip-dependencies)
- [Using Arm Compiler](../sections/building.md#using-arm-compiler)
- [Building for different Ethos-U variants](../sections/building.md#building-for-different-ethos_u-npu-variants)

> **Note:** If you are rebuilding with changed parameters values, it is highly advised that you
> clean the build directory and rerun the CMake command.

If the CMake command is successful, then build the application as follows:

```commandline
make -j4
```

> **Note:** To see compilation and link details, add `VERBOSE=1`.

The build results are placed under the `build/bin` folder. For example:

```tree
bin
 ├── mlek_noise_reduction.axf
 ├── mlek_noise_reduction.htm
 ├── mlek_noise_reduction.map
 ├── images-noise_reduction.txt
 └── sectors
      └── noise_reduction
           ├── dram.bin
           └── itcm.bin
```

Based on the preceding output, the files contain the following information:

- `mlek_noise_reduction.axf`: The built application binary for the noise reduction use case.

- `mlek_noise_reduction.map`: Information from building the application (for example. The
  libraries used, what was optimized, and location of objects).

- `mlek_noise_reduction.htm`: A human readable file containing the call graph of application
  functions.

- `sectors/`: This folder contains the built application, which is split into files for loading into
  different FPGA memory regions.

- `Images-noise_reduction.txt`: Tells the FPGA which memory regions to use for loading the binaries
  in the `sectors/...` folder.

### Add custom input

To run with inputs different to the ones supplied, the parameter `noise_reduction_FILE_PATH` can be
pointed to a WAV file, or a directory containing WAV files. Once you have a directory with WAV files,
run the following command:

```commandline
cmake .. \
    -DUSE_CASE_BUILD=noise_reduction \
    -Dnoise_reduction_FILE_PATH=/path/to/custom/wav_files
```

### Add custom model

The application performs inference using the model pointed to by the CMake parameter
`noise_reduction_MODEL_PATH`.

> **Note:** If you want to run the model using *Ethos-U* ensure that your custom model has been
> run through the Vela compiler successfully before continuing.

For further information: [Optimize model with Vela compiler](../sections/building.md#optimize-custom-model-with-vela-compiler).

An example:

```commandline
cmake .. \
    -Dnoise_reduction_MODEL_PATH=<path/to/custom_model_after_vela.tflite> \
    -DUSE_CASE_BUILD=noise_reduction
```

> **Note** Changing the neural network model often also requires the pre-processing implementation
> to be changed. Please refer to:
> [How the default neural network model works](./noise_reduction.md#how-the-default-neural-network-model-works).
>
> **Note:** Before re-running the CMake command, clean the build directory.

The `.tflite` model file, which is pointed to by `noise_reduction_MODEL_PATH`, is converted
to C++ files during the CMake configuration stage. It is then compiled into the application for
performing inference with.

To see which model path was used, inspect the configuration stage log:

```log
-- User option noise_reduction_MODEL_PATH is set to <path/to/custom_model_after_vela.tflite>
...
-- Using <path/to/custom_model_after_vela.tflite>
++ Converting custom_model_after_vela.tflite to custom_model_after_vela.tflite.cc
-- Generating labels file from <path/to/labels_custom_model.txt>
-- writing to <path/to/build/generated/src/Labels.cc>
...
```

After compiling, your custom model replaces the default one in the application.

## Setting up and running Ethos-U NPU code sample

### Setting up the Ethos-U NPU Fast Model

The FVP is available publicly from [Arm Ecosystem FVP downloads](https://developer.arm.com/tools-and-software/open-source-software/arm-platforms-software/arm-ecosystem-fvps).

For the *Ethos-U* evaluation, please download the MPS3 based version of the Arm® *Corstone™-300* model that contains *Cortex-M55*
and offers a choice of the *Ethos-U55* and *Ethos-U65* processors.

To install the FVP:

- Unpack the archive,

- Run the install script in the extracted package:

```commandline
$./FVP_Corstone_SSE-300.sh
```

- Follow the instructions to install the FVP to your required location.

### Starting Fast Model simulation

Once the building step has completed, the application binary `mlek_noise_reduction.axf` can be
found in the `build/bin` folder. Assuming the install location of the FVP was set to
`~/FVP_install_location`, start the simulation with the following command:

```commandline
~/FVP_install_location/models/Linux64_GCC-6.4/FVP_Corstone_SSE-300_Ethos-U55 ./bin/mps3-sse-300/mlek_noise_reduction.axf
```

A log output then appears on the terminal:

```log
telnetterminal0: Listening for serial connection on port 5000
telnetterminal1: Listening for serial connection on port 5001
telnetterminal2: Listening for serial connection on port 5002
telnetterminal5: Listening for serial connection on port 5003
```

This also launches a telnet window with the standard output of the sample application. It also
includes error log entries containing information about the pre-built application version,
TensorFlow Lite Micro library version used, and the data type. As well as the input and output
tensor sizes of the model that was compiled into the executable binary.

After the application has started inferences are executed on inputs from `noise_reduction_FILE_PATH`

### Running Noise Reduction

The following example illustrates profiling output:

```log
INFO - Final results:
INFO - Profile for Inference:
INFO - NPU AXI0_RD_DATA_BEAT_RECEIVED beats: 530
INFO - NPU AXI0_WR_DATA_BEAT_WRITTEN beats: 376
INFO - NPU AXI1_RD_DATA_BEAT_RECEIVED beats: 13911
INFO - NPU ACTIVE cycles: 103870
INFO - NPU IDLE cycles: 643
INFO - NPU TOTAL cycles: 104514
```

> **Note:** When running Fast Model, each inference can take several seconds on most systems.

Each inference dumps the post processed output to memory. For further information, please refer to:
[Dumping post processed results for all inferences](./noise_reduction.md#dumping-post_processed-results-for-all-inferences).

The profiling section of the log shows that for this inference:

- *Ethos-U* NPU PMU report for each inference:

  - 104514: The total number of NPU cycles.

  - 103870: How many NPU cycles were used for computation.

  - 643: How many cycles the NPU was idle for.

  - 530: The number of AXI beats with read transactions from AXI0 bus.
    > **Note:** The AXI0 is the bus where the *Ethos-U* NPU reads and writes to the computation
    > buffers, or the activation buf or tensor arenas.

  - 370: The number of AXI beats with write transactions to the AXI0 bus.

  - 13911: The number of AXI beats with read transactions from AXI1 bus.
    > **Note:** The AXI1 is the bus where *Ethos-U* NPU reads the model, which is read-only.

- For FPGA platforms, the CPU cycle count can also be enabled. However, for FVP, do not use the CPU
  cycle counters as the CPU model is not cycle-approximate or cycle-accurate.
