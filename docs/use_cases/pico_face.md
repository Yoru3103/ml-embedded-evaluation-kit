# Pico-face on MLEK

Pico-face generates a 128 x 128 RGB face with a latent rectified-flow transformer
and a convolutional VAE decoder. This integration supports **Corstone-320,
Ethos-U85 Z1024, Dedicated_Sram with 384 KiB fast cache, and Shared_Sram**, using
the normal MLEK CMake/HAL/driver path. ATFE builds have been validated on the
Dedicated FVP and on physical FPGA hardware in both memory modes.

A [Chinese quick start and explanation](#中文说明与简短复现流程) follows the English documentation.

## Learning material

Read the [Hugging Face diffusion introduction](https://huggingface.co/learn/diffusion-course/en/unit1/1),
then the [VAE/latent diffusion explanation](https://huggingface.co/learn/diffusion-course/en/unit3/2),
then the [DiT authors' project page](https://www.wpeebles.com/DiT).
Pico-face uses rectified-flow velocity prediction and Euler integration rather
than the traditional noise-prediction sampler used in some introductory lessons.

## Build and run on Windows

Run from the MLEK root. The existing sibling checkout supplies the prepared
export Python environment and checkpoint files. It is read only during export;
model export outputs go into this repository's ignored resources directory.
Firmware builds use MLEK's own ExecuTorch environment, ATFE, CMake and Ninja.
Checkpoint acquisition, export-environment setup and the initial standalone
reproduction were completed in the sibling CMSIS-Executorch project before this
port. MLEK now contains the adapted model/export code and firmware integration,
but does not provision that export environment or download checkpoints.
`Export` runs MLEK's Python exporter using the sibling interpreter and checkpoint
files. `Build` consumes the generated PTE, parameter header and manifest without
loading checkpoints or re-exporting the model.
The default headless simulator is the sibling project's MDK FVP 11.32.23.
The installed Ecosystem FVP 11.27 generates the correct image but has reproduced
a Windows abnormal exit after simulation; the runner does not ignore that exit.

```powershell
# Export once: a16w8 DiT, a8w8 decoder, 16 calibration trajectories.
.\scripts\windows\build.ps1 -Profile PicoFace -Action Export

# Build the FVP firmware; no LCD initialization or framebuffer accesses.
.\scripts\windows\build.ps1 -Profile PicoFace -Action Build `
    -CompilerBin 'C:\Program Files\Arm\Development Studio 2026.0-1\sw\atfe-pro22.1.0\bin'

# Generate seed 3 / four steps / class 1 / guidance 4, save RGB and PNG.
.\scripts\windows\run-fvp.ps1 -Profile PicoFace
```

After exporting, build with the same model variant. `-Toolchain GNU` selects
the GNU toolchain path;
the validation reported here uses ATFE 22.1.0, not GNU. Override `-ExportPython`, `-Checkpoints`,
`-CompilerBin`, `-BuildDirectory` or `-FvpExecutable` for another workstation.
The checkpoint directory must contain `m3_long_cfg/{dit_qat.pt,vae_final.pt,latent_stats.npz}`.
The export environment needs the Arm backend dependencies used by the pinned
CMSIS-ExecuTorch 1.5.1 example (including TOSA tools and Vela).

Set the firmware defaults with `-Seed`, `-Steps` (1/2/4/8), `-ClassId` (0..3 or
unconditional 4) and `-Guidance` on **Build**. The FVP launcher with `-Preview`
also accepts runtime `-Seed` and `-ClassId` overrides.
Default output directory:

```text
build/pico_face/pte/atfe/fvp/dedicated_sram/384k-release-default-pmu/
  bin/mlek_pico_face.axf
  build-info.json
  pico_face.rgb
  pico_face.png
build/logs/pico_face_logs/
  build/  # Timestamped build/export logs
  fvp/    # Timestamped FVP execution logs
  fpga/   # Board UART logs
  out/    # Received images
```

The runner requires `PICO_FACE_PASS`, a successful simulator exit, and exactly
49,152 output bytes. It removes an earlier RGB result before running. Headless
FVP is the supported mode of this initial integration; the shared MPS4 display
option defaults to ON for existing use cases and is explicitly OFF here.

## Source and data ownership

* `source/lib/mlek/fwk/executorch/EtProgram.*`: owns one program/data loader and
  persistent method states. Each method owns its span descriptors, planner and
  memory manager; method allocations use a caller-supplied pool. Execution is
  synchronous, with the temporary allocator shared between methods.
* `source/lib/mlek/use_case/pico_face/`: framework-independent noise generation,
  guidance, Euler sampling, RGB conversion and CRC. No HAL dependencies.
* `source/app/use_case/pico_face/`: ExecuTorch tensor adapter, shape checks,
  per-call PMU records, end-to-end timing, UART image service and optional
  semihosted RGB output. The debugger-visible
  `arm::app::pico_face::g_image` also holds the result; its address is printed in
  the log and appears in the linker map (Release ELF symbols may be stripped).
* `scripts/py/mlek_tools/pico_face/`: model/checkpoint loading and strict
  two-method export, adapted from Arm CMSIS-ExecuTorch commit
  `0d600669b83f2daf49e1a30b9ec37027c6446030`. Model reimplementation retains
  the upstream pico-faces MIT notice in `LICENSE-pico-faces`.
* `resources_downloaded/pico_face/u85-1024-dedicated-384k/`: generated PTE,
  matching condition table and manifest with identities, versions and hashes.

The PTE exposes `dit_step(z[1,8,16,16], c[1,128]) -> v[1,8,16,16]` and
`decode(z[1,8,16,16]) -> image[1,3,128,128]`. Each has one NPU delegate;
float/integer boundary operations run on the CPU. Sampling runs on the CPU.
Guided four-step generation executes eight DiT calls and one final decode.
Each guided step evaluates the same DiT twice with the same latent and timestep:
once with the requested class, and once with the null class. The CPU combines
their velocities as `v = v_uncond + guidance * (v_cond - v_uncond)` before making
one Euler update. These are two evaluations of one model, not two sampling steps.
The optional eight-step FVP preview also decodes the intermediate latents.
The FVP preview accepts a runtime seed and class. Steps and guidance are set at
build time. The UART service can retrieve the stored image or exit.

## Memory and platform constraints

* Model array: `nn_model`, mapped to DDR by the existing linker script.
* Method pool: 4 MiB capacity in `activation_buf_dram`, DDR. This includes each
  method's planned buffers; planned byte counts are not total runtime usage.
* Sampler state and final RGB: `activation_buf_dram`, DDR, not the main stack.
* Dedicated temporary pool: 4 MiB capacity at `0x94000000` on FVP or
  `0x72000000` on FPGA. Shared uses a 2 MiB SRAM temporary pool at `0x31000000`.
  The NPU backend allocates scratch from the temporary pool.
* Dedicated fast cache: 393,216 bytes in the existing NPU SRAM section. The compiler
  `--arena-cache-size` and firmware `ETHOS_U_NPU_CACHE_SIZE` must match.
* No display buffer is used. `MPS4_DISPLAY_ENABLED=OFF` selects display stubs
  and skips the platform HDLCD initialization. Register addresses and NPU/TA
  initialization routines are unchanged.

Inspect the ELF map and pool high-water logs before reducing capacities.
The Dedicated FVP temporary address is not an FPGA deployment address. The FPGA
build uses the existing platform's DDR or SRAM temporary region, with a matching
PTE/cache configuration for the selected memory mode. Do not place the method
pool in SRAM merely because MLEK calls its enclosing buffer an activation buffer.

## Algorithm tests

These tests use an analytic constant-velocity model, require no checkpoints and
do not execute NPU code. They check guidance, integration sign, state reset,
call counts, RGB channel layout/clamping/rounding and the standard CRC vector.

```sh
cmake -S tests/processing/pico_face -B build/pico-face-processing-tests
cmake --build build/pico-face-processing-tests -j4
ctest --test-dir build/pico-face-processing-tests --output-on-failure
```

Compare the default generated image against the previously validated host
fake-quant reference in the sibling checkout:

```powershell
$env:PYTHONPATH = 'scripts/py'
.\resources_downloaded\windows-executorch\env\Scripts\python.exe `
    -m mlek_tools.pico_face.verify `
    build/pico_face/pte/atfe/fvp/dedicated_sram/384k-release-default-pmu/pico_face.rgb `
    ../arm-pico-face/CMSIS-Executorch/out/pf_fakequant.png
```

Use a reference generated with matching seed, class, step count, guidance,
checkpoints and quantization settings when changing the defaults.

## Validation (2026-10-22)

ATFE 22.1.0 builds successfully. The algorithm test passes with GCC 13.3 in WSL.
MDK FVP 11.32.23 exits with code zero and prints `PICO_FACE_PASS`. The generated
RGB has 49,152 bytes and CRC32 `5557eceb`, identical to the standalone example.
Against its host fake-quant reference, PSNR is 47.98 dB (threshold 35 dB).
Observed method-pool peak is 263,724 bytes; temporary-pool peak is 524,288 bytes.
These figures describe the allocator pools, not total firmware RAM use. The map
places cache at `0x31000000` (384 KiB), RGB at `0x702a8860`, sampler at
`0x702b4860`, and method pool at `0x702ba860` for Dedicated mode.

Physical FPGA runs supplied by the user validate both memory modes with seed 3,
8 Euler steps, class 1 and guidance 4. Both execute 16 DiT calls and one decode,
print `PICO_FACE_PASS` and produce the same RGB CRC32, `0223caac`.

| FPGA measurement | Dedicated, 384 KiB cache | Shared |
|---|---:|---:|
| Generation, initial noise through final RGB | 798.981 ms | 804.300 ms |
| First image including model initialization | 837.499 ms | 842.801 ms |
| DiT mean NPU TOTAL, cycles per call | 1,890,508.06 | 1,909,315.63 |
| DiT mean CPU TOTAL, cycles per call | 2,274,849.00 | 2,292,383.75 |
| Method-pool peak | 263,724 bytes | 263,724 bytes |
| Temporary-pool peak | 524,288 bytes | 527,056 bytes |

Means include all 16 conditional/unconditional DiT calls and exclude decode.
CPU TOTAL measures the method call including the NPU wait, so CPU and NPU
totals must not be added. Times use the existing 50 MHz FPGA clock fallback.
These measurements describe the current model, memory setup and timing-adapter
configuration; they are not a general ranking of the two memory modes.
The FPGA UART image capture also passes CRC verification (`0223caac`).

## FPGA board files: Dedicated and Shared SRAM

The same use case can be built for the existing MLEK Corstone-320 FPGA platform.
This generates board files only; it does not connect to or program a board.
Dedicated reuses the FVP-validated U85 Z1024 / 384 KiB PTE. Shared requires a
separate PTE exported with `Shared_Sram`, without an arena-cache-size flag.

```powershell
# Dedicated model is already available after the FVP reproduction.
.\scripts\windows\build.ps1 -Profile PicoFace -Action Build -Target FPGA -MemoryMode Dedicated_Sram

# Export Shared once, then build its FPGA firmware.
.\scripts\windows\build.ps1 -Profile PicoFace -Action Export -MemoryMode Shared_Sram
.\scripts\windows\build.ps1 -Profile PicoFace -Action Build -Target FPGA -MemoryMode Shared_Sram
```

Both builds use ATFE by default, Release optimization, UART 115200 8N1,
`FPGA_PLATFORM_SSE_320=ON`, semihosting OFF, image-file saving OFF and display OFF.
They reuse the existing FPGA UART, NPU, timing-adapter and startup definitions.

| Memory | Dedicated | Shared |
|---|---|---|
| Model and 4 MiB method pool | DDR | DDR |
| Temp pool base | `0x72000000` (FPGA platform DDR) | `0x31000000` (SRAM) |
| Temp capacity | 4 MiB | 2 MiB |
| NPU fast cache | 384 KiB SRAM | None |
| Framebuffer | None | None |

The exporter records per-method NPU scratch sizes in `manifest.json`. The Shared
builder rejects a model exceeding the 2 MiB temporary budget. Methods execute
sequentially and the backend uses the runtime temporary allocator, whose
allocations are released after delegate execution; the actual peak must still
be checked on hardware. These budgets follow the existing QR/YOLO DDR-method
and SRAM-temp layout. No register definitions are changed for this addition.

Outputs are separate from the FVP firmware:

```text
build/pico_face/pte/atfe/fpga/
  dedicated_sram/384k-release-default-pmu/
  shared_sram/no-cache-release-default-pmu/
```

Each contains `bin/mlek_pico_face.axf`, a linker map, the original
`bin/sectors/pico_face/{boot.bin,bram.bin,ddr.bin}` and a convenience package:

```text
board-package/
  SOFTWARE/boot.bin
  SOFTWARE/bram.bin
  SOFTWARE/ddr.bin
  images.txt
  mlek_pico_face.axf
  mlek_pico_face.map
  build-info.json
  model-manifest.json
  SHA256SUMS.txt
  README.txt
pico-face-fpga-dedicated-384k.zip     # or pico-face-fpga-shared.zip
```

`images.txt` retains the standard loader addresses: boot `0x11000000`, BRAM
`0x12000000`, DDR `0x70000000`. To deploy, place the three binaries
in the board's `SOFTWARE` directory and install `images.txt` in its active
Corstone-320 configuration directory, following the already verified use cases.
The repository's revision-B/FI101 example uses `MB/HBI0376B/FI101/images.txt`;
use the path appropriate to the actual board. Do not substitute the FVP ELF.

For packaging an existing successful FPGA build without rebuilding:

```powershell
.\scripts\windows\build.ps1 -Profile PicoFace -Action Package -Target FPGA -MemoryMode Dedicated_Sram
.\scripts\windows\build.ps1 -Profile PicoFace -Action Package -Target FPGA -MemoryMode Shared_Sram
```

On-board success is indicated by UART `PICO_FACE_PASS`; the RGB buffer address
and CRC are printed before it. Retrieve 49,152 bytes from that address with the
debugger to view the image, or enable the UART image service described below.
Both memory modes have now passed physical inference; see the validation table.

Both FPGA variants were built with ATFE 22.1.0 on 2026-10-22. File inspection
confirmed that the three packaged binaries exactly match their selected ELF
sections, the loader addresses match `images.txt`, and the package SHA-256 and
ZIP contents agree. Register configuration matches the existing FPGA profile;
the UART `_exit` implementation is linked instead of semihosted exit.
The initial packages have `fpga-artifact-validation.json`; UART-enabled packages
have `fpga-uart-artifact-validation.json` in their build directories.

The inspected seed-3, 8-step UART packages contain:

| Artifact | Dedicated bytes | Shared bytes |
|---|---:|---:|
| `boot.bin` | 1,984 | 1,984 |
| `bram.bin` | 412,840 | 412,840 |
| `ddr.bin` | 7,055,456 | 7,054,832 |
| RGB buffer address | `0x702a8860` | `0x702a85f0` |

The Shared PTE is 2,786,800 bytes. Its Vela scratch requirements are 284,416 bytes
for DiT and 527,056 bytes for decode, below the 2 MiB SRAM budget. Method pools
are at `0x702ba860` (Dedicated) and `0x702ba5f0` (Shared), both in DDR.
The physical runs confirm the corresponding runtime pool peaks shown above.

## UART image retrieval and end-to-end timing

Build with `-UartImage` to keep the final image available on the application UART:

```powershell
.\scripts\windows\build.ps1 -Profile PicoFace -Action Build -Target FPGA -MemoryMode Dedicated_Sram -UartImage
.\scripts\windows\build.ps1 -Profile PicoFace -Action Build -Target FPGA -MemoryMode Shared_Sram -UartImage
```

The archives have suffix `-uart.zip`; install those versions to use the receiver.
Non-default generation settings add a seed/steps/class/guidance suffix to the
archive name. The `board-package` directory contains the most recently packaged
configuration; check its `README.txt` or `build-info.json` before deployment.
UART remains 115200 8N1 with flow control OFF. After generation the firmware prints
`PICO_FACE_PASS` and `SERIAL_IMAGE_READY`, then waits for commands. `I` retrieves
the stored image without running the model again; `Q` leaves the service.

After observing `SERIAL_IMAGE_READY`, close the terminal holding the application
COM port and run the following from the MLEK root (replace COM7 with that port):

```powershell
.\scripts\windows\receive-pico-face.ps1 -ListPorts
.\scripts\windows\receive-pico-face.ps1 -Port COM7
```

The Windows receiver uses .NET serial I/O and the existing Python/Pillow
environment, with no additional serial package required. It saves
`build/logs/pico_face_logs/out/face_seed<seed>_steps<steps>_<timestamp>.pfframe`
(binary capture) and, after CRC verification, `.rgb`, `.png`, and `.json`.
The automatic name preserves previous runs. `-Output face` and the legacy
`-Output out/face` both save under this directory, never under the repository
root. Absolute output paths must also stay inside this directory.
To receive from startup, close the
terminal, start the receiver with `-WaitReady`, then reset the board. Do not use
`-WaitReady` for an already completed generation whose ready message has passed.
Retrieval can be repeated. Receiving a 49,152-byte image takes about 4.3 seconds
at 115200 baud, plus USB and host overhead.

The existing UART text writer expands LF to CRLF. The new HAL raw writer sends
unmodified bytes using the same CMSDK DATA/STATE registers as that driver,
without changing UART CTRL, baud, NPU, TA or memory definitions. All application
text is flushed before the binary frame, and no logs are emitted inside it.
The wire format is a 56-byte little-endian header (`<8s8IfIQ`) followed by RGB:

```text
magic[8] = PFIMAGE1
uint32: version=1, width=128, height=128, channels=3, payload_bytes=49152,
        seed, steps, class_id
float32: guidance
uint32: RGB CRC32
uint64: generation_us
uint8:  RGB888 HWC payload[49152]
```

The Python equivalent is `python -m mlek_tools.pico_face.receive --port COM7` with
`PYTHONPATH=scripts/py`; that optional cross-platform serial entry needs pyserial.
Offline captures can be checked with the Windows wrapper's `-Capture` option.

Timing now uses the existing SysTick CPU counter continuously: per-method NPU
PMU initialization no longer resets the CPU timer. Other use cases retain their
existing profiling behavior. Timing is printed as separate profiles, each with
`CPU TOTAL: <value> cycles` and `DURATION: <value> milliseconds`:

* Model initialization: PTE/method setup, including initialization logging.
* Image generation (noise to RGB): initial noise, all Euler steps and guidance
  calls, final decode and RGB conversion. Includes profiling instrumentation;
  excludes model setup, CRC/log output, image saving and UART transfer.
* First image: MainLoop entry through RGB ready, including setup and preceding
  application logs; excludes platform boot.

Every DiT call has its own profile labeled with the step and conditional or
unconditional branch. Decode has a separate profile. Counters use one value per
line, as in the other use cases, without total/average/minimum/maximum columns.
Records are printed after generation timing stops so UART output is excluded.
`Timing core clock` reports the frequency used for conversion. `enqueue_us`
measures UART frame enqueue time separately; it excludes host PNG saving and the
final shift-register drain.

The supported Euler schedules are:

| Steps | Time points (ending at zero) | DiT calls with guidance | Final decode calls |
|---:|---|---:|---:|
| 1 | 1 -> 0 | 2 | 1 |
| 2 | 1 -> 0.5 -> 0 | 4 | 1 |
| 4 | 1 -> 0.75 -> 0.5 -> 0.25 -> 0 | 8 | 1 |
| 8 | 1 -> 0.875 -> ... -> 0.125 -> 0 | 16 | 1 |

Without guidance, DiT is called once per step. Four steps is the reproduction
baseline; try eight as a quality/performance comparison with the same seed and
class. More steps change the integration approximation and do not guarantee a
better result for every seed. One/two steps are useful for latency experiments.
The table and conditions support only 1/2/4/8; arbitrary step counts need a new
schedule/conditioning setup. Steps can be changed at build time without exporting
the existing PTE again:

```powershell
.\scripts\windows\build.ps1 -Profile PicoFace -Action Build -Target FPGA -MemoryMode Dedicated_Sram -UartImage -Steps 8
```

The two native CTest targets include sampling tests and eight Python tests:
five cover framing, fragmented reads, CRC rejection, truncated/invalid payloads
and actual C++-encoder/Python-decoder interoperability; three cover output paths.
The C++ interoperability test needs the built fixture and skips when it is not
provided. FPGA UART image reception has also been confirmed on hardware.

Build/export logs are saved automatically under `build/logs/pico_face_logs/build`.
FVP run logs are saved under `build/logs/pico_face_logs/fvp`. These directories
sit alongside `fpga` (board logs) and `out` (received images). Each new build or
FVP log includes its target, memory variant, seed, steps and timestamp.

The UART-enabled Dedicated and Shared archives were rebuilt and their ELF
sections, ZIP contents and hashes were inspected successfully. See
`fpga-uart-artifact-validation.json` in each build directory. The updated FVP
run exits normally with 8 DiT calls, 1 decode, CRC `5557eceb` and 47.98 dB PSNR.
The native sampling test and all five serial interoperability tests pass; the
Windows capture-to-PNG wrapper also passes an offline frame smoke test.

Clock caveat for this MDK FVP distribution: the existing SCC mapping reports
17,039,392 Hz instead of the stock startup clock. That simulation confirms image
correctness and counter continuity, but its converted microseconds are not a
calibrated performance measurement. The FPGA log supplied by the user uses the
existing FI101 50 MHz fallback. No clock register definitions or selection logic
were changed here; read `Timing core clock` alongside the on-board timing results.

Host seed/schedule comparisons can be generated without changing the PTE:

```powershell
$env:PYTHONPATH = "$PWD/scripts/py"
& ../arm-pico-face/CMSIS-Executorch/.venv/Scripts/python.exe -m mlek_tools.pico_face.compare `
    --checkpoints ../arm-pico-face/CMSIS-Executorch/model/pico_faces `
    --vela-config scripts/vela/default_vela.ini
```

Outputs default to `build/logs/pico_face_logs/out/comparison`. The contact sheet
compares seeds 1, 2, 3, 7 and 11 at 4/8 steps, with class 1 and guidance 4 fixed.
These are calibrated host fake-quant graphs, not NPU execution or FPGA timings.
The seed-3 float/fake-quant sheet helps distinguish quantization changes from
features already present in the float model. More iterations can change edges,
texture and identity; they do not guarantee sharper or better images.

## 中文说明与简短复现流程

### 功能与已验证范围

pico-face 是生成 128×128 RGB 人脸的用例，不需要输入照片。它从随机噪声开始，
用 DiT 多次预测 latent 的更新方向，经 guidance 和 Euler 更新后，最后调用一次
VAE decoder 得到图像。一个 PTE 包含两个可分别调用的 method：

| Method | 输入 | 输出 | 作用 |
|---|---|---|---|
| `dit_step` | latent `[1,8,16,16]`、条件 `[1,128]` | velocity `[1,8,16,16]` | 一次更新方向预测 |
| `decode` | 最终 latent `[1,8,16,16]` | 图像 `[1,3,128,128]` | 转换为 128×128 图像 |

DiT 使用 a(ctivate)16w(eight)8，decoder 使用 a8w8；method 边界的量化/反量化，以及 guidance、
Euler 更新和 RGB 转换在 CPU 上执行。开启 guidance 时，每步调用两次 DiT；
4 步是 8 次 DiT，8 步是 16 次 DiT，最终均只 decode 一次。

两次调用使用同一个 DiT、相同 latent 和时间步，区别在于类别条件：

* `conditional`：使用目标类别，例如 class 1（笑脸女性），预测符合该条件的方向 `v_cond`。
* `unconditional`：使用空类别（class 4），预测不指定目标类别的参考方向 `v_uncond`。

CPU 用 `v = v_uncond + guidance × (v_cond - v_uncond)` 加强目标类别的引导，
然后只做一次 Euler 更新。因此，两次 DiT 是同一步内的两次方向预测，不是连续更新两步。

已验证 ATFE 22.1.0、Corstone-320、Ethos-U85 Z1024：Dedicated 模式已通过 FVP，
Dedicated 384 KiB 和 Shared 模式均已通过 FPGA 实板推理。GNU 仅提供构建入口，
本文不声明已完成 GNU 验证。当前 FPGA 不使用显示设备，也不分配 framebuffer。

### 环境准备

下面命令均在 MLEK 仓库根目录执行，基于当前已准备好的工作站环境：

* 固件使用 MLEK 的 ExecuTorch、CMake、Ninja 环境和独立 ATFE 编译链。
* 导出默认使用同级 `../arm-pico-face/CMSIS-Executorch/.venv`，checkpoint 位于该工程的
  `model/pico_faces/m3_long_cfg/`，包含 `dit_qat.pt`、`vae_final.pt` 和 `latent_stats.npz`。
* FVP 默认使用同级工程的 MDK FVP 11.32.23。更换工作站时通过 `-ExportPython`、
  `-Checkpoints`、`-CompilerBin`、`-FvpExecutable` 指定实际位置。

当前工程的准备与迁移边界如下：

| 阶段 | 位置与已完成的工作 |
|---|---|
| 原工程准备 | `C:\Users\MC\Documents\mlek-client\arm-pico-face\CMSIS-Executorch`：此前已运行 `py -3.12 setup_venv.py`，准备导出环境与 checkpoint，并完成独立工程的模型导出和 FVP 复现。 |
| 迁入 MLEK | 移入并适配模型定义、量化校准与导出逻辑，新增双 method 运行、采样、构建打包、计时和串口读图，接入 MLEK 的 CMake、HAL 和驱动。 |
| 当前 `Export` | 执行 MLEK 内的 `mlek_tools.pico_face.export`，仍复用原工程的 Python 环境和 checkpoint；按本工程 U85 Z1024、内存模式和 cache 配置生成模型。 |
| 当前 `Build` | 读取本工程已有的 PTE、参数表和 manifest，检查配置及哈希后编译固件；不读取 checkpoint，也不自动重新导出。 |

导出产物位于 MLEK 的 `resources_downloaded/pico_face/u85-1024-dedicated-384k/`
或 `u85-1024-shared/`，每组包含 `pico_face.pte`、`PicoFaceParams.hpp`、`manifest.json`。
三者需要配套保留，不能只拷贝一个 PTE，也不能混用不同内存模式的产物。
现有产物可直接用于 `Build`；下文的 `Export` 命令用于首次生成或重新导出。

checkpoint 获取和导出环境安装尚未纳入 MLEK 的自动资源准备流程；单独 clone MLEK
不会自动获得它们。若要脱离原工程导出，需要自行准备兼容的 Python 环境与上述三个
checkpoint 文件，并用 `-ExportPython`、`-Checkpoints` 指定位置。
checkpoint、导出环境和生成的 PTE/参数表/manifest 均不随本次 feature commit 入库。

### 复现 FVP：Dedicated、4 步

```powershell
# 首次导出；后续只改 seed 或支持的步数时不需要重新导出。
.\scripts\windows\build.ps1 -Profile PicoFace -Action Export -MemoryMode Dedicated_Sram

# 构建 seed 3、4 步、class 1、guidance 4 的固件。
.\scripts\windows\build.ps1 -Profile PicoFace -Action Build -Target FVP `
    -MemoryMode Dedicated_Sram -Seed 3 -Steps 4 -ClassId 1 -Guidance 4

# 运行已经构建的固件。
.\scripts\windows\run-fvp.ps1 -Profile PicoFace -Target FVP -MemoryMode Dedicated_Sram
```

成功标志为 `PICO_FACE_PASS`、模拟器正常退出，并生成 49,152 字节 RGB。
此配置的已验证 CRC32 为 `5557eceb`。图片在固件构建目录：

```text
build/pico_face/pte/atfe/fvp/dedicated_sram/384k-release-default-pmu/
  pico_face.rgb
  pico_face.png
```

FVP 用于验证功能和数值；当前 MDK FVP 的时钟寄存器读数存在差异，性能时间以实板为准。

### FVP 八步逐帧演示

复用 Dedicated 384 KiB PTE。预览固件默认单独构建到带 `-preview` 后缀的目录。
运行使用 FVP 的 functional fast 模式，预览固件禁用 timing adapter。
主机界面使用支持 Tkinter 的 Python 3，可通过 `-PreviewPython` 指定解释器。

```powershell
.\scripts\windows\build.ps1 -Profile PicoFace -Action Build -Preview
.\scripts\windows\run-fvp.ps1 -Profile PicoFace -Preview
.\scripts\windows\run-fvp.ps1 -Profile PicoFace -Preview -Seed 7 -ClassId 3

# 无界面执行一轮，检查九帧、CRC 和 FVP 正常退出。
.\scripts\windows\run-fvp.ps1 -Profile PicoFace -Preview -Headless
```

`-Action Run -Preview -Seed` 可在运行时选择 0 到 4294967295 的 seed，
`-ClassId` 可在运行时选择 0（女性、中性）、1（女性、微笑）、2（男性、中性）、
3（男性、微笑）或 4（无条件）。省略任一参数时使用构建时的默认值。
运行时更换 seed 或 class 无需重建或重新导出。
点击 **Generate** 后，窗口从初始 latent 解码画面 `0/8` 开始，
依次显示同一条采样轨迹，最后停在 `8/8`。点击 **Replay** 可重播。
每次运行的 RGB 帧和 FVP 日志保存在 `build/logs/pico_face_logs/fvp/preview/run-*/`。
预览会额外执行八次中间解码，预览运行耗时包含这些调用和中间帧导出。

### 画质版 FVP 演示和 FPGA 单张测试

从 [Pico-Faces 固定提交](https://github.com/cpldcpu/pico-faces/tree/ee15d9d183d8428efaeb3078edf47669cfc23994/checkpoints/m3_decD_deep_full)
获取 `dit_qat.pt`、`vae_final.pt`、`latent_stats.npz`，放在
`resources_downloaded/pico_face/checkpoints/m3_decD_deep_full/`，或通过
`-Checkpoints` 指定其父目录。导出时会核对 checkpoint 的 SHA-256。
画质版使用独立的模型和构建目录。以下两种固件共用 Dedicated SRAM 384 KiB PTE。

```powershell
$variant = 'm3_decD_deep_full'
.\scripts\windows\build.ps1 -Profile PicoFace -Action Export -Variant $variant
.\scripts\windows\build.ps1 -Profile PicoFace -Action Build -Variant $variant -Preview
.\scripts\windows\run-fvp.ps1 -Profile PicoFace -Variant $variant -Preview -Seed 3

# 8 步主测固件，支持计时结束后通过 UART 取图。
.\scripts\windows\build.ps1 -Profile PicoFace -Action Build -Variant $variant -Target FPGA -Steps 8 -UartImage

# 4 步独立构建目录，保留两套可部署包。
.\scripts\windows\build.ps1 -Profile PicoFace -Action Build -Variant $variant -Target FPGA -Steps 4 -UartImage `
    -BuildDirectory build/pico_face/quality-fpga-4steps
```

FVP 预览检查 `0/8` 至 `8/8` 的九帧和最终 RGB。FPGA 上每次完整启动生成一张，
8 步应报告 16 次 DiT 和 1 次 decode，4 步应报告 8 次 DiT 和 1 次 decode。
要统计重复性，每档完整启动 5 次，记录 `Timing core clock`、`Image generation (noise to RGB)`、
`First image`、每次调用的 NPU cycles、内存池峰值和 CRC，汇报耗时中位数及范围。
速度版用相同参数复测。UART 传输在生成计时结束后进行。
比较最终图时，先确认两端的模型哈希、seed、步数、class 和 guidance 相同。

2026-11-12 在同一台 MPS4 FI101 上取得如下单次样本。四组均为 seed 3、
class 1、guidance 4、Dedicated 384 KiB，毫秒按日志的 50 MHz 回退时钟换算。

| 模型 | 步数 | 噪声到 RGB | 含初始化的首图 | DiT 平均 NPU cycles | Decoder NPU cycles | RGB CRC32 |
|---|---:|---:|---:|---:|---:|---|
| 速度版 | 4 | 432.978 ms | 471.498 ms | 1,890,516.63 | 314,909 | `5557eceb` |
| 画质版 | 4 | 606.858 ms | 645.356 ms | 2,802,873.13 | 748,073 | `57913c38` |
| 速度版 | 8 | 799.396 ms | 837.913 ms | 1,890,523.69 | 316,828 | `0223caac` |
| 画质版 | 8 | 1,136.712 ms | 1,175.220 ms | 2,803,087.25 | 748,341 | `92277e03` |

四组 method pool 峰值都是 263,724 字节。临时池峰值为速度版 524,288 字节、
画质版 786,432 字节。画质版 8 步最终 RGB 在 FVP 预览、FVP 单次解码和
FPGA UART 取图三条路径中逐字节一致。当前 MCC `RESET` 未重新触发应用输出，
表中每档来自一次 MCC `REBOOT` 后的有效样本。

### 构建 FPGA：Dedicated 或 Shared、8 步

Dedicated 可以复用前面导出的模型。Shared 必须单独导出，不能混用两种模式的 PTE。

```powershell
# Dedicated：384 KiB NPU SRAM 快速区，开启串口图片服务。
.\scripts\windows\build.ps1 -Profile PicoFace -Action Build -Target FPGA `
    -MemoryMode Dedicated_Sram -UartImage -Seed 3 -Steps 8 -ClassId 1 -Guidance 4

# Shared：仅首次需要导出该模式的模型。
.\scripts\windows\build.ps1 -Profile PicoFace -Action Export -MemoryMode Shared_Sram
.\scripts\windows\build.ps1 -Profile PicoFace -Action Build -Target FPGA `
    -MemoryMode Shared_Sram -UartImage -Seed 3 -Steps 8 -ClassId 1 -Guidance 4
```

两种固件分别输出到以下目录，构建脚本也会生成对应 ZIP：

```text
build/pico_face/pte/atfe/fpga/
  dedicated_sram/384k-release-default-pmu/board-package/
  shared_sram/no-cache-release-default-pmu/board-package/
```

检查包内 `README.txt` 的 seed、步数和内存模式。将 `SOFTWARE` 下的三个 BIN 放到板卡的
`SOFTWARE` 目录，并按照现有已验证用例，将包内 `images.txt` 放到实际启用的
Corstone-320 配置目录；不要覆盖 FPGA bitstream。此脚本只构建和打包，不自动烧录板卡。
非默认参数会写入 ZIP 文件名；`board-package` 则始终对应最近一次打包的配置。

两种模式的模型、method pool、采样状态和最终 RGB 都在 DDR。Dedicated 的临时池在
`0x72000000`，容量 4 MiB，另有 384 KiB SRAM 快速区；Shared 的临时池在
`0x31000000`，容量 2 MiB。容量是预留上限，不是实际使用量；实测峰值见下表。
本用例复用现有板级 UART、NPU、TA 和启动配置，没有新增寄存器地址定义。

### 从串口取回图片

串口为 115200、8N1、无流控。等待 `PICO_FACE_PASS` 和 `SERIAL_IMAGE_READY` 后，
关闭占用应用串口的终端，再执行以下命令，COM5 替换成实际端口：

```powershell
.\scripts\windows\receive-pico-face.ps1 -ListPorts
.\scripts\windows\receive-pico-face.ps1 -Port COM5

# 可选：指定固定文件名前缀；这个旧写法也只会写入专用 out 目录。
.\scripts\windows\receive-pico-face.ps1 -Port COM5 -Output out/face
```

接收脚本发送 `I`，仅取回已经生成的图像，不重新推理。校验 CRC 后保存 `.rgb`、`.png`、
`.json`，同时保留原始 `.pfframe`。不传 `-Output` 会自动使用 seed、步数和时间戳命名；
显式指定固定前缀会覆盖该前缀的旧文件。图片输出均限制在下面的 `out` 目录内：

```text
build/logs/pico_face_logs/
  build/   构建与导出日志，自动保存
  fvp/     FVP 运行日志，自动保存
  fpga/    实板串口日志，由串口终端或用户保存
  out/     串口图片、RGB、原始帧和元数据
```

如果需要先启动接收脚本再复位板卡，可加 `-WaitReady`；已经看到就绪消息后再接收时不要加。
图像传输在 115200 波特率下约需 4.3 秒，这部分不计入图片生成耗时。

### 参数选择、计时与验证结果

普通 FVP 和 FPGA 的 `-Seed`、`-Steps`、`-ClassId`、`-Guidance` 都是构建参数。修改后重新执行 `Build`；
只给 `Run` 或接收脚本传参数不会改变已编译的采样设置。步数支持 1、2、4、8，
改变这些步数不需要重新导出 PTE。建议固定 class 和 guidance，先比较相同 seed 的
4/8 步，再改变 seed。更多步数不保证所有图片都更清晰。

2026-10-22 的实板结果如下，均为 seed 3、8 步、class 1、guidance 4，RGB CRC32 均为
`0223caac`，说明两种模式输出一致：

| 项目 | Dedicated 384 KiB | Shared |
|---|---:|---:|
| 图片生成端到端 | 798.981 ms | 804.300 ms |
| 包含模型初始化的首张耗时 | 837.499 ms | 842.801 ms |
| DiT 平均 NPU TOTAL | 1,890,508.06 cycles | 1,909,315.63 cycles |
| DiT 平均 CPU TOTAL | 2,274,849.00 cycles | 2,292,383.75 cycles |
| Method pool 实测峰值 | 263,724 字节 | 263,724 字节 |
| Temp pool 实测峰值 | 524,288 字节 | 527,056 字节 |

DiT 平均值统计全部 16 次调用，不包含 decode。CPU TOTAL 包含等待 NPU 的时间，
不能与 NPU TOTAL 相加。图片生成端到端从初始噪声开始，到最终 RGB 准备完成为止，
包含采样和最终 decode，不包含模型初始化、结果日志打印和串口传图；每次调用的 cycles
在生成结束后分别打印。实板毫秒值按当前日志的 50 MHz 时钟回退值换算。

本次 Shared 的 decode 更快，但 DiT 累计耗时略长，最终生成耗时比 Dedicated 高约 0.67%。
两种模式还会影响 Vela 的内存规划和调度，不能仅凭临时区位于 SRAM 就断定整体更快。

算法与串口协议的 native 测试命令见前文。`compare` 工具可生成多个 seed 的 4/8 步
主机参考图，默认位于 `build/logs/pico_face_logs/out/comparison`；这些是主机量化模拟结果，
不代表 FPGA 执行或 FPGA 性能。上板文件、PTE、图片和日志均不应随 feature commit 入库。
