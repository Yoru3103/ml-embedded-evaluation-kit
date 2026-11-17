# Windows camera and Corstone-320 VSI demonstrations

## Native Windows

Run the PowerShell commands below from the Windows MLEK checkout. The builds use
Arm Toolchain for Embedded and Corstone-320 with Ethos-U85-1024 and 384 KiB of
dedicated SRAM cache. For a fresh checkout, start with the
[Windows static-image smoke test](../../docs/sections/windows.md).

| `-Demo` | Model and default location |
| --- | --- |
| `gesture` | 33-class YOLO, 320 px, `resources_downloaded/gesture_detection/exported_33_fulltrain_320/pte_Dedicated_Sram/gesture_33.pte` |
| `coco` | COCO YOLO, 256 px, `resources_downloaded/coco_detection/tflm_Dedicated_Sram/best_int8_vela.tflite` |
| `img-class` | MobileNetV2, 224 px, supplied by the Windows TFLM resource setup |

The gesture and COCO models are external inputs. Supply them at the default locations
or pass `-ModelPath` with the appropriate PTE or compiled TFLite file. For gesture,
pass `-LabelsPath` with the checkpoint-order labels or place `labels.txt` in
`exported_33_fulltrain_320`. Use models compiled for U85-1024, `Dedicated_Sram`,
and a 384 KiB NPU cache. See [gesture export](../../docs/sections/gesture_all_deployment.md)
and [COCO model provenance and compilation](../../resources/coco_detection/README.md).

Use `-Action All` on first use to prepare the selected build environment, install
the host camera package, and build the firmware. For later builds, omit the action.

```powershell
.\scripts\windows\build.ps1 -Profile Vision -Demo gesture -Action All
.\scripts\windows\build.ps1 -Profile Vision -Demo coco -Action All
.\scripts\windows\build.ps1 -Profile Vision -Demo img-class -Action All
.\scripts\windows\build.ps1 -Profile Vision -Demo img-class -Mode Camera -Action All
```

Supply `-CompilerBin` with the ATfE `bin` directory if discovery fails.

For VSI with a live camera, the launcher opens **VSI camera preview** and starts
the local video server on port 6000. The preview shows the live camera and the
last frame sent to FVP. Install the 64-bit CPython version matching the FVP
runtime, typically Python 3.9 for Corstone-320 FVP 11.27. FVP uses its bundled
interpreter with the socket extensions from that installation. Pass
`-FvpPythonExtensions` if its `DLLs` directory is
outside `%LOCALAPPDATA%/Programs/Python/Python39`. Run one VSI demo per port,
and wait for the firmware's `Press KEY0` prompt. Click **KEY0** in the board
window titled **MLEK <demo> - KEY0**. The simulated button stays pressed after
the first click. Click it again to release it, then click to capture the next
frame. Results appear in the separate FVP LCD window and the build log.

For img_class without VSI, use `-Mode Camera` and press **Space** in the Windows
camera preview.

```powershell
.\scripts\windows\run-fvp.ps1 -Profile Vision -Demo gesture
.\scripts\windows\run-fvp.ps1 -Profile Vision -Demo coco
.\scripts\windows\run-fvp.ps1 -Profile Vision -Demo img-class
.\scripts\windows\run-fvp.ps1 -Profile Vision -Demo img-class -Mode Camera
```

Use `-CameraIndex` to select a different device or `-VsiPort` to select an
available port if port 6000 is in use. For an unattended VSI
inference check, build with `-AutoCapture` and run with
`-BuildDirectory build/fvp-<demo>/vsi-auto`, `-ImagePath`, `-Headless` and
`-SmokeTimeoutSeconds 180`. For semihosting, pass `-ImagePath`, `-Headless`
and `-SmokeTimeoutSeconds 180` to the `-Mode Camera` launcher.

To trigger VSI capture from the preview, build the demo with `-AutoCapture`
and start it with `-CaptureFromPreview`. Press **Space** in **VSI camera preview**
to send one frame to FVP. This is available for gesture, COCO and img_class.

Use `-Fast` for live demonstrations. The build disables the Ethos-U timing
adapter and the launcher enables the functional NPU model. The gesture
firmware goes in `build/fvp-gesture/vsi-fast`. KEY0 opens the board and HDLCD
windows alongside the camera preview.

```powershell
.\scripts\windows\build.ps1 -Profile Vision -Demo gesture -Fast
.\scripts\windows\run-fvp.ps1 -Profile Vision -Demo gesture -Fast -VsiPort 6133
```

Use the timing-enabled build for NPU cycle and memory timing measurements.

### If the FVP board panel stops accepting KEY0 clicks

In Windows Corstone-320 FVP 11.27, HDLCD input events can block the shared SDL queue
when both FVP windows are open. The optional `Key0Runtime` profile prepares a local
copy of the installed FVP that uses the project's SDL event filter and forwards
graphics calls to the 64-bit SDL2 DLL from Arm Development Studio. Supply the paths
from your installation. Run the command from an x64 Visual Studio Native Tools
shell so the profile can find `cl.exe`, or supply `-HostCompiler` with the path to a
host LLVM `clang.exe` that can compile for x86-64 Windows. `-CompilerBin` selects
the ATfE bin directory for the linker and PE inspection tool:

```powershell
.\scripts\windows\build.ps1 -Profile Key0Runtime `
    -FvpExecutable 'C:/Program Files/Arm/FVP_Corstone_SSE-320/models/Win64_VC2019/FVP_Corstone_SSE-320.exe' `
    -SdlDll 'D:/Arm/Development Studio/sw/models/bin/SDL2.dll' `
    -CompilerBin 'D:/Arm/Development Studio/sw/atfe/bin'
```

The default output is `resources_downloaded/windows/corstone-320-key0`.
In KEY0 mode the launcher uses this local copy automatically when neither
`-FvpExecutable` nor `FVP_PATH` is set. If you specify a different output
directory, pass the executable path printed by the build to `-FvpExecutable`.

### Fixed-image timing benchmark

After the first Windows setup, build a static-image benchmark, then run the
timing-enabled FVP once. The following example uses the bundled cat image:

```powershell
.\scripts\windows\build.ps1 -Profile VisionBenchmark -Demo img-class -Target FVP
.\scripts\windows\run-fvp.ps1 -Profile VisionBenchmark -Demo img-class -Headless
```

Use `-Target MPS4` to build and package board firmware, or `-Target Both` for
both targets. The output directory is `build/vision-benchmark/<demo>/`.
For gesture or COCO, first prepare the matching model and software environment
as described above. Use `-InputPath` to evaluate a different image.

## WSL FVP

Start the Windows `vsi_video_server.py` preview server on TCP port 6000, then run
one launcher from the MLEK root. Each launcher starts an already-built U85-1024
firmware with the board button panel and HDLCD enabled. Press KEY0 to capture
and infer one frame; the result remains on the LCD until the next KEY0 press.

| Demonstration | Command | Default firmware |
| --- | --- | --- |
| Image classification | `bash scripts/vsi/run_img_class_fvp320.sh` | `build-img-class-vsi/bin/mlek_img_class.axf` |
| 10-class gesture detection | `bash scripts/vsi/run_gesture_fvp320.sh` | `build-img-class-vsi/bin/mlek_yolov8_detection.axf` |
| 80-class COCO detection | `bash scripts/vsi/run_coco_fvp320.sh` | `build-coco-vsi/bin/mlek_yolov8_detection.axf` |

The two YOLO executables have the same filename in different build directories;
the launchers select the corresponding model. Run one demonstration at a time.
Exit and restart the Windows server before reconnecting another FVP session,
since the current VSI client closes the server connection on exit.

All launchers accept these environment overrides:

| Variable | Purpose |
| --- | --- |
| `VSI_VIDEO_HOST` | Windows server IP; defaults to the IPv4 gateway for WSL NAT |
| `VSI_PYTHON_PATH` | Directory containing `arm_vsi4.py` and the modified `vsi_video.py` |
| `FVP_PATH` | Corstone-320 FVP executable |
| `IMG_CLASS_BUILD_DIR` | Image-classification firmware build directory |
| `GESTURE_BUILD_DIR` | Gesture firmware build directory |
| `COCO_BUILD_DIR` | COCO firmware build directory |

With mirrored WSL networking, explicitly set `VSI_VIDEO_HOST=127.0.0.1`.
Additional command-line arguments are forwarded to the FVP, for example
`bash scripts/vsi/run_gesture_fvp320.sh --stat`.
